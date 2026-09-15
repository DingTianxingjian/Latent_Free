"""LIBERO evaluation of the CONTINUOUS-latent arm (finetune_libero_ctrl.py --latent acdc_cont).

The token arms are evaluated by UniVLA's own script (run_libero_eval_ctrl.py): the VLM GENERATES latent tokens and the
decoder reads their hidden states.  This arm has no latent vocabulary, so the VLM is run forward once per step and the
same heads that were trained (MAP pooling -> continuous code -> action) produce the chunk.  Everything else is kept
identical to their loop: same prompt, same image pipeline, same per-step query with temporal ensembling of the
overlapping chunks, same gripper conventions, same max_steps, 50 trials per task.

  python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint <run_dir> --task_suite_name libero_spatial
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from PIL import Image

from ...paths import UNIVLA_ROOT
sys.path.insert(0, UNIVLA_ROOT); sys.path.insert(0, os.path.join(UNIVLA_ROOT, "experiments", "robot", "libero"))
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env, get_libero_image, get_libero_dummy_action
from experiments.robot.robot_utils import set_seed_everywhere, normalize_gripper_action, invert_gripper_action
from .finetune_libero_ctrl import ContHeads

MAX_STEPS = {"libero_spatial": 240, "libero_object": 300, "libero_goal": 320, "libero_10": 550, "libero_90": 420}


class Ensembler:
    """UniVLA's temporal aggregation of overlapping chunks (run_libero_eval.py:118-165), verbatim in behaviour."""
    def __init__(self, window):
        self.w = window
        self.mask = torch.flip(torch.triu(torch.ones(window, window, dtype=torch.bool)), dims=[1]).numpy()
        self.weights = np.array([np.exp(-0.1 * i) for i in range(window)])[:, None]
        self.reset()

    def reset(self):
        self.buf = np.zeros((self.w, self.w, 7)); self.buf_mask = np.zeros((self.w, self.w), dtype=bool)

    def __call__(self, chunk, mask, low, high):
        self.buf[1:] = self.buf[:-1]; self.buf_mask[1:] = self.buf_mask[:-1]
        self.buf[:, :-1] = self.buf[:, 1:]; self.buf_mask[:, :-1] = self.buf_mask[:, 1:]
        self.buf_mask = self.buf_mask * self.mask
        self.buf[0] = chunk; self.buf_mask[0] = True
        a = (self.buf[:, 0, :] * self.buf_mask[:, 0:1] * self.weights).sum(0) / (self.buf_mask[:, 0:1] * self.weights).sum()
        return np.where(mask, 0.5 * (a + 1) * (high - low) + low, a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pretrained_checkpoint", required=True); ap.add_argument("--task_suite_name", default="libero_spatial")
    ap.add_argument("--num_trials_per_task", type=int, default=50); ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--local_log_dir", default="acdc_la/results/vla_eval"); ap.add_argument("--run_id_note", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(); set_seed_everywhere(a.seed); dev = "cuda"
    meta = json.load(open(os.path.join(a.pretrained_checkpoint, "ctrl_meta.json")))
    assert meta["latent"] == "acdc_cont", meta["latent"]
    W = int(meta["window_size"]); dz = int(meta["dz"])

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(a.pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(a.pretrained_checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 trust_remote_code=True, attn_implementation="sdpa").to(dev).eval()
    vla.norm_stats = json.load(open(os.path.join(a.pretrained_checkpoint, "dataset_statistics.json")))
    heads = ContHeads(dz, W).to(dev); heads.load_state_dict(torch.load(os.path.join(a.pretrained_checkpoint, "action_decoder.pt"), map_location="cpu")); heads.eval()
    n_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    unnorm_key = a.task_suite_name if a.task_suite_name in vla.norm_stats else f"{a.task_suite_name}_no_noops"
    stats = vla.get_action_stats(unnorm_key); mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    low, high = np.array(stats["q01"]), np.array(stats["q99"])

    suite = benchmark.get_benchmark_dict()[a.task_suite_name]()
    os.makedirs(a.local_log_dir, exist_ok=True)
    out = {"checkpoint": a.pretrained_checkpoint, "suite": a.task_suite_name, "trials": a.num_trials_per_task, "per_task": {}}
    ens = Ensembler(W); t0 = time.time(); tot = succ = 0
    for ti in range(suite.n_tasks):
        task = suite.get_task(ti); inits = suite.get_task_init_states(ti)
        env, desc = get_libero_env(task, "openvla", resolution=256)
        pb0 = PurePromptBuilder("openvla"); pb0.add_turn("human", f"What action should the robot take to {desc.lower()}?"); pb0.add_turn("gpt", "")
        prompt = pb0.get_prompt()
        tsucc = 0
        for ep in range(a.num_trials_per_task):
            env.reset(); ens.reset(); obs = env.set_init_state(inits[ep]); t, done = 0, False
            while t < MAX_STEPS[a.task_suite_name] + a.num_steps_wait and not done:
                if t < a.num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action("openvla")); t += 1; continue
                img = Image.fromarray(get_libero_image(obs, 224)).convert("RGB")
                inputs = processor(prompt, img)
                inputs = {k: (v.to(dev, dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to(dev)) for k, v in inputs.items()}
                with torch.no_grad():
                    h = vla(**inputs, output_hidden_states=True).hidden_states[-1]
                    chunk = heads(h[:, :n_patches].float(), h[:, n_patches:].float())[0].reshape(W, 7).cpu().numpy()
                act = ens(chunk, mask, low, high)
                act = invert_gripper_action(normalize_gripper_action(act, binarize=True))
                obs, _, done, _ = env.step(act.tolist()); t += 1
            tsucc += int(done); tot += 1; succ += int(done)
        env.close(); out["per_task"][desc] = tsucc / a.num_trials_per_task
        print(f"[cont-eval] task {ti} {desc[:44]}: {tsucc}/{a.num_trials_per_task}  running total {succ}/{tot} ({time.time()-t0:.0f}s)", flush=True)
        out["success"] = succ / tot
        json.dump(out, open(os.path.join(a.local_log_dir, f"cont_{a.task_suite_name}{('_' + a.run_id_note) if a.run_id_note else ''}.json"), "w"), indent=1)
    print(f"[cont-eval] Current total success rate: {succ / tot:.3f} ({succ}/{tot})", flush=True)


if __name__ == "__main__":
    main()
