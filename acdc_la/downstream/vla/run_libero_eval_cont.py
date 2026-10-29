"""LIBERO evaluation of the CONTINUOUS-latent arms (finetune_libero_ctrl.py --latent acdc_cont | tc_cont | none).

The token arms are evaluated by UniVLA's own script (run_libero_eval_ctrl.py): the VLM GENERATES latent tokens and the
decoder reads their hidden states.  This arm has no latent vocabulary, so the VLM is run forward once per step and the
same heads that were trained (MAP pooling -> continuous code -> action) produce the chunk.  Everything else is kept
identical to their loop: same prompt, same image pipeline, same per-step query with temporal ensembling of the
overlapping chunks, same gripper conventions, same max_steps, 50 trials per task.

  python -m acdc_la.downstream.vla.run_libero_eval_cont --pretrained_checkpoint <run_dir> --task_suite_name libero_spatial
"""
import os, re, sys, json, argparse, time
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
from .finetune_libero_ctrl import ContHeads, FlowHeads
from experiments.robot.openvla_utils import crop_and_resize
import tensorflow as tf
try: tf.config.set_visible_devices([], "GPU")                 # TF only does the crop below; keep it off the policy's GPU
except RuntimeError: pass                                     # already initialised (and hidden) by the RLDS import


def center_crop(img, crop_scale=0.9):
    """UniVLA's get_vla_latent_action(center_crop=True), verbatim in behaviour (openvla_utils.py:182-203).  The token arms
    are evaluated WITH it (their script asserts center_crop because training uses random-crop augmentation); this
    evaluator shipped without it until 2026-09-16, which would have fed the continuous arms an uncropped image that no
    training batch ever looked like."""
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(img)), tf.float32)
    x = tf.clip_by_value(crop_and_resize(x, crop_scale, 1), 0, 1)
    return Image.fromarray(tf.image.convert_image_dtype(x, tf.uint8, saturate=True).numpy()).convert("RGB")

MAX_STEPS = {"libero_spatial": 240, "libero_object": 300, "libero_goal": 320, "libero_10": 550, "libero_90": 420}


def max_steps(suite):
    """LIBERO-PRO suites (libero_goal_swap, libero_10_lan, ...) keep their base suite's step budget (LIBERO-PRO README)."""
    return MAX_STEPS[suite] if suite in MAX_STEPS else MAX_STEPS[next(b for b in MAX_STEPS if suite.startswith(b + "_"))]


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
    ap.add_argument("--flow_steps", type=int, default=0, help="flow head: Euler steps at evaluation (0 = the value in ctrl_meta.json)")
    a = ap.parse_args(); set_seed_everywhere(a.seed); dev = "cuda"
    meta = json.load(open(os.path.join(a.pretrained_checkpoint, "ctrl_meta.json")))
    assert meta["latent"] in ("acdc_cont", "tc_cont", "none", "univla_cont"), meta["latent"]     # every arm trained through ContHeads
    W = int(meta["window_size"]); dz = int(meta["dz"])

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(a.pretrained_checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(a.pretrained_checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 trust_remote_code=True, attn_implementation="sdpa").to(dev).eval()
    vla.norm_stats = json.load(open(os.path.join(a.pretrained_checkpoint, "dataset_statistics.json")))
    flow = meta.get("head", "reg") == "flow"; fk = meta.get("flow", {})
    heads = (FlowHeads(dz, W, width=fk["width"], layers=fk["layers"], z_tok=fk["z_tok"]) if flow
             else ContHeads(dz, W, text_direct=(meta.get("head") == "reg_text"))).to(dev)
    heads.load_state_dict(torch.load(os.path.join(a.pretrained_checkpoint, "action_decoder.pt"), map_location="cpu")); heads.eval()
    steps = int(a.flow_steps or fk.get("steps", 10)) if flow else 0
    print(f"[cont-eval] head {'flow, ' + str(steps) + ' Euler steps' if flow else 'regression, text_direct=' + str(meta.get('head') == 'reg_text')}", flush=True)
    n_patches = vla.vision_backbone.featurizer.patch_embed.num_patches
    base = a.task_suite_name if a.task_suite_name in MAX_STEPS else next(b for b in MAX_STEPS if a.task_suite_name.startswith(b + "_"))
    # LIBERO-PRO suites (libero_goal_swap, ...) act with the statistics of the base suite the policy was trained on
    unnorm_key = next(k for k in (a.task_suite_name, f"{a.task_suite_name}_no_noops", base, f"{base}_no_noops") if k in vla.norm_stats)
    stats = vla.get_action_stats(unnorm_key); mask = np.array(stats.get("mask", np.ones_like(stats["q01"], dtype=bool)))
    low, high = np.array(stats["q01"]), np.array(stats["q99"])

    suite = benchmark.get_benchmark_dict()[a.task_suite_name]()
    os.makedirs(a.local_log_dir, exist_ok=True)
    out = {"checkpoint": a.pretrained_checkpoint, "suite": a.task_suite_name, "trials": a.num_trials_per_task, "per_task": {}}
    ens = Ensembler(W); t0 = time.time(); tot = succ = 0
    ids = os.environ.get("LIBERO_TASK_IDS")                      # see the note in run_libero_eval_ctrl.py
    ids = [int(x) for x in re.split(r"[,_]", ids)] if ids else list(range(suite.n_tasks))   # "_" as well as "," : sbatch --export splits on commas
    shard = os.environ.get("LIBERO_TASK_SHARD")                  # "k/n": every n-th task from k (LIBERO-Plus: ~2,500 tasks per suite)
    if shard: k, n = map(int, shard.split("/")); ids = ids[k::n]
    assert all(0 <= i < suite.n_tasks for i in ids), (ids, suite.n_tasks)
    out["task_ids"] = ids
    out_name = "cont_" + a.task_suite_name + (f"_{a.run_id_note}" if a.run_id_note else "")
    if os.environ.get("LIBERO_TASK_IDS"): out_name += "_t" + "_".join(str(i) for i in ids)
    if shard: out_name += f"_s{k}of{n}"
    out["per_task_id"] = {}                                      # keyed by task index: LIBERO-Plus variants share their instruction
    if os.environ.get("LIBERO_PAD_TO"): out_name += "_pad" + os.environ["LIBERO_PAD_TO"]
    out_name += ".json"
    # LIBERO_RESUME=1 (2026-10-29, off-Newton LIBERO-Plus runs): keep the tasks an earlier run of the same shard finished
    if os.environ.get("LIBERO_RESUME") and os.path.exists(os.path.join(a.local_log_dir, out_name)):
        prev = json.load(open(os.path.join(a.local_log_dir, out_name)))
        if prev.get("task_ids") == ids and prev.get("trials") == a.num_trials_per_task:
            out["per_task"], out["per_task_id"] = prev.get("per_task", {}), prev.get("per_task_id", {})
            tot = len(out["per_task_id"]) * a.num_trials_per_task; succ = round(sum(out["per_task_id"].values()) * a.num_trials_per_task)
            print(f"[cont-eval] resuming {out_name}: {len(out['per_task_id'])}/{len(ids)} tasks already done", flush=True)
    # LIBERO_PAD_TO=max (2026-09-18, diagnostic): training batches right-pad shorter prompts to the batch's longest, and the
    # heads pool over ALL text positions, pads included (pad states attend to the whole instruction).  Evaluation has no
    # pads.  This pads each prompt to the suite's longest prompt, the typical training condition, to test whether that
    # train/test mismatch is what makes the continuous arms instruction-blind.
    pad_to = 0
    if os.environ.get("LIBERO_PAD_TO"):
        def _plen(d):
            pb = PurePromptBuilder("openvla"); pb.add_turn("human", f"What action should the robot take to {d.lower()}?"); pb.add_turn("gpt", "")
            return processor(pb.get_prompt(), Image.new("RGB", (224, 224)))["input_ids"].shape[-1]
        pad_to = max(_plen(suite.get_task(i).language) for i in range(suite.n_tasks)) if os.environ["LIBERO_PAD_TO"] == "max" else int(os.environ["LIBERO_PAD_TO"])
        out["pad_to"] = pad_to; print(f"[cont-eval] prompts right-padded to {pad_to} tokens (training-like)", flush=True)
    for ti in ids:
        if str(ti) in out["per_task_id"]: continue                 # LIBERO_RESUME: done already
        task = suite.get_task(ti); inits = suite.get_task_init_states(ti)
        env, desc = get_libero_env(task, "openvla", resolution=256)
        nq = int(meta.get("query_tokens") or 0)
        pb0 = PurePromptBuilder("openvla"); pb0.add_turn("human", f"What action should the robot take to {desc.lower()}?")
        pb0.add_turn("gpt", "".join(f"<ACT_{i}>" for i in range(nq)))                  # same placeholders as training
        prompt = pb0.get_prompt()
        tsucc = 0
        for ep in range(a.num_trials_per_task):
            env.reset(); ens.reset(); obs = env.set_init_state(inits[ep]); t, done = 0, False
            gen = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + ti * 1_000 + ep)   # flow noise: fixed per episode
            while t < max_steps(a.task_suite_name) + a.num_steps_wait and not done:
                if t < a.num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action("openvla")); t += 1; continue
                img = center_crop(Image.fromarray(get_libero_image(obs, 224)).convert("RGB"))
                inputs = processor(prompt, img)
                if pad_to and inputs["input_ids"].shape[-1] < pad_to:
                    n = pad_to - inputs["input_ids"].shape[-1]
                    inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.full((1, n), processor.tokenizer.pad_token_id, dtype=inputs["input_ids"].dtype)], 1)
                    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.zeros((1, n), dtype=inputs["attention_mask"].dtype)], 1)
                inputs = {k: (v.to(dev, dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to(dev)) for k, v in inputs.items()}
                with torch.no_grad():
                    h = vla(**inputs, output_hidden_states=True).hidden_states[-1]
                    vis, txt = h[:, :n_patches].float(), h[:, n_patches:].float()
                    if nq:                                                             # read the query-token states
                        qm = inputs["input_ids"][0] > 32000; txt = txt[:, qm]; assert txt.shape[1] == nq, txt.shape
                    chunk = (heads.sample(vis, txt, steps, gen) if flow else heads(vis, txt))[0].reshape(W, 7).cpu().numpy()
                act = ens(chunk, mask, low, high)
                act = invert_gripper_action(normalize_gripper_action(act, binarize=True))
                obs, _, done, _ = env.step(act.tolist()); t += 1
            tsucc += int(done); tot += 1; succ += int(done)
        env.close(); out["per_task"][desc] = tsucc / a.num_trials_per_task; out["per_task_id"][str(ti)] = tsucc / a.num_trials_per_task
        print(f"[cont-eval] task {ti} {desc[:44]}: {tsucc}/{a.num_trials_per_task}  running total {succ}/{tot} ({time.time()-t0:.0f}s)", flush=True)
        out["success"] = succ / tot
        json.dump(out, open(os.path.join(a.local_log_dir, out_name), "w"), indent=1)
    print(f"[cont-eval] Current total success rate: {succ / max(tot, 1):.3f} ({succ}/{tot})", flush=True)


if __name__ == "__main__":
    main()
