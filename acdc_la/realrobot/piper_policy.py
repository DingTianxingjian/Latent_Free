"""Deployment wrapper for the Piper real-robot policy (2026-11-08) -- the inference loop that goes with the
`piper_pick_can_10k` checkpoint on HF `Potestates/TERRA`.

There is no simulator for this data, so nothing here can be checked by rollout.  What CAN be checked is that the
wrapper feeds the policy exactly what training fed it, and `--replay` does that: it replays the held-out episodes
(`train[95%:]`, never trained on) out of the RLDS and prints the action error in DEGREES.  Run that first; if its
numbers are far off the ones in the module docstring of acdc_la/realrobot/eval_heldout.py, the preprocessing is
wrong and the robot numbers would be meaningless.

Everything that has to agree with training, with the line it comes from:
  * 10 Hz.  `convert_piper_rlds.py` keeps every 3rd frame of 30 Hz, so a 12-step chunk spans 1.2 s.  Calling act()
    faster or slower than 10 Hz puts the policy off its training distribution.
  * ONE image, the FRONT camera.  `datasets_ctrl.py:67` reads `image_primary` only -- the wrist camera is in the
    dataset and this policy never sees it.
  * front frame -> 256x256 (cv2 INTER_AREA, aspect SQUASHED, RGB) as the converter stored it -> 224x224 (the RLDS
    `decode_and_resize`, `vla.config.image_sizes`) -> centre crop of 0.9 area.  Training ran with
    `random_resized_crop(scale=[0.9, 0.9], ratio=[1.0, 1.0])` (`datasets.py:363`), so evaluation must crop the same
    area deterministically; feeding the uncropped frame is a distribution shift.
  * prompt: PurePromptBuilder, "What action should the robot take to {instruction.lower()}?", then the nq
    `<ACT_i>` placeholders as the gpt turn -- identical to `run_libero_eval_cont.py` and to training.
  * action: 6 ABSOLUTE joint targets in degrees + gripper.  Dims 0-5 are un-normalised from q01/q99
    (`dataset_statistics.json`); the gripper dim is raw because its mask is false, and the dataset stores it as
    raw/8.2 clipped to [0, 1], so `gripper_to_raw()` takes it back to the arm's units.
  * NONE of LIBERO's gripper post-processing (`normalize_gripper_action` / `invert_gripper_action`) applies here:
    that is LIBERO's +1/-1 convention.  Piper's gripper is [0, 1] and passes through untouched.

  from acdc_la.realrobot.piper_policy import PiperPolicy
  p = PiperPolicy("<checkpoint dir>"); p.reset()
  while True:                                        # at 10 Hz
      a = p.act(front_frame_rgb, "pick up the can and place it on the plate")
      send_joint_targets(a[:6]); send_gripper(PiperPolicy.gripper_to_raw(a[6]))

  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -m acdc_la.realrobot.piper_policy \
      --checkpoint /data/ti874297/vla_runs/tc_cont+piper_pick_can+...--terra_piper --replay data/real_rlds
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image

GRIP_SCALE = 8.2          # convert_piper_rlds.py: stored gripper = raw / 8.2, clipped to [0, 1]
CONTROL_HZ = 10           # convert_piper_rlds.py: 30 Hz subsampled by 3


def _tf():
    import tensorflow as tf
    try: tf.config.set_visible_devices([], "GPU")      # TF only does the crop; keep it off the policy's GPU
    except RuntimeError: pass
    return tf


class Ensembler:
    """UniVLA's temporal aggregation of overlapping chunks, copied verbatim in behaviour from
    run_libero_eval_cont.py:50-66 rather than imported, because that module imports LIBERO (and mujoco) at the top
    level and a robot host has no reason to install either.  If one changes, change both."""
    def __init__(self, window, adim):
        self.w, self.adim = window, adim
        self.mask = torch.flip(torch.triu(torch.ones(window, window, dtype=torch.bool)), dims=[1]).numpy()
        self.weights = np.array([np.exp(-0.1 * i) for i in range(window)])[:, None]
        self.reset()

    def reset(self):
        self.buf = np.zeros((self.w, self.w, self.adim)); self.buf_mask = np.zeros((self.w, self.w), dtype=bool)

    def __call__(self, chunk, mask, low, high):
        self.buf[1:] = self.buf[:-1]; self.buf_mask[1:] = self.buf_mask[:-1]
        self.buf[:, :-1] = self.buf[:, 1:]; self.buf_mask[:, :-1] = self.buf_mask[:, 1:]
        self.buf_mask = self.buf_mask * self.mask
        self.buf[0] = chunk; self.buf_mask[0] = True
        a = (self.buf[:, 0, :] * self.buf_mask[:, 0:1] * self.weights).sum(0) / (self.buf_mask[:, 0:1] * self.weights).sum()
        return np.where(mask, 0.5 * (a + 1) * (high - low) + low, a)


class PiperPolicy:
    def __init__(self, checkpoint, device="cuda", flow_steps=None, unnorm_key=None, seed=7, ensemble=True):
        # ensemble=False sends chunk[0] straight through.  On this data the ensembling is slightly WORSE offline
        # (3.69 deg vs 3.24 deg mean per joint, 2026-11-08): it is UniVLA's LIBERO setting, where the action is a
        # delta and the lag it introduces costs nothing.  Here the action is an absolute joint target, so the lag
        # shows up directly.  Both are worth trying on the robot; offline cannot decide it.
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from ..downstream.vla.finetune_libero_ctrl import ContHeads, FlowHeads

        self.meta = json.load(open(os.path.join(checkpoint, "ctrl_meta.json")))
        assert self.meta["latent"] in ("tc_cont", "acdc_cont", "univla_cont", "none"), self.meta["latent"]
        self.W = int(self.meta["window_size"]); dz = int(self.meta["dz"]); self.nq = int(self.meta.get("query_tokens") or 0)
        AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                          trust_remote_code=True, attn_implementation="sdpa").to(device).eval()
        self.flow = self.meta.get("head", "reg") == "flow"; fk = self.meta.get("flow", {})
        self.heads = (FlowHeads(dz, self.W, width=fk["width"], layers=fk["layers"], z_tok=fk["z_tok"]) if self.flow
                      else ContHeads(dz, self.W, text_direct=(self.meta.get("head") == "reg_text"))).to(device)
        self.heads.load_state_dict(torch.load(os.path.join(checkpoint, "action_decoder.pt"), map_location="cpu")); self.heads.eval()
        self.flow_steps = int(flow_steps or fk.get("steps", 10)) if self.flow else 0

        stats = json.load(open(os.path.join(checkpoint, "dataset_statistics.json")))
        if unnorm_key is None:
            assert len(stats) == 1, f"several datasets in dataset_statistics.json, pass unnorm_key: {list(stats)}"
            unnorm_key = next(iter(stats))
        s = stats[unnorm_key]["action"]
        self.low, self.high = np.array(s["q01"]), np.array(s["q99"])
        self.norm_mask = np.array(s.get("mask", np.ones_like(self.low, dtype=bool)))
        self.adim = len(self.low)
        self.n_patches = self.vla.vision_backbone.featurizer.patch_embed.num_patches
        self.ens = Ensembler(self.W, self.adim)
        self.seed = seed; self.use_ens = ensemble; self.reset()
        print(f"[piper-policy] {checkpoint}\n  {'flow, ' + str(self.flow_steps) + ' Euler steps' if self.flow else 'regression'}"
              f" | window {self.W} | action dim {self.adim} | query tokens {self.nq} | unnorm '{unnorm_key}'"
              f" | raw gripper dim(s) {list(np.nonzero(~self.norm_mask)[0])}", flush=True)

    # ---------------------------------------------------------------- per-episode state
    def reset(self, episode=0):
        """Call at the start of every episode: clears the chunk history and fixes the flow head's noise for the
        episode, the way the LIBERO evaluator seeds it per episode."""
        self.ens.reset()
        self.gen = torch.Generator(device=self.device).manual_seed(self.seed * 1_000_003 + int(episode))
        self._prompt = self._instr = None

    def _build_prompt(self, instruction):
        from prismatic.models.backbones.llm.prompting import PurePromptBuilder
        pb = PurePromptBuilder("openvla")
        pb.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        pb.add_turn("gpt", "".join(f"<ACT_{i}>" for i in range(self.nq)))
        return pb.get_prompt()

    # ---------------------------------------------------------------- image
    def preprocess(self, image):
        """front camera frame (RGB, any size) -> the 224x224 PIL image training produced.  `image` may be an
        HxWx3 uint8 array or a PIL image; pass `already_256=True` images straight from the RLDS."""
        tf = _tf()
        import cv2
        a = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        if a.shape[:2] != (256, 256):
            a = cv2.resize(a, (256, 256), interpolation=cv2.INTER_AREA)   # convert_piper_rlds.py: squashed, INTER_AREA
        x = tf.image.convert_image_dtype(tf.convert_to_tensor(a), tf.float32)
        x = tf.image.resize(x, (224, 224))                                # the RLDS decode_and_resize
        from experiments.robot.openvla_utils import crop_and_resize       # dlimp's crop, so the area matches training
        x = tf.clip_by_value(crop_and_resize(x, 0.9, 1), 0, 1)
        return Image.fromarray(tf.image.convert_image_dtype(x, tf.uint8, saturate=True).numpy()).convert("RGB")

    # ---------------------------------------------------------------- policy
    @torch.no_grad()
    def act_chunk(self, image, instruction, normalised=False):
        """The policy's whole chunk: (window, action_dim).  Un-normalised unless `normalised`."""
        if instruction != self._instr: self._prompt, self._instr = self._build_prompt(instruction), instruction
        inputs = self.processor(self._prompt, self.preprocess(image))
        inputs = {k: (v.to(self.device, dtype=torch.bfloat16) if v.dtype.is_floating_point else v.to(self.device))
                  for k, v in inputs.items()}
        h = self.vla(**inputs, output_hidden_states=True).hidden_states[-1]
        vis, txt = h[:, :self.n_patches].float(), h[:, self.n_patches:].float()
        if self.nq:                                                       # the query-token states the heads were trained on
            qm = inputs["input_ids"][0] > 32000; txt = txt[:, qm]
            assert txt.shape[1] == self.nq, f"{txt.shape[1]} query tokens in the prompt, expected {self.nq}"
        out = (self.heads.sample(vis, txt, self.flow_steps, self.gen) if self.flow else self.heads(vis, txt))
        chunk = out[0].reshape(self.W, self.adim).cpu().numpy()
        if normalised: return chunk
        return np.where(self.norm_mask, 0.5 * (chunk + 1) * (self.high - self.low) + self.low, chunk)

    def act(self, image, instruction):
        """One action to send to the arm, temporally ensembled over the overlapping chunks, exactly as the LIBERO
        evaluator does (unless ensemble=False): (action_dim,) = 6 absolute joint targets in degrees + gripper
        in [0, 1]."""
        ch = self.act_chunk(image, instruction, normalised=True)
        if not self.use_ens:
            return np.where(self.norm_mask, 0.5 * (ch[0] + 1) * (self.high - self.low) + self.low, ch[0])
        return self.ens(ch, self.norm_mask, self.low, self.high)

    @staticmethod
    def gripper_to_raw(g):
        """[0, 1] back to the arm's own gripper units (convert_piper_rlds.py GRIP_SCALE)."""
        return float(g) * GRIP_SCALE


# -------------------------------------------------------------------- self-check on the held-out episodes
def replay(policy, data_root, dataset="piper_pick_can", episodes=None, split="train[95%:]"):
    import tensorflow_datasets as tfds
    b = tfds.builder_from_directory(os.path.join(data_root, dataset, "1.0.0"))
    ds = b.as_dataset(split=split)
    per_dim_first, per_dim_ens, norm_chunk, n = [], [], [], 0
    for ei, epi in enumerate(ds):
        if episodes is not None and ei >= episodes: break
        steps = list(epi["steps"])
        imgs = [s["observation"]["image"].numpy() for s in steps]
        acts = np.stack([s["action"].numpy() for s in steps]).astype(np.float64)
        instr = steps[0]["language_instruction"].numpy().decode()
        policy.reset(episode=ei)
        for t in range(len(steps)):
            ch = policy.act_chunk(imgs[t], instr, normalised=True)
            un = np.where(policy.norm_mask, 0.5 * (ch + 1) * (policy.high - policy.low) + policy.low, ch)
            gt = acts[t:t + policy.W]
            gtn = np.where(policy.norm_mask, 2 * (gt - policy.low) / (policy.high - policy.low) - 1, gt)
            norm_chunk.append(np.abs(ch[:len(gt)] - gtn).mean())
            per_dim_first.append(np.abs(un[0] - acts[t]))
            policy.ens.buf[1:] = policy.ens.buf[:-1]; policy.ens.buf_mask[1:] = policy.ens.buf_mask[:-1]
            policy.ens.buf[:, :-1] = policy.ens.buf[:, 1:]; policy.ens.buf_mask[:, :-1] = policy.ens.buf_mask[:, 1:]
            policy.ens.buf_mask = policy.ens.buf_mask * policy.ens.mask
            policy.ens.buf[0] = ch; policy.ens.buf_mask[0] = True
            a = (policy.ens.buf[:, 0, :] * policy.ens.buf_mask[:, 0:1] * policy.ens.weights).sum(0) / (policy.ens.buf_mask[:, 0:1] * policy.ens.weights).sum()
            a = np.where(policy.norm_mask, 0.5 * (a + 1) * (policy.high - policy.low) + policy.low, a)
            per_dim_ens.append(np.abs(a - acts[t])); n += 1
        print(f"[replay] episode {ei}: {len(steps)} steps, instruction {instr!r}", flush=True)
    f, e = np.stack(per_dim_first).mean(0), np.stack(per_dim_ens).mean(0)
    print(f"\n{n} held-out steps, mean |error| per action dimension")
    print("dim          " + "".join(f"{i:>9}" for i in range(policy.adim)))
    print("chunk[0]     " + "".join(f"{v:9.3f}" for v in f))
    print("ensembled    " + "".join(f"{v:9.3f}" for v in e))
    print(f"\njoints 0-5 (degrees): chunk[0] {f[:6].mean():.3f} deg mean, ensembled {e[:6].mean():.3f} deg mean")
    print(f"gripper (0-1 units):  chunk[0] {f[6]:.4f}, ensembled {e[6]:.4f}  (x{GRIP_SCALE} for the arm's units)")
    print(f"whole-chunk L1 in NORMALISED space: {np.mean(norm_chunk):.4f}"
          f"   <- compare with eval_heldout.py's one-step action L1 for the same checkpoint")
    return dict(steps=n, per_dim_first=f.tolist(), per_dim_ens=e.tolist(), norm_chunk_l1=float(np.mean(norm_chunk)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--replay", metavar="DATA_ROOT", help="replay the held-out episodes out of this RLDS root")
    ap.add_argument("--dataset", default="piper_pick_can")
    ap.add_argument("--episodes", type=int, default=None, help="only the first N held-out episodes")
    ap.add_argument("--flow_steps", type=int, default=0, help="0 = the value in ctrl_meta.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = PiperPolicy(a.checkpoint, flow_steps=a.flow_steps or None)
    if not a.replay:
        print("loaded; pass --replay <rlds root> to check it against the held-out episodes"); return
    r = replay(p, a.replay, a.dataset, a.episodes)
    if a.out: json.dump(r, open(a.out, "w"), indent=1); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
