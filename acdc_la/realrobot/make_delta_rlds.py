"""Derive a DELTA-action copy of the Piper RLDS from the absolute-action one (2026-11-10).

Why: `piper_pick_can` stores 6 ABSOLUTE joint targets in degrees.  Joint 1's q01/q99 span 164 deg while the arm
moves ~2 deg per step, so the informative part of each prediction is ~1.2 % of the output range, against a measured
error of 2.4-11.5 % of each joint's range -- the signal sits below the error (LEDGER 2026-11-10).  Every policy in
this line (OpenVLA, RT-1/2, Octo) predicts deltas instead, and OpenVLA's own recipe is delta EEF; the absolute
parameterisation is this project's deviation.

The delta is `action[:6] - joint_state[:6]`, the target MINUS WHERE THE ARM ACTUALLY IS (both in degrees, correlation
0.99+, mean |difference| 1.22 deg per joint).  That is the self-correcting form: at deployment the robot adds the
predicted delta to its own measured position, so a prediction error does not accumulate.  The alternative,
`action[t] - action[t-1]` (mean |.| 2.84 deg), is a difference of COMMANDS and would require integrating the policy's
own output at deployment, which drifts.  The gripper (dim 6) stays absolute in [0, 1]: it is a near-binary command
and a delta buys nothing.

Built from the EXISTING RLDS rather than by re-running convert_piper_rlds.py on the LeRobot source, so the frames,
the 10 Hz subsampling, the episode boundaries and the instruction are bit-identical and the ONLY difference between
the two datasets is the action field -- which is what makes the two training runs a controlled comparison.

The dataset NAME stays `piper_pick_can`, so no change to prismatic's OXE configs/transforms is needed; the delta
copy lives under a different ROOT and is selected with `--data_root_dir`.  Verify which one a run used from its
`dataset_statistics.json`: absolute has joint-1 mean ~111, delta has every joint mean ~0.

  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -I acdc_la/realrobot/make_delta_rlds.py \
      --src data/real_rlds --out data/real_rlds_delta
"""
import argparse, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/real_rlds"); ap.add_argument("--out", default="data/real_rlds_delta")
    ap.add_argument("--dataset", default="piper_pick_can")
    a = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    src = tfds.builder_from_directory(os.path.join(a.src, a.dataset, "1.0.0"))
    eps = []
    for epi in src.as_dataset(split="train"):
        steps = list(epi["steps"])
        acts = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
        js = np.stack([s["observation"]["joint_state"].numpy() for s in steps]).astype(np.float32)
        d = acts.copy()
        d[:, :6] = acts[:, :6] - js[:, :6]                      # target minus the measured configuration
        eps.append(dict(
            idx=int(epi["episode_metadata"]["episode_index"].numpy()),
            path=epi["episode_metadata"]["file_path"].numpy().decode(),
            instr=steps[0]["language_instruction"].numpy().decode(),
            img=[s["observation"]["image"].numpy() for s in steps],
            wrist=[s["observation"]["wrist_image"].numpy() for s in steps],
            state=[s["observation"]["state"].numpy() for s in steps],
            jstate=[s["observation"]["joint_state"].numpy() for s in steps],
            action=d))
    n_st = sum(len(e["action"]) for e in eps)
    allied = np.concatenate([e["action"] for e in eps])
    print(f"[delta] {len(eps)} episodes, {n_st} steps")
    print(f"[delta] action[:6] mean {np.round(allied[:, :6].mean(0), 3)}")
    print(f"[delta] action[:6] q01  {np.round(np.quantile(allied[:, :6], 0.01, axis=0), 2)}")
    print(f"[delta] action[:6] q99  {np.round(np.quantile(allied[:, :6], 0.99, axis=0), 2)}")
    print(f"[delta] gripper range {allied[:, 6].min():.4f} .. {allied[:, 6].max():.4f} (unchanged, absolute)")

    class PiperPickCan(tfds.core.GeneratorBasedBuilder):
        VERSION = tfds.core.Version("1.0.0")
        def _info(self):
            F = tfds.features
            return self.dataset_info_from_configs(features=F.FeaturesDict({
                "steps": F.Dataset({
                    "observation": F.FeaturesDict({
                        "image": F.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "wrist_image": F.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "state": F.Tensor(shape=(14,), dtype=np.float32),
                        "joint_state": F.Tensor(shape=(7,), dtype=np.float32)}),
                    "action": F.Tensor(shape=(7,), dtype=np.float32),
                    "discount": F.Scalar(dtype=np.float32), "reward": F.Scalar(dtype=np.float32),
                    "is_first": F.Scalar(dtype=np.bool_), "is_last": F.Scalar(dtype=np.bool_),
                    "is_terminal": F.Scalar(dtype=np.bool_), "language_instruction": F.Text()}),
                "episode_metadata": F.FeaturesDict({"file_path": F.Text(), "episode_index": F.Scalar(dtype=np.int64)})}))
        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}
        def _generate_examples(self):
            for e in eps:
                n = len(e["action"]); steps = []
                for i in range(n):
                    steps.append({"observation": {"image": e["img"][i], "wrist_image": e["wrist"][i],
                                                  "state": e["state"][i], "joint_state": e["jstate"][i]},
                                  "action": e["action"][i], "discount": 1.0, "reward": float(i == n - 1),
                                  "is_first": i == 0, "is_last": i == n - 1, "is_terminal": i == n - 1,
                                  "language_instruction": e["instr"]})
                yield f"ep{e['idx']}", {"steps": steps, "episode_metadata": {"file_path": e["path"], "episode_index": e["idx"]}}

    b = PiperPickCan(data_dir=a.out)
    b.download_and_prepare()
    ds = b.as_dataset(split="train"); n_ep2 = n_st2 = 0
    for x in ds:
        n_ep2 += 1; n_st2 += int(x["steps"].cardinality())
    assert (n_ep2, n_st2) == (len(eps), n_st), (n_ep2, n_st2, len(eps), n_st)
    print(f"[delta] wrote {b.data_dir}: {n_ep2} episodes, {n_st2} steps")


if __name__ == "__main__":
    main()
