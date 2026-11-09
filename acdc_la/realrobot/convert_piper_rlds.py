"""LeRobot v3 Piper dataset (jachansantiago/pick_can_place_plate_150_trimmed) -> RLDS/TFDS in the LIBERO layout that
prismatic's RLDSDataset reads (2026-11-06, real-robot experiment).

Choices (keep identical at deployment, see acdc_la/realrobot/README.md):
  * 30 Hz -> 10 Hz: every 3rd frame of each episode (offset 0), so a 12-step action chunk spans 1.2 s and the
    latent labeller's frame stride (window 12 // 4 = 3 steps) is 0.3 s.
  * action = the recorded 7-D command at the kept frame: 6 ABSOLUTE joint targets (degrees, Piper convention) +
    gripper.  prismatic's EEF_POS contract normalises dims 0-5 (q01/q99) and leaves the last one raw, so the gripper
    is stored as gripper / GRIP_SCALE, clipped to [0, 1] (raw range in this dataset: 0 .. 8.15).
  * images: front -> `image` (primary), wrist -> `wrist_image`, both resized 640x480 -> 256x256 (cv2 INTER_AREA,
    aspect squashed, RGB).
  * state (14-D, raw observation.state) and joint_state (first 7 dims) are stored but not used by the policy.
Usage (compute node; login-node thread limits break numpy/TF):
  PYTHONPATH=envs/convert_extra envs/univla_vla/bin/python latent_free/acdc_la/realrobot/convert_piper_rlds.py \
      --src <lerobot dir> --out <rlds root>   # writes <rlds root>/piper_pick_can/1.0.0
"""
import argparse, os
import numpy as np

GRIP_SCALE = 8.2
FPS_IN, SUB = 30, 3


def load(src):
    import pyarrow.parquet as pq, av, cv2
    t = pq.read_table(os.path.join(src, "data/chunk-000/file-000.parquet"))
    act = np.asarray(t.column("action").to_pylist(), np.float32)
    st = np.asarray(t.column("observation.state").to_pylist(), np.float32)
    ep = np.asarray(t.column("episode_index").to_pylist())
    task = pq.read_table(os.path.join(src, "meta/tasks.parquet")).to_pydict()["__index_level_0__"][0]
    assert np.all(np.diff(ep) >= 0) and len(act) == len(ep)
    keep = np.zeros(len(ep), bool)                      # frames kept at 10 Hz, decided before decoding
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]; keep[idx[::SUB]] = True
    frames = {}
    for cam in ("front", "wrist"):
        c = av.open(os.path.join(src, f"videos/observation.images.{cam}/chunk-000/file-000.mp4"))
        s = c.streams.video[0]; out = {}
        for k, f in enumerate(c.decode(s)):
            # one video frame per data row (checked: 19246 = 19246, pts 0 at row 0); assert the clock agrees
            if k < len(keep) and keep[k]:
                assert abs(float(f.pts * s.time_base) - k / FPS_IN) < 0.5 / FPS_IN, (cam, k, float(f.pts * s.time_base))
                out[k] = cv2.resize(f.to_ndarray(format="rgb24"), (256, 256), interpolation=cv2.INTER_AREA)
        assert len(out) == keep.sum(), (cam, len(out), keep.sum())
        frames[cam] = out
    return act, st, ep, keep, frames, task


def build(src, out):
    import tensorflow_datasets as tfds
    act, st, ep, keep, frames, task = load(src)
    g = np.clip(act[:, 6] / GRIP_SCALE, 0, 1)

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
            for e in np.unique(ep):
                idx = [k for k in np.nonzero(ep == e)[0] if keep[k]]; n = len(idx); steps = []
                for i, k in enumerate(idx):
                    a = np.concatenate([act[k, :6], [g[k]]]).astype(np.float32)
                    steps.append({"observation": {"image": frames["front"][k], "wrist_image": frames["wrist"][k],
                                                  "state": st[k], "joint_state": st[k, :7]},
                                  "action": a, "discount": 1.0, "reward": float(i == n - 1),
                                  "is_first": i == 0, "is_last": i == n - 1, "is_terminal": i == n - 1,
                                  "language_instruction": task})
                yield f"ep{int(e)}", {"steps": steps, "episode_metadata": {"file_path": f"lerobot/ep{int(e)}", "episode_index": int(e)}}

    b = PiperPickCan(data_dir=out)
    b.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=src))
    ds = b.as_dataset(split="train")
    n_ep = n_st = 0
    for x in ds:
        n_ep += 1; n_st += int(x["steps"].cardinality())
    print(f"[piper] wrote {b.data_dir}: {n_ep} episodes, {n_st} steps at 10 Hz, task={task!r}, grip_scale={GRIP_SCALE}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(); build(a.src, a.out)
