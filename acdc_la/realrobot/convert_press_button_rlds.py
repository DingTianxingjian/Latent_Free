"""LeRobot v3 Piper dataset anhdao69/press_button_task -> RLDS/TFDS `press_button` (2026-11-10, Jetstream2).

Copy of convert_piper_rlds.py (same robot, same action/state layout, same image keys) with three differences:
  * builder class `PressButton` -> TFDS name `press_button`, registered in prismatic by univla_piper.patch
    (configs.py + transforms.py, entries copied verbatim from piper_pick_can).
  * MULTI-TASK: the instruction is taken per episode from `task_index` -> meta/tasks.parquet
    (4 tasks, 25 episodes each: press the blue / orange / pink / red button).  pick-can had one task.
  * GRIPPER DECISION (handoff 4a): keep GRIP_SCALE = 8.2, i.e. the SAME calibration as piper_pick_can.
    Raw gripper here is 0.04 .. 0.06 (button pressing never opens the gripper); it is the same physical
    sensor/units as pick-can (raw 0 .. 8.15), so the honest value is "closed" ~0.005-0.007.  We do NOT
    rescale to [0, 1] per dataset: that would invent a signal out of sensor noise and make the two Piper
    datasets' gripper dims mean different things.  The dim is ~constant here and carries no information.
Everything else is identical: 30 Hz -> 10 Hz (every 3rd frame, offset 0), 6 ABSOLUTE joint targets in degrees,
images front -> `image`, wrist -> `wrist_image`, 640x480 -> 256x256 cv2 INTER_AREA RGB.
Padding (handoff 4b): at 10 Hz only ~31 % of samples have a fully real 14-frame window (577/1877 steps; 2 episodes
have none).  Accepted as-is (user decision 2026-11-10, option 1): it is identical across arms.

Run as an IMPORT, not a script (TFDS refuses builders defined in `__main__`):
  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -c "
  import sys; sys.argv = ['x', '--src', '<lerobot dir>', '--out', 'data/real_rlds']
  from acdc_la.realrobot.convert_press_button_rlds import main; main()"
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
    ti = np.asarray(t.column("task_index").to_pylist())
    tk = pq.read_table(os.path.join(src, "meta/tasks.parquet")).to_pydict()
    tasks = dict(zip(tk["task_index"], tk["__index_level_0__"]))
    assert np.all(np.diff(ep) >= 0) and len(act) == len(ep)
    ep_task = {}
    for e in np.unique(ep):
        u = np.unique(ti[ep == e]); assert len(u) == 1, (e, u)       # one instruction per episode
        ep_task[int(e)] = tasks[int(u[0])]
    keep = np.zeros(len(ep), bool)
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]; keep[idx[::SUB]] = True
    # Unlike pick-can, the concatenated mp4 is NOT one frame per data row: it holds 6190 frames for 5533 rows,
    # with dead footage between some episodes (e.g. 5.03 s between ep 2 and 3).  The mapping is LeRobot v3's:
    # video time = episodes[e].videos/<cam>/from_timestamp + row timestamp.  Frames are matched by pts.
    ts = np.asarray(t.column("timestamp").to_pylist(), np.float64)
    me = pq.read_table(os.path.join(src, "meta/episodes/chunk-000/file-000.parquet")).to_pydict()
    frames = {}
    for cam in ("front", "wrist"):
        off = dict(zip(me["episode_index"], me[f"videos/observation.images.{cam}/from_timestamp"]))
        assert all(v == 0 for v in me[f"videos/observation.images.{cam}/file_index"])
        want = {}                                       # video frame index -> data row
        for k in np.nonzero(keep)[0]:
            fi = int(round((off[int(ep[k])] + ts[k]) * FPS_IN)); assert fi not in want; want[fi] = k
        c = av.open(os.path.join(src, f"videos/observation.images.{cam}/chunk-000/file-000.mp4"))
        s = c.streams.video[0]; out = {}
        for f in c.decode(s):
            tsec = float(f.pts * s.time_base); fi = int(round(tsec * FPS_IN))
            assert abs(tsec - fi / FPS_IN) < 0.25 / FPS_IN, (cam, tsec)
            if fi in want:
                out[want[fi]] = cv2.resize(f.to_ndarray(format="rgb24"), (256, 256), interpolation=cv2.INTER_AREA)
        assert len(out) == keep.sum(), (cam, len(out), keep.sum())
        frames[cam] = out
    return act, st, ep, keep, frames, ep_task


def build(src, out, dataset="press_button"):
    import tensorflow_datasets as tfds
    act, st, ep, keep, frames, ep_task = load(src)
    g = np.clip(act[:, 6] / GRIP_SCALE, 0, 1)

    class _Base(tfds.core.GeneratorBasedBuilder):
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
                                  "language_instruction": ep_task[int(e)]})
                yield f"ep{int(e)}", {"steps": steps, "episode_metadata": {"file_path": f"lerobot/ep{int(e)}", "episode_index": int(e)}}

    # TFDS takes the dataset name from the class name (press_button -> PressButton, stack_bowls -> StackBowls);
    # parameterised 2026-11-12 so the same converter serves every single- or multi-task LeRobot v3 Piper folder.
    Builder = type("".join(w.capitalize() for w in dataset.split("_")), (_Base,), {})
    assert Builder.name == dataset, (Builder.name, dataset)
    b = Builder(data_dir=out)
    b.download_and_prepare(download_config=tfds.download.DownloadConfig(manual_dir=src))
    ds = b.as_dataset(split="train")
    n_ep = n_st = 0; instr = {}
    for x in ds:
        n_ep += 1; n_st += int(x["steps"].cardinality())
        s0 = next(iter(x["steps"])); k = s0["language_instruction"].numpy().decode(); instr[k] = instr.get(k, 0) + 1
    print(f"[{dataset}] wrote {b.data_dir}: {n_ep} episodes, {n_st} steps at 10 Hz, grip_scale={GRIP_SCALE}")
    print(f"[{dataset}] episodes per instruction: {instr}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="press_button", help="TFDS name to write (also the prismatic registration key)")
    a = p.parse_args(); build(a.src, a.out, a.dataset)


if __name__ == "__main__":
    main()
