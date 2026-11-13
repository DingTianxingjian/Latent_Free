"""Full-resolution example frames from a LeRobot v3 Piper folder, for a paper figure (2026-11-13).

Our RLDS copies are 256x256 (the policy's input); for a figure the ORIGINAL 640x480 video frames are what you
want.  This pulls a time series of frames from chosen episodes of the front and/or wrist camera, at full
resolution, using the same v3 frame mapping the converters use (video time = episode's `from_timestamp` for that
camera + the row's `timestamp`, frames matched by pts).

  envs/univla_vla/bin/python -m acdc_la.realrobot.extract_paper_frames \
      --src <lerobot task dir> --out <png dir> --n 5 --cams front wrist

Episodes: `--episodes 3 7` picks those; `--per_task 1` instead picks one episode per distinct instruction (what a
multi-instruction task like press_button needs, so every colour appears).  `--n` frames are taken at equal spacing
over the episode, first and last included.
"""
import argparse, json, os, sys

# pyarrow/av live in a --target install next to the project envs (uv pip install --target envs/convert_extra);
# added here so this can be run with python -I on downloaded data.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "envs", "convert_extra"))
import numpy as np

FPS_IN = 30


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="a LeRobot v3 task folder (holds data/, meta/, videos/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5, help="frames per episode, equally spaced, ends included")
    ap.add_argument("--fracs", type=float, nargs="*", default=None,
                    help="explicit sample points as fractions of the episode, instead of --n equal spacing. "
                         "Tasks whose motion starts late (press_button: the arm is parked for the first ~40%) "
                         "waste half a figure row on identical frames otherwise.")
    ap.add_argument("--cams", nargs="+", default=["front"], choices=["front", "wrist"])
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--per_task", type=int, default=1, help="if --episodes is not given: episodes per instruction")
    ap.add_argument("--tag", default=None, help="filename prefix (default: the folder name)")
    a = ap.parse_args()
    import pyarrow.parquet as pq, av
    from PIL import Image

    tag = a.tag or os.path.basename(a.src.rstrip("/"))
    os.makedirs(a.out, exist_ok=True)
    t = pq.read_table(os.path.join(a.src, "data/chunk-000/file-000.parquet"))
    ep = np.asarray(t.column("episode_index").to_pylist())
    ts = np.asarray(t.column("timestamp").to_pylist(), np.float64)
    ti = np.asarray(t.column("task_index").to_pylist())
    tk = pq.read_table(os.path.join(a.src, "meta/tasks.parquet")).to_pydict()
    names = dict(zip(tk["task_index"], tk["__index_level_0__"]))
    ep_task = {int(e): names[int(np.unique(ti[ep == e])[0])] for e in np.unique(ep)}

    if a.episodes:
        picks = list(a.episodes)
    else:                                     # one (or --per_task) episode per distinct instruction
        picks, seen = [], {}
        for e in sorted(ep_task):
            k = ep_task[e]
            if seen.get(k, 0) < a.per_task: picks.append(e); seen[k] = seen.get(k, 0) + 1
    print(f"[{tag}] {len(np.unique(ep))} episodes, {len(set(ep_task.values()))} instruction(s); taking {picks}")

    me = pq.read_table(os.path.join(a.src, "meta/episodes/chunk-000/file-000.parquet")).to_pydict()
    index = []
    for cam in a.cams:
        off = dict(zip(me["episode_index"], me[f"videos/observation.images.{cam}/from_timestamp"]))
        want = {}                                                     # video frame index -> output filename
        for e in picks:
            rows = np.nonzero(ep == e)[0]
            sel = (np.unique((np.asarray(a.fracs) * (len(rows) - 1)).round().astype(int)) if a.fracs
                   else np.unique(np.linspace(0, len(rows) - 1, a.n).round().astype(int)))
            for j, s in enumerate(sel):
                k = rows[s]
                fi = int(round((off[int(e)] + ts[k]) * FPS_IN))
                want[fi] = f"{tag}_ep{int(e):03d}_{cam}_{j}of{len(sel)}.png"
                index.append(dict(file=want[fi], task=tag, episode=int(e), camera=cam, frame_in_episode=int(s),
                                  n_steps_30hz=len(rows), t_sec=round(float(ts[k]), 3),
                                  instruction=ep_task[int(e)]))
        c = av.open(os.path.join(a.src, f"videos/observation.images.{cam}/chunk-000/file-000.mp4"))
        st = c.streams.video[0]; got = 0
        for f in c.decode(st):
            fi = int(round(float(f.pts * st.time_base) * FPS_IN))
            if fi in want:
                im = f.to_ndarray(format="rgb24")
                Image.fromarray(im).save(os.path.join(a.out, want[fi]))
                got += 1
        assert got == len(want), (cam, got, len(want))
        print(f"[{tag}] {cam}: wrote {got} frames at {im.shape[1]}x{im.shape[0]}")
    p = os.path.join(a.out, f"{tag}_index.json")
    json.dump(index, open(p, "w"), indent=1)
    print(f"[{tag}] index -> {p}")


if __name__ == "__main__":
    main()
