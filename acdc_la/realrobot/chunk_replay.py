"""Open-loop error at EVERY chunk position k (2026-11-11): what executing k steps of a chunk without re-predicting costs.

piper_policy.py --replay scores chunk[0] only, which hides how errors grow along the chunk.  Here, for each held-out
step s, the policy predicts a chunk from image s and we compare the joint TARGETS it would send at s+k with the
demonstrated absolute targets action[s+k], k = 0..W-1, in degrees:

  absolute        target_k = chunk_k
  delta_anchored  target_k = q_s + chunk_k                 (exact: needs only the joints at s, which the robot has)
  delta (per-step)  two bounds, because target_k needs the joints AT s+k, which depend on how the arm tracked:
      "demo state"      q_{s+k} taken from the demonstration -- the arm is exactly where the demonstrator's was
                        (optimistic: this is what piper_policy --replay assumes)
      "ideal tracking"  the arm reaches every command before the next tick, so q_{s+k} = target_{k-1}: the
                        per-step deltas INTEGRATE.  The demos were teleoperated with a CONTINUOUS (30 Hz) command, so
                        the demo arm at s+1 is already past the step-s command; an arm that settles at each 10 Hz command
                        is BEHIND it, and on press_button the step-2 target comes out -2.7 deg on j1 and +2.4 deg on j2
                        (data only, 2026-11-12): different per joint, so the end effector's DIRECTION bends.
      The robot is somewhere in that range.  The anchored form has no such range: its targets don't depend on tracking.

  PYTHONPATH=.:third_party/UniVLA python -m acdc_la.realrobot.chunk_replay --checkpoint <dir> --dataset press_button
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--dataset", default="press_button")
    ap.add_argument("--abs_root", default="data/real_rlds"); ap.add_argument("--split", default="train[95%:]")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda", help="cpu runs the VLA in fp32 (slow, ~10 s/step, but needs no GPU)")
    a = ap.parse_args()
    from acdc_la.realrobot.piper_policy import PiperPolicy, _tf, _unnorm_key_for
    stats = json.load(open(os.path.join(a.checkpoint, "dataset_statistics.json")))
    key = None if len(stats) == 1 else next(k for k in (a.dataset, a.dataset + "_anc") if k in stats)
    p = PiperPolicy(a.checkpoint, unnorm_key=key, ensemble=False, device=a.device); _tf()
    if a.device == "cpu":
        import torch; torch.set_num_threads(16); p.vla.float()
    import tensorflow_datasets as tfds
    ds = tfds.builder_from_directory(os.path.join(a.abs_root, a.dataset, "1.0.0")).as_dataset(split=a.split)
    W, sp = p.W, p.action_space
    rows = {m: [[] for _ in range(W)] for m in (["anchored"] if sp == "delta_anchored" else
                                               ["demo_state", "ideal_tracking"] if sp == "delta" else ["absolute"])}
    for ei, epi in enumerate(ds):
        steps = list(epi["steps"]); T = len(steps)
        imgs = [s["observation"]["image"].numpy() for s in steps]
        A = np.stack([s["action"].numpy() for s in steps]).astype(np.float64)[:, :6]
        Q = np.stack([s["observation"]["joint_state"].numpy() for s in steps]).astype(np.float64)[:, :6]
        instr = steps[0]["language_instruction"].numpy().decode(); p.reset(episode=ei)
        for s in range(T):
            ch = p.act_chunk(imgs[s], instr)[:, :6]; K = min(W, T - s)
            if sp == "delta_anchored":
                tg = {"anchored": Q[s][None] + ch}
            elif sp == "delta":
                tg = {"demo_state": np.stack([Q[min(s + k, T - 1)] + ch[k] for k in range(W)]),
                      "ideal_tracking": Q[s][None] + np.cumsum(ch, 0)}
            else:
                tg = {"absolute": ch}
            for m, t in tg.items():
                for k in range(K): rows[m][k].append(np.abs(t[k] - A[s + k]).mean())
        print(f"[chunk-replay] episode {ei}: {T} steps, {instr!r}", flush=True)
    res = {"checkpoint": a.checkpoint, "dataset": a.dataset, "action_space": sp, "per_k": {}}
    for m, r in rows.items():
        e = [float(np.mean(x)) if x else float("nan") for x in r]; res["per_k"][m] = e
        print(f"\n{m:15s} error by chunk position k (deg, mean over joints; n at k=0: {len(r[0])})")
        print("  k   " + "".join(f"{k:>7d}" for k in range(W)))
        print("  err " + "".join(f"{v:7.2f}" for v in e))
        for n in (1, 2, 4, 8, 12):
            if n <= W: print(f"  executing {n:2d} steps per chunk: mean error {np.nanmean(e[:n]):.2f} deg")
    if a.out: json.dump(res, open(a.out, "w"), indent=1); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
