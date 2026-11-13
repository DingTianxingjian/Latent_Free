"""Score a replay dump against the do-nothing nulls ON THE SAME held-out steps, paired by episode (2026-11-13).

`piper_policy.py --replay ... --dump d.npz` writes the PER-STEP per-dim errors and the episode id of each step.
This reads that back, recomputes the two nulls on exactly those steps, and bootstraps the difference by resampling
EPISODES -- the independent unit is the episode, not the step (184 steps from 5 episodes is a sample of 5).

  * null "stay where you are": predict delta = 0            -> error |a[t]|
  * null "repeat the last command": predict delta[t-1]       -> error |a[t] - a[t-1]|, undefined at t = 0

Dataset-wide null values are NOT the ones to quote: on stack_bowls they are 0.901 / 0.719 over all 100 episodes but
0.863 / 0.667 on the `train[95%:]` episodes a policy is actually scored on (LEDGER.md 2026-11-13 07:00).

  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -m acdc_la.realrobot.replay_vs_nulls \
      --dump <d.npz> --data_root data/real_rlds_delta --dataset stack_bowls
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="npz from piper_policy.py --replay --dump")
    ap.add_argument("--data_root", default="data/real_rlds_delta")
    ap.add_argument("--dataset", default="stack_bowls")
    ap.add_argument("--split", default="train[95%:]", help="must be the split the replay used")
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()

    import os
    import tensorflow_datasets as tfds
    b = tfds.builder_from_directory(os.path.join(a.data_root, a.dataset, "1.0.0"))
    eps = [np.stack([s["action"] for s in list(e["steps"])]).astype(np.float64)
           for e in tfds.as_numpy(b.as_dataset(split=a.split))]
    print(f"held-out {a.split}: {len(eps)} episodes, {sum(len(e) for e in eps)} steps")

    d = np.load(a.dump)
    mod_first, mod_ens, ep = d["first"], d["ens"], d["ep"]
    assert len(mod_first) == sum(len(e) for e in eps), (len(mod_first), sum(len(e) for e in eps))

    stay, repeat, ep2 = [], [], []
    for ei, e in enumerate(eps):
        for t in range(len(e)):
            stay.append(np.abs(e[t, :6]))
            repeat.append(np.abs(e[t, :6] - e[t - 1, :6]) if t else np.full(6, np.nan))
            ep2.append(ei)
    stay, repeat, ep2 = np.stack(stay), np.stack(repeat), np.array(ep2)
    assert (ep2 == ep).all(), "dump's episode ids do not match this split -- different conversion order?"

    arms = {"model chunk[0]": mod_first[:, :6].mean(1), "model ensembled": mod_ens[:, :6].mean(1),
            "null stay still": stay.mean(1), "null repeat last": repeat.mean(1)}
    ok = ~np.isnan(arms["null repeat last"])
    print(f"\n--- mean over joints 0-5, degrees ({len(ok)} steps all / {ok.sum()} where every arm is defined)")
    for k, v in arms.items():
        print(f"  {k:<18} all {np.nanmean(v):.3f}   t>=1 {v[ok].mean():.3f}")

    def boot(x, y, m, seed=0):
        r = np.random.default_rng(seed); u = np.unique(ep2[m]); idx = {e: np.nonzero(m & (ep2 == e))[0] for e in u}
        o = np.array([(lambda s: x[s].mean() - y[s].mean())(np.concatenate([idx[e] for e in r.choice(u, len(u), True)]))
                      for _ in range(a.draws)])
        return o.mean(), np.percentile(o, 2.5), np.percentile(o, 97.5), (o >= 0).mean()

    print(f"\n--- model - null (negative = model better), 95% paired cluster bootstrap over {len(np.unique(ep2))} episodes")
    for mk in ("model chunk[0]", "model ensembled"):
        for nk in ("null repeat last", "null stay still"):
            m, lo, hi, p = boot(arms[mk], arms[nk], ok)
            v = "model better" if hi < 0 else ("null better" if lo > 0 else "NOT DISTINGUISHABLE")
            print(f"  {mk:<16} - {nk:<17} {m:+.3f} deg  [{lo:+.3f}, {hi:+.3f}]  P(model worse)={p:.2f}  -> {v}")

    print("\n--- per-episode means, degrees: chunk[0] / ensembled / stay-still / repeat-last")
    for e in np.unique(ep2):
        m = ok & (ep2 == e)
        print(f"  ep {e} ({m.sum():3d} steps)  " + "  ".join(f"{arms[k][m].mean():6.3f}" for k in arms))

    print("\n--- per joint, degrees")
    print("  joint        " + "".join(f"{i:>9}" for i in range(6)))
    for k, v in (("chunk[0]", mod_first[:, :6].mean(0)), ("ensembled", mod_ens[:, :6].mean(0)),
                 ("repeat last", np.nanmean(repeat, 0)), ("stay still", stay.mean(0))):
        print(f"  {k:<12} " + "".join(f"{x:9.3f}" for x in v))


if __name__ == "__main__":
    main()
