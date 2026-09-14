"""Exp A -- frame-count null for the Wan temporal-fusion result (Exp 5b).

Exp 5b headline: Wan's fused window latent `zlast` (9408-d) decodes net action
better than differencing independent endpoints -- xyz R2 0.858 vs `diff` 0.731
(and vs `pair` 0.786) at m=1.  But `zlast` SAW FIVE FRAMES (o_t..o_{t+4}); `diff`
and `pair` saw only the two endpoints.  So +0.127 could be *learned temporal
fusion* (Pillar 1) or merely *more frames*.  Untested -- this tests it.

Design.  Build `multi_k` = concat of k INDEPENDENT single-frame latents evenly
spaced over the same 4m-step window (each frame encoded ALONE -- the z0 path, no
fusion), plus `multi_k_pca9408` = its own top-9408 train PCs (width-matched to
`zlast`).  For m=1 the 5-frame window means `multi_5` uses EXACTLY the frames the
fused latent saw -- the decisive row.

Verdict rule (printed):
  * multi_5 (esp. multi_5_pca9408) REACHES zlast (<=~0.01 xyz R2)
        -> the advantage is frame count, not fusion; Pillar 1 NOT supported.
  * zlast still leads at matched frames AND matched width -> fusion is real;
        report the gap per xyz/rot/cont6.
  * intermediate -> fraction of the gap explained by frame count.

Every single-frame latent is cached (cache/wfeats_clean.npy, one row per frame),
so this needs NO window encoding -- pure indexing + ridge.  build_pairs is
deterministic, so `single`/`diff`/`pair` recomputed here reproduce
exp_wan_temporal.csv exactly (asserted); `zlast`/`video` are copied from that CSV
as the fused reference so the comparison is against the *published* numbers.
"""

import os

import numpy as np
import pandas as pd
import torch

import data
import probes_torch

MS = (1, 2)
KS = {1: (3, 5), 2: (3, 5, 9)}       # k frames evenly spanning the 4m-step window
PCA_DIM = 9408                        # width of zlast, the fused single-chunk latent
RAW_CAP = 60000                       # ridge gram is d*d float64; skip raw above this
TOL = 0.01                            # "reaches" tolerance on xyz R2


def pca_project(Xtr, Xva, q, device):
    """Top-q train principal directions.  Center by TRAIN mean; free train before
    loading val to keep the peak footprint of the wide multi_k cases down."""
    Xt = torch.as_tensor(np.ascontiguousarray(Xtr), dtype=torch.float32, device=device)
    mu = Xt.mean(0, keepdim=True)
    Xt -= mu
    _, _, V = torch.pca_lowrank(Xt, q=q, center=False, niter=6)   # V: (d, q)
    Zt = (Xt @ V).cpu().numpy()
    del Xt
    if device == "cuda":
        torch.cuda.empty_cache()
    Xv = torch.as_tensor(np.ascontiguousarray(Xva), dtype=torch.float32, device=device)
    Xv -= mu
    Zv = (Xv @ V).cpu().numpy()
    del Xv, V
    if device == "cuda":
        torch.cuda.empty_cache()
    return Zt, Zv


def probe(Ftr, ytr, Fva, yva, dev, feat, m, note, rows):
    r, _ = probes_torch.ridge_probe_t(Ftr.astype(np.float32), ytr,
                                      Fva.astype(np.float32), yva, device=dev)
    rows.append(dict(m=m, span=4 * m, feature=feat, dim=Ftr.shape[1],
                     r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"],
                     alpha=r["alpha"], note=note))
    print(f"[A] m={m} {feat:16s} d={Ftr.shape[1]:6d} xyz={r['r2_xyz']:.3f} "
          f"rot={r['r2_rot']:.3f} cont6={r['r2_cont6']:.3f}  ({note})", flush=True)
    return r


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"[A] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)

    index = data.build_index()
    wf = np.load(os.path.join(data.CACHE_DIR, "wfeats_clean.npy"))     # (N, 9408)
    ref = pd.read_csv(os.path.join(data.RESULTS_DIR, "exp_wan_temporal.csv"))
    print(f"[A] wf {wf.shape}; ref rows {len(ref)}", flush=True)

    rows = []
    zlast_xyz = {}
    for m in MS:
        stride = 4 * m
        it, jt, yct, _ = data.build_pairs(index, stride, "train")
        iv, jv, ycv, _ = data.build_pairs(index, stride, "val")
        assert (jt == it + stride).all() and (jv == iv + stride).all()
        offs = {k: np.unique(np.linspace(0, 4 * m, k).round().astype(int)) for k in KS[m]}
        for k in KS[m]:
            assert len(offs[k]) == k, (k, offs[k])   # integer offsets stay distinct

        def build(i, kind):
            if kind == "single":
                return wf[i]
            if kind == "diff":
                return wf[i + stride] - wf[i]
            if kind == "pair":
                return np.concatenate([wf[i + 4 * g] for g in range(m + 1)], axis=1)

        # --- reference rows recomputed from wf: must match exp_wan_temporal.csv ---
        for kind in ("single", "diff", "pair"):
            r = probe(build(it, kind), yct, build(iv, kind), ycv, dev, kind, m,
                      "recomputed-ref", rows)
            rr = ref[(ref.m == m) & (ref.feature == kind)]
            if len(rr):
                d = abs(float(rr.iloc[0].r2_xyz) - r["r2_xyz"])
                assert d < 0.02, f"split mismatch: {kind} m={m} recomputed {r['r2_xyz']:.3f} " \
                                 f"vs published {float(rr.iloc[0].r2_xyz):.3f} (d={d:.3f})"

        # --- the frame-count null ---
        for k in KS[m]:
            o = offs[k]
            Xtr = np.concatenate([wf[it + off] for off in o], axis=1)
            Xva = np.concatenate([wf[iv + off] for off in o], axis=1)
            note = f"frames@{list(map(int, o))}"
            if Xtr.shape[1] <= RAW_CAP:
                probe(Xtr, yct, Xva, ycv, dev, f"multi_{k}", m, note, rows)
            else:
                print(f"[A] multi_{k} raw d={Xtr.shape[1]} > cap {RAW_CAP}; pca-only", flush=True)
            Ztr, Zva = pca_project(Xtr, Xva, PCA_DIM, dev)
            probe(Ztr, yct, Zva, ycv, dev, f"multi_{k}_pca{PCA_DIM}", m,
                  note + f"|pca{PCA_DIM}", rows)
            del Xtr, Xva

        # --- fused reference copied from the published temporal CSV ---
        for kind in ("zlast", "video"):
            rr = ref[(ref.m == m) & (ref.feature == kind)]
            if len(rr):
                rr = rr.iloc[0]
                rows.append(dict(m=m, span=4 * m, feature=kind, dim=int(rr.dim),
                                 r2_xyz=float(rr.r2_xyz), r2_rot=float(rr.r2_rot),
                                 r2_cont6=float(rr.r2_cont6), alpha=float(rr.alpha),
                                 note="fused-from-exp_wan_temporal"))
                if kind == "zlast":
                    zlast_xyz[m] = float(rr.r2_xyz)

    df = pd.DataFrame(rows)
    out = os.path.join(data.RESULTS_DIR, "exp5c_framecount_null.csv")
    df.to_csv(out, index=False)
    print(f"\n[A] wrote {out}\n", flush=True)

    # --- verdict ---
    print("=" * 72)
    print("VERDICT -- frame count vs learned fusion (xyz R2)")
    print("=" * 72)
    for m in MS:
        zl = zlast_xyz.get(m)
        sub = df[df.m == m]
        diff = float(sub[sub.feature == "diff"].r2_xyz.iloc[0])
        kmax = max(KS[m])
        for tag in (f"multi_{kmax}_pca{PCA_DIM}", f"multi_{kmax}"):
            hit = sub[sub.feature == tag]
            if not len(hit):
                continue
            mv = float(hit.r2_xyz.iloc[0])
            gap_total = zl - diff
            gap_remain = zl - mv
            frac = 1 - gap_remain / gap_total if gap_total > 1e-9 else float("nan")
            reaches = gap_remain <= TOL
            print(f"m={m} span={4*m}: diff {diff:.3f} -> {tag} {mv:.3f} -> zlast {zl:.3f} | "
                  f"multi reaches zlast: {reaches} (gap {gap_remain:+.3f}); "
                  f"frame-count explains {frac*100:.0f}% of the diff->zlast gap")
        if zl is not None:
            print(f"    -> Pillar 1 (learned temporal fusion) "
                  f"{'NOT supported' if (zl - float(sub[sub.feature==f'multi_{kmax}_pca{PCA_DIM}'].r2_xyz.iloc[0])) <= TOL else 'SUPPORTED'} "
                  f"at m={m} (pca-width-matched test)")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
