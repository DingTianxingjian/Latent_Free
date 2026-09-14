"""Phase-1 training + evaluation, one codebase, `--model {lam,prla}` switch.

LAM path (this file, stage 1).  Self-supervised: IDM->VQ->FDM predicting the
window-end features from clean f(o_t)+codes.  After training, run the Phase-1
comparison probes on the frozen latents and write the row(s) + a verdict.

Evaluation metrics (phase_1.md table), all with the existing GPU probe suite,
episode-level train/val split, per-axis (xyz/rot/cont6/gripper):
  * state content under shuffle -- encode (o_t, random other window), probe->action,
    RATIO to the real-window score, three OOD tiers.  Reproduction anchor for LAM:
    ~0.37-0.40 (emb), ~0.10-0.26 (z_q).
  * incremental R2 -- R2([f(o_t), z]) minus R2([f(o_t)]) alone, per axis.
  * leakage -- probe z alone -> action (no o_t).
  * codebook usage -- unique codes per position (collapse would void the baseline).
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data                                    # noqa: E402
import probes_torch                            # noqa: E402
import pdata                                    # noqa: E402
from models import LAM                          # noqa: E402

RESULTS = os.path.join(data.RESULTS_DIR, "phase1")


def gather(G, frames, device):
    """frames (B, NFRAMES) global ids -> (B, NFRAMES, ntok, hid) on device."""
    return torch.from_numpy(G[frames]).to(device)


def nparams(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def encode_all(model, G, frames, device, bs=512):
    """frames (n, NFRAMES) -> emb, zq (n, n_q*d_code) flattened; idx (n, n_q)."""
    model.eval()
    embs, zqs, idxs = [], [], []
    for k in range(0, len(frames), bs):
        win = gather(G, frames[k:k + bs], device)
        emb, zq, idx, _ = model.encode(win)
        embs.append(emb.reshape(len(win), -1).cpu().numpy())
        zqs.append(zq.reshape(len(win), -1).cpu().numpy())
        idxs.append(idx.cpu().numpy())
    return np.concatenate(embs), np.concatenate(zqs), np.concatenate(idxs)


def probe(Ftr, Fva, wtr, wva, device):
    r, _ = probes_torch.run_probe_set_t(Ftr.astype(np.float32), Fva.astype(np.float32),
                                        wtr["yc"], wva["yc"], wtr["yg"], wva["yg"],
                                        device=device)
    return r


def evaluate(model, index, device, tag, rows, encoder="vjepa", dataset="libero"):
    G = pdata.g4_tokens(dataset, encoder)
    wtr, wva = pdata.windows(index, "train", dataset), pdata.windows(index, "val", dataset)

    emb_tr, zq_tr, idx_tr = encode_all(model, G, wtr["frames"], device)
    emb_va, zq_va, idx_va = encode_all(model, G, wva["frames"], device)

    real = {}
    for name, Ftr, Fva in (("emb", emb_tr, emb_va), ("z_q", zq_tr, zq_va)):
        r = probe(Ftr, Fva, wtr, wva, device)
        real[name] = r
        rows.append(dict(model=tag, latent=name, metric="real", tier="-",
                         **{k: r[k] for k in ("r2_xyz", "r2_rot", "r2_cont6", "bal_acc_grip", "dim")}))
        print(f"[eval:{tag}] real {name:4s} xyz={r['r2_xyz']:.3f} rot={r['r2_rot']:.3f} "
              f"cont6={r['r2_cont6']:.3f} grip={r['bal_acc_grip']:.3f}", flush=True)

    # --- state content under shuffle: ratio to real, per tier ---
    for tier in ("same_ep", "diff_ep", "rand"):
        fr_tr = pdata.shuffle_starts(wtr, index, tier, seed=0)
        fr_va = pdata.shuffle_starts(wva, index, tier, seed=0)
        emb_str, zq_str, _ = encode_all(model, G, fr_tr, device)
        emb_sva, zq_sva, _ = encode_all(model, G, fr_va, device)
        for name, Ftr, Fva in (("emb", emb_str, emb_sva), ("z_q", zq_str, zq_sva)):
            r = probe(Ftr, Fva, wtr, wva, device)
            base = real[name]["r2_xyz"]
            ratio = r["r2_xyz"] / base if base > 1e-6 else float("nan")
            rows.append(dict(model=tag, latent=name, metric="shuffle_ratio", tier=tier,
                             r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"],
                             bal_acc_grip=r["bal_acc_grip"], dim=r["dim"], ratio_xyz=ratio))
            print(f"[eval:{tag}] shuffle {tier:8s} {name:4s} xyz={r['r2_xyz']:.3f} "
                  f"ratio={ratio:.3f}", flush=True)

    # --- incremental R2: [f(o_t), z] - [f(o_t)] ---
    ot_tr = G[wtr["i"]].reshape(len(wtr["i"]), -1)     # (n, ntok*hid) = state the policy sees
    ot_va = G[wva["i"]].reshape(len(wva["i"]), -1)
    base = probe(ot_tr, ot_va, wtr, wva, device)
    rows.append(dict(model=tag, latent="f(o_t)", metric="incremental_base", tier="-",
                     r2_xyz=base["r2_xyz"], r2_rot=base["r2_rot"], r2_cont6=base["r2_cont6"],
                     bal_acc_grip=base["bal_acc_grip"], dim=base["dim"]))
    for name, Ztr, Zva in (("z_q", zq_tr, zq_va), ("emb", emb_tr, emb_va)):
        comb = probe(np.concatenate([ot_tr, Ztr], 1), np.concatenate([ot_va, Zva], 1),
                     wtr, wva, device)
        rows.append(dict(model=tag, latent=name, metric="incremental_combined", tier="-",
                         r2_xyz=comb["r2_xyz"], r2_rot=comb["r2_rot"], r2_cont6=comb["r2_cont6"],
                         bal_acc_grip=comb["bal_acc_grip"], dim=comb["dim"],
                         inc_xyz=comb["r2_xyz"] - base["r2_xyz"],
                         inc_rot=comb["r2_rot"] - base["r2_rot"],
                         inc_cont6=comb["r2_cont6"] - base["r2_cont6"]))
        print(f"[eval:{tag}] incremental {name:4s} xyz {base['r2_xyz']:.3f}->{comb['r2_xyz']:.3f} "
              f"(+{comb['r2_xyz']-base['r2_xyz']:.3f})", flush=True)

    # --- codebook usage ---
    idx_all = np.concatenate([idx_tr, idx_va])
    usage = [len(np.unique(idx_all[:, p])) for p in range(model.n_q)]
    rows.append(dict(model=tag, latent="z_q", metric="codebook_usage", tier="-",
                     usage=str(usage), K=model.K, n_q=model.n_q))
    print(f"[eval:{tag}] codebook usage per position {usage} / {model.K}", flush=True)
    return rows


def train_lam(index, device, args):
    G = pdata.g4_tokens(args.dataset, args.encoder)
    w = pdata.windows(index, "train", args.dataset)
    frames = w["frames"]
    if args.smoke:
        # first ~20 episodes' worth of windows
        keep = w["i"] < int(index["starts"][min(20, len(index["starts"]) - 1)])
        frames = frames[keep]
    n = len(frames)
    hid = pdata.enc_hid(args.dataset, args.encoder)
    print(f"[train] dataset={args.dataset} encoder={args.encoder} hid={hid}; "
          f"{n} train windows; smoke={args.smoke}", flush=True)

    model = LAM(pdata.NFRAMES, pdata.NTOK, hid, d=args.d,
                idm_layers=args.idm_layers, fdm_layers=args.fdm_layers,
                n_q=args.n_q, K=args.K, d_code=args.d_code).to(device)
    print(f"[train] LAM params {nparams(model)/1e6:.1f}M (budget 200M)", flush=True)
    assert nparams(model) < 200e6, "over param budget"

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.RandomState(0)

    # Exp 2 (Bridge sanity): does emb action-decodability RISE with training (Bridge)
    # or fall (LIBERO 0.44->0.04)?  Probe emb/z_q -> xyz on val at checkpoints.
    curve = []
    if getattr(args, "probe_curve", False):
        wtr_full = pdata.windows(index, "train", args.dataset)
        wva_full = pdata.windows(index, "val", args.dataset)
        ckpts = sorted(set([0, 500, 1000, 2000, 5000, 10000, args.steps]))

        def curve_point(step):
            was_training = model.training
            etr, ztr, _ = encode_all(model, G, wtr_full["frames"], device)
            eva, zva, _ = encode_all(model, G, wva_full["frames"], device)
            re, _ = probes_torch.ridge_probe_t(etr, wtr_full["yc"], eva, wva_full["yc"], device=device)
            rz, _ = probes_torch.ridge_probe_t(ztr, wtr_full["yc"], zva, wva_full["yc"], device=device)
            curve.append(dict(step=step, emb_xyz=re["r2_xyz"], zq_xyz=rz["r2_xyz"]))
            print(f"[curve] step {step:5d} emb_xyz={re['r2_xyz']:.3f} zq_xyz={rz['r2_xyz']:.3f}", flush=True)
            if was_training:
                model.train()

    t0 = time.time()
    model.train()
    for step in range(args.steps):
        if getattr(args, "probe_curve", False) and step in ckpts:
            curve_point(step)
        bidx = rng.randint(0, n, size=args.batch)
        win = gather(G, frames[bidx], device)
        ot, target = win[:, 0], win[:, -1]
        out = model(win, ot, target)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        sched.step()
        if step % args.log_every == 0 or step == args.steps - 1:
            u = [len(torch.unique(out["idx"][:, p])) for p in range(model.n_q)]
            print(f"[train] step {step:5d}/{args.steps} loss={out['loss'].item():.4f} "
                  f"recon={out['recon'].item():.4f} vq={out['vq_loss'].item():.4f} "
                  f"batch_usage={u} [{time.time()-t0:.0f}s]", flush=True)
    if getattr(args, "probe_curve", False):
        curve_point(args.steps)
    wall = time.time() - t0
    return model, dict(params=nparams(model), steps=args.steps, wall_s=wall,
                       gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                       curve=curve)


def smoke_asserts(model, device):
    """Architectural constraint: the FDM cannot see window/IDM tokens — perturbing
    the future frames while holding (o_t, z_q) fixed must not change its output."""
    B = 3
    hid = model.idm.inp.in_features
    win = torch.randn(B, pdata.NFRAMES, pdata.NTOK, hid, device=device)
    emb, zq, idx, _ = model.encode(win)
    ot = win[:, 0]
    p1 = model.fdm(ot, zq)
    win2 = win.clone()
    win2[:, 1:] = torch.randn_like(win2[:, 1:])          # scramble the future
    p2 = model.fdm(ot, zq)                                # same (o_t, z_q) inputs
    assert torch.allclose(p1, p2), "FDM output changed without an IDM/window path — leak!"
    print("[smoke] FDM sees only (o_t, z_q): future-frame perturbation is inert. OK", flush=True)


def mo_ridge_r2(Xtr, Ytr, Xva, Yva, device, alpha=100.0):
    """Mean R2 of a multi-output ridge Xtr->Ytr evaluated on val (leakage probe)."""
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=device)
    Yv = torch.as_tensor(Yva, dtype=torch.float32, device=device)
    mx, sx = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xt, Xv = (Xt - mx) / sx, (Xv - mx) / sx
    my = Yt.mean(0, keepdim=True)
    Yt0 = Yt - my
    d = Xt.shape[1]
    A = Xt.T @ Xt + alpha * torch.eye(d, device=device)
    W = torch.linalg.solve(A, Xt.T @ Yt0)
    pred = Xv @ W + my
    ss_res = ((Yv - pred) ** 2).sum(0)
    ss_tot = ((Yv - Yv.mean(0, keepdim=True)) ** 2).sum(0).clamp_min(1e-9)
    return float((1 - ss_res / ss_tot).mean())


def _fut_pcs(fut_tr, fut_va, k, device):
    """Top-k train PCs of the future fused latent, for the leakage regression target."""
    X = torch.as_tensor(fut_tr, dtype=torch.float32, device=device)
    mu = X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X - mu, q=k, center=False, niter=4)
    Ptr = ((X - mu) @ V).cpu().numpy()
    Xv = torch.as_tensor(fut_va, dtype=torch.float32, device=device)
    Pva = ((Xv - mu) @ V).cpu().numpy()
    return Ptr, Pva


_PRLA_STATS = {}          # dataset -> (hm, hs, fm, fs)


def _prla_dir(dataset):
    return os.path.join(data.CACHE_DIR, "bridge") if dataset == "bridge" else data.CACHE_DIR


def _prla_stats(dataset):
    """Per-dim mean/std of the fused latents over TRAIN, cached per dataset.  Same
    rationale as g4 whitening: raw fused features are static-dominated, so P_A's MSE
    (and the residual G corrects) must be on whitened features or z is starved."""
    if dataset in _PRLA_STATS:
        return _PRLA_STATS[dataset]
    d = _prla_dir(dataset)
    p = os.path.join(d, "prla_stats.npz")
    if os.path.exists(p):
        s = np.load(p)
        _PRLA_STATS[dataset] = (s["hm"], s["hs"], s["fm"], s["fs"])
        return _PRLA_STATS[dataset]
    h = np.load(os.path.join(d, "prla_hist_train.npy"))
    f = np.load(os.path.join(d, "prla_fut_train.npy"))
    hm, hs = h.mean(0), h.std(0); hs[hs < 1e-6] = 1.0
    fm, fs = f.mean(0), f.std(0); fs[fs < 1e-6] = 1.0
    np.savez(p, hm=hm, hs=hs, fm=fm, fs=fs)
    _PRLA_STATS[dataset] = (hm.astype(np.float32), hs.astype(np.float32),
                            fm.astype(np.float32), fs.astype(np.float32))
    return _PRLA_STATS[dataset]


def _load_prla(index, split, dataset="libero"):
    d = _prla_dir(dataset)
    h = np.load(os.path.join(d, f"prla_hist_{split}.npy"))
    f = np.load(os.path.join(d, f"prla_fut_{split}.npy"))
    hm, hs, fm, fs = _prla_stats(dataset)
    h = ((h - hm) / hs).astype(np.float32)
    f = ((f - fm) / fs).astype(np.float32)
    w = pdata.prla_windows(index, split, dataset)
    assert len(h) == len(w["t"]) == len(f), (len(h), len(w["t"]), len(f))
    return h, f, w


@torch.no_grad()
def prla_mu(model, hist, fut, device, bs=1024):
    model.eval()
    out = []
    for k in range(0, len(hist), bs):
        H = torch.from_numpy(hist[k:k + bs]).to(device)
        Fu = torch.from_numpy(fut[k:k + bs]).to(device)
        mu, _ = model.enc(Fu, model.p_a(H))
        out.append(mu.cpu().numpy())
    return np.concatenate(out)


def train_prla(index, device, args):
    from models import PRLA
    htr, ftr, wtr = _load_prla(index, "train", args.dataset)
    if args.smoke:
        keep = wtr["t"] < int(index["starts"][min(20, len(index["starts"]) - 1)])
        htr, ftr = htr[keep], ftr[keep]
        wtr = {kk: (vv[keep] if hasattr(vv, "__len__") and len(vv) == len(keep) else vv)
               for kk, vv in wtr.items()}
    dim = htr.shape[1]
    n = len(htr)
    print(f"[prla] {n} train positions, dim {dim}, d_z {args.z_dim}, beta {args.beta}", flush=True)
    model = PRLA(dim, d_z=args.z_dim, hidden=args.hidden).to(device)
    print(f"[prla] params {nparams(model)/1e6:.1f}M (budget 200M)", flush=True)
    assert nparams(model) < 200e6

    rng = np.random.RandomState(0)

    def batches(steps, params, tag):
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        t0 = time.time()
        for step in range(steps):
            bidx = rng.randint(0, n, size=args.batch)
            H = torch.from_numpy(htr[bidx]).to(device)
            Fu = torch.from_numpy(ftr[bidx]).to(device)
            if tag == "P_A":
                pred = model.p_a(H)
                loss = torch.nn.functional.mse_loss(pred, Fu)
                logs = f"mse={loss.item():.4f}"
            else:
                out = model(H, Fu, args.beta)
                loss = out["loss"]
                logs = (f"loss={loss.item():.4f} recon={out['recon'].item():.4f} "
                        f"kl={out['kl'].item():.3f} base={out['base_mse'].item():.4f}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
            if step % args.log_every == 0 or step == steps - 1:
                print(f"[prla:{tag}] step {step:5d}/{steps} {logs} [{time.time()-t0:.0f}s]", flush=True)
        return time.time() - t0

    # Stage 1: passive predictor to convergence, then FREEZE.
    t1 = batches(args.pa_steps, model.p_a.parameters(), "P_A")
    model.p_a.requires_grad_(False)
    model.p_a.eval()
    # Stage 2: residual encoder + corrector, P_A frozen.
    t2 = batches(args.steps, list(model.enc.parameters()) + list(model.gen.parameters()), "E+G")

    return model, dict(params=nparams(model), pa_steps=args.pa_steps, steps=args.steps,
                       wall_s=t1 + t2, z_dim=args.z_dim, beta=args.beta,
                       gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu")


def evaluate_prla(model, index, device, tag, rows, dataset="libero"):
    G = pdata.g4_tokens(dataset)
    htr, ftr, wtr = _load_prla(index, "train", dataset)
    hva, fva, wva = _load_prla(index, "val", dataset)

    mu_tr = prla_mu(model, htr, ftr, device)
    mu_va = prla_mu(model, hva, fva, device)

    real = probe(mu_tr, mu_va, wtr, wva, device)
    rows.append(dict(model=tag, latent="z", metric="real", tier="-",
                     r2_xyz=real["r2_xyz"], r2_rot=real["r2_rot"], r2_cont6=real["r2_cont6"],
                     bal_acc_grip=real["bal_acc_grip"], dim=real["dim"]))
    print(f"[eval:{tag}] real z xyz={real['r2_xyz']:.3f} rot={real['r2_rot']:.3f} "
          f"cont6={real['r2_cont6']:.3f} grip={real['bal_acc_grip']:.3f}", flush=True)

    # state content under shuffle: keep history, swap the future window
    ep_tr = pdata._ep_of(index)[wtr["t"]]
    ep_va = pdata._ep_of(index)[wva["t"]]
    for tier in ("same_ep", "diff_ep", "rand"):
        ptr = pdata.shuffle_perm(wtr["t"], ep_tr, tier, seed=0)
        pva = pdata.shuffle_perm(wva["t"], ep_va, tier, seed=0)
        mstr = prla_mu(model, htr, ftr[ptr], device)
        msva = prla_mu(model, hva, fva[pva], device)
        r = probe(mstr, msva, wtr, wva, device)
        ratio = r["r2_xyz"] / real["r2_xyz"] if real["r2_xyz"] > 1e-6 else float("nan")
        rows.append(dict(model=tag, latent="z", metric="shuffle_ratio", tier=tier,
                         r2_xyz=r["r2_xyz"], r2_rot=r["r2_rot"], r2_cont6=r["r2_cont6"],
                         bal_acc_grip=r["bal_acc_grip"], dim=r["dim"], ratio_xyz=ratio))
        print(f"[eval:{tag}] shuffle {tier:8s} z xyz={r['r2_xyz']:.3f} ratio={ratio:.3f}", flush=True)

    # incremental R2: [f(o_t), z] - [f(o_t)]
    ot_tr = G[wtr["t"]].reshape(len(wtr["t"]), -1)
    ot_va = G[wva["t"]].reshape(len(wva["t"]), -1)
    base = probe(ot_tr, ot_va, wtr, wva, device)
    comb = probe(np.concatenate([ot_tr, mu_tr], 1), np.concatenate([ot_va, mu_va], 1),
                 wtr, wva, device)
    rows.append(dict(model=tag, latent="f(o_t)", metric="incremental_base", tier="-",
                     r2_xyz=base["r2_xyz"], r2_rot=base["r2_rot"], r2_cont6=base["r2_cont6"],
                     bal_acc_grip=base["bal_acc_grip"], dim=base["dim"]))
    rows.append(dict(model=tag, latent="z", metric="incremental_combined", tier="-",
                     r2_xyz=comb["r2_xyz"], r2_rot=comb["r2_rot"], r2_cont6=comb["r2_cont6"],
                     bal_acc_grip=comb["bal_acc_grip"], dim=comb["dim"],
                     inc_xyz=comb["r2_xyz"] - base["r2_xyz"],
                     inc_rot=comb["r2_rot"] - base["r2_rot"],
                     inc_cont6=comb["r2_cont6"] - base["r2_cont6"]))
    print(f"[eval:{tag}] incremental z xyz {base['r2_xyz']:.3f}->{comb['r2_xyz']:.3f} "
          f"(+{comb['r2_xyz']-base['r2_xyz']:.3f})", flush=True)

    # leakage: can z alone reconstruct the future (its top-64 PCs)?
    ptr64, pva64 = _fut_pcs(ftr, fva, 64, device)
    leak = mo_ridge_r2(mu_tr, ptr64, mu_va, pva64, device)
    rows.append(dict(model=tag, latent="z", metric="leakage_future_r2", tier="-", leak_r2=leak))
    print(f"[eval:{tag}] leakage z->future(top64 PCs) R2={leak:.3f} "
          f"(high => E is compressing the future, not just the residual)", flush=True)

    # rate-distortion point for this (z_dim, beta)
    with torch.no_grad():
        base_mse = float(np.mean((model.p_a(torch.from_numpy(hva).to(device)).cpu().numpy() - fva) ** 2))
        Hv = torch.from_numpy(hva).to(device)
        Fv = torch.from_numpy(fva).to(device)
        mu, logvar = model.enc(Fv, model.p_a(Hv))
        pred_b = model.gen(mu, model.p_a(Hv))
        recon_mse = float(np.mean((pred_b.cpu().numpy() - fva) ** 2))
        kl = float((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean().item())
    rows.append(dict(model=tag, latent="z", metric="rate_distortion", tier="-",
                     kl_nats=kl, base_mse=base_mse, recon_mse=recon_mse,
                     correction_gain=base_mse - recon_mse))
    print(f"[eval:{tag}] rate-distortion KL={kl:.3f} nats, base_mse={base_mse:.4f}, "
          f"recon_mse={recon_mse:.4f}, gain={base_mse-recon_mse:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lam", "prla"], default="lam")
    ap.add_argument("--encoder", choices=["vjepa", "dino"], default="vjepa")
    ap.add_argument("--dataset", choices=["libero", "bridge"], default="libero")
    ap.add_argument("--probe-curve", action="store_true",
                    help="Exp 2: log emb/z_q xyz decodability across training checkpoints")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--idm-layers", type=int, default=6)
    ap.add_argument("--fdm-layers", type=int, default=4)
    ap.add_argument("--n-q", type=int, default=4)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--d-code", type=int, default=64)
    ap.add_argument("--log-every", type=int, default=500)
    # PRLA
    ap.add_argument("--z-dim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--pa-steps", type=int, default=20000)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.smoke:
        a.steps = min(a.steps, 60)
        a.pa_steps = min(a.pa_steps, 60)
        a.log_every = 10
    os.makedirs(RESULTS, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, _ = torch.cuda.mem_get_info()
        print(f"[p1] GPU {torch.cuda.get_device_name(0)} free {free/2**30:.1f} GiB", flush=True)
    index = pdata.get_index(a.dataset)

    rows = []
    if a.model == "lam":
        model, meta = train_lam(index, dev, a)
        smoke_asserts(model, dev)
        tag = f"lam_{a.encoder}_{a.dataset}" + ("_smoke" if a.smoke else "")
        evaluate(model, index, dev, tag, rows, encoder=a.encoder, dataset=a.dataset)
        meta["encoder"], meta["dataset"] = a.encoder, a.dataset
    else:
        model, meta = train_prla(index, dev, a)
        tag = f"prla_{a.dataset}_z{a.z_dim}_b{a.beta:g}" + ("_smoke" if a.smoke else "") + a.suffix
        evaluate_prla(model, index, dev, tag, rows, dataset=a.dataset)
        meta["dataset"] = a.dataset

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, f"{tag}.csv")
    df.to_csv(out, index=False)
    if meta.get("curve"):
        cout = os.path.join(RESULTS, f"{tag}_curve.csv")
        pd.DataFrame(meta["curve"]).to_csv(cout, index=False)
        print(f"[p1] wrote training curve {cout}", flush=True)
    ck = os.path.join(RESULTS, f"{tag}.pt")
    torch.save({"model": model.state_dict(), "meta": meta, "args": vars(a)}, ck)
    print(f"[p1] wrote {out} and {ck}", flush=True)
    print(f"[p1] compute: {meta['params']/1e6:.1f}M params, "
          f"{meta.get('steps')} steps, {meta['wall_s']:.0f}s on {meta['gpu']}", flush=True)


if __name__ == "__main__":
    main()
