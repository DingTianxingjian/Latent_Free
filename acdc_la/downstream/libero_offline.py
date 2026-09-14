"""Downstream ACCURACY on LIBERO-spatial demonstrations (offline, the pre-simulator numbers a VLA paper reports).

Data: 10 tasks x 50 demos (latent_free/data/LIBERO), episode-disjoint 80/20 split (data.episode_split, seed 0).
A "step" is a frame t with a K*S-frame future (S = stride 4, K = 4 segments = 16 frames).  Instruction = task one-hot
(libero_spatial has exactly 10 instructions, so a one-hot carries what a T5 embedding of them would).

Two heads, trained on the train split from frozen inputs, read on held-out episodes:
  (P) latent policy      pi(z | g4(o_t), task)           what a VLA does: predict the latent action from what it sees
  (D) action decoder     g(a_{t:t+16} | z, g4(o_t))       UniVLA's action decoder: latent -> real action chunk
Metrics (all on held-out episodes):
  latent_acc     top-1 over latent tokens = the 'Action Token Accuracy' UniVLA's code logs while training the VLA (prismatic/training/
                 strategies/base_strategy.py; the paper mentions it once, descriptively, Sec. IV-B, with no numbers).  UniVLA: its 4 VQ indices per pair
                 (16-way), 16 tokens per step.  Ours: z is continuous, so each of the n_cells cell codes is k-means
                 quantised (16 centroids fit on train) and the policy's prediction is scored by nearest centroid.
                 The majority-class accuracy is reported next to it (a collapsed codebook scores 1.0).
  latent_cos     cosine between predicted and true latent (continuous view of the same head)
  action_tok_acc OpenVLA / UniVLA action-token accuracy: every action dim of the 16-step chunk discretised into 256
                 uniform bins between the train 1st and 99th percentiles; fraction of correct bins.
                 oracle   = decoder fed the TRUE latent (decodability of the code)
                 chain    = decoder fed the POLICY's latent (the deployable path, the offline stand-in for success rate)
                 nolatent = g4(o_t) + task -> chunk, no latent at all (the floor every latent must beat)
  action_l1      mean |a - a^| on the standardised chunk, same three variants
Latents compared on the SAME steps: ours (--tags, Stage-I/III tokenizers; K=1 tokenizers applied per segment and
concatenated), UniVLA official stage-2 z_q (4 pairs concatenated), and the no-latent floor.
    python -m acdc_la.downstream.libero_offline --tags s1_acdc s1_mean s1_acdc_1ep
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from ..paths import RESULTS, CKPT, LF, CLAUDE_SURVEY, IMG
from ..common import features as Fz
from ..common.model import load_ckpt, legacy_path
for p in (LF, os.path.join(LF, "diagnostics"), CLAUDE_SURVEY):
    sys.path.insert(0, p)
import data as libero                                  # noqa: E402  latent_free/data.py (LIBERO hdf5 access, split)

S, K = 4, 4
N_TASKS, BINS = 10, 256


@torch.no_grad()
def frame_features(dino, dev):
    """g4 features of every LIBERO frame (N,16,768) fp16 on CPU, plus the episode index."""
    idx = np.load(os.path.join(libero.CACHE_DIR, "index.npz")); eps = libero.list_episodes()
    N = int(idx["lengths"].sum()); G = torch.empty(N, 16, 768, dtype=torch.float16); t0 = time.time()
    cache = os.path.join(LF, "cache", "libero_g4.pt")                       # frozen features: computed once (1.6 GB)
    if os.path.exists(cache):
        G = torch.load(cache); assert G.shape[0] == N, (G.shape, N); return G, idx
    for e, ep in enumerate(eps):
        fr = torch.from_numpy(libero.load_episode_frames(ep)).to(dev).permute(0, 3, 1, 2).float().div_(255.)
        fr = F.interpolate(fr, size=(IMG, IMG), mode="bilinear", align_corners=False)
        assert ep["demo"] == str(idx["demo"][e]) and os.path.basename(ep["path"]) == os.path.basename(str(idx["path"][e])), (e, ep)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            out = [Fz.dino_g4(dino, fr[i:i + 128]).float() for i in range(0, len(fr), 128)]
        s = int(idx["starts"][e]); G[s:s + len(fr)] = torch.cat(out).half().cpu()
        if (e + 1) % 100 == 0: print(f"[libero] features {e+1}/{len(eps)} {time.time()-t0:.0f}s", flush=True)
    torch.save(G, cache + ".tmp"); os.replace(cache + ".tmp", cache)
    return G, idx


def steps_of(idx):
    """(frame t, episode e, task) for every t with K*S frames of future inside the episode."""
    T_, E_ = [], []
    for e, (s, L) in enumerate(zip(idx["starts"], idx["lengths"])):
        for t in range(int(s), int(s + L) - K * S): T_.append(t); E_.append(e)
    return np.array(T_), np.array(E_)


@torch.no_grad()
def our_latents(model, dstats, G, T_, dev, bs=512):
    """Clip code of the T=K*S+1 window starting at every step (K=1 tokenizers: per-segment codes concatenated)."""
    Z = []; basis = Fz.dct_basis(model.K, model.n_cells // 4, dev); one = Fz.dct_basis(1, 1, dev)
    for i in range(0, len(T_), bs):
        t = torch.from_numpy(T_[i:i + bs]); rows = t[:, None] + torch.arange(0, K * S + 1, S)[None]
        g4 = G[rows].to(dev).float(); dF = (g4[:, 1:] - g4[:, :-1]) / dstats
        if model.K == K: z = model.encode(Fz.effect_cells(Fz.quadrants(dF), basis))[0]
        else: z = torch.cat([model.encode(Fz.effect_cells(Fz.quadrants(dF[:, k:k + 1]), one))[0] for k in range(K)], 1)
        Z.append(z.flatten(1).cpu())
    return torch.cat(Z).numpy()


@torch.no_grad()
def univla_latents(dev, T_, dino_frames_fn, bs=64, arm="official_stage2"):
    """Official stage-2 z_q for the K stride pairs of every step: continuous (n, K*4*128) and indices (n, K*4)."""
    import eval_rala2 as E
    model, T2, _ = E.load_arm(arm, dev); cb = model.vq_action.codebook.weight            # (16,128)
    Z, I = [], []
    for i in range(0, len(T_), bs):
        t = T_[i:i + bs]; zs, ids = [], []
        for k in range(K):
            x = dino_frames_fn(np.stack([t + k * S, t + (k + 1) * S], 1))                            # (B,2,3,224,224)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                zq = model.vq_encode(x)["z_q"].float().reshape(len(t), 4, -1)                        # (B,4,128)
            ids.append(torch.cdist(zq, cb[None].expand(len(t), -1, -1)).argmin(-1)); zs.append(zq.flatten(1))
        Z.append(torch.cat(zs, 1).cpu()); I.append(torch.cat(ids, 1).cpu())
        if (i // bs) % 50 == 0: print(f"[libero] univla {i}/{len(T_)}", flush=True)
    return torch.cat(Z).numpy(), torch.cat(I).numpy()


@torch.no_grad()
def decoder_feats(model, G, T_, Z, dev, bs=512):
    """Reuse the (LATP-pretrained) transport decoder at deployment: h = quadrant-pooled hidden state of the 16 state
    tokens of D(f0, c=0, z, u=prior mean 0) -> (n, 4*768).  c = 0 is what the tokenizer saw for sources without a
    language cache; u = 0 is the prior mean -- the REAL recipient residual (which needs the future) is never used.
    Feeding the same z through the Stage-I decoder vs the LATP decoder isolates what LATP adds downstream."""
    H = []
    for i in range(0, len(T_), bs):
        f0 = G[torch.from_numpy(T_[i:i + bs])].to(dev).float(); z = torch.as_tensor(Z[i:i + bs], dtype=torch.float32, device=dev).view(len(f0), model.n_cells, model.d)
        u = torch.zeros(len(f0), model.d_u, device=dev); c = torch.zeros(len(f0), 768, device=dev)
        tok = torch.cat([model.f0_in(f0), model.z_in(z), model.u_in(u)[:, None], model.c_in(c)[:, None]], 1) + model.pos
        h = model.dec(tok)[:, :16]
        H.append(Fz.quadrants(h).flatten(1).cpu())
    return torch.cat(H).numpy()


@torch.no_grad()
def effect_feats(model, G, T_, Z, dev, bs=512):
    """The transport decoder's OUTPUT effect dF^ = D(f0, c=0, z, u=0), quadrant-pooled per segment -> (n, K*4*768).
    This is the quantity LATP trains (swap realism + cycle act on dF^), so it is the right downstream interface for a
    Stage-III model; the hidden state used by `decoder_feats` was never optimised for anything (review 2026-09-12)."""
    H = []
    for i in range(0, len(T_), bs):
        f0 = G[torch.from_numpy(T_[i:i + bs])].to(dev).float(); z = torch.as_tensor(Z[i:i + bs], dtype=torch.float32, device=dev).view(len(f0), model.n_cells, model.d)
        u = torch.zeros(len(f0), model.d_u, device=dev); c = torch.zeros(len(f0), 768, device=dev)
        H.append(Fz.quadrants(model.decode(f0, c, z, u)).flatten(1).cpu())
    return torch.cat(H).numpy()


def mlp(i, o, h=512):
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


def fit_net(X, Y, dev, steps=3000, bs=512, ce_classes=None, seed=0):
    """Train the MLP head and return a predictor callable (numpy in -> numpy out, un-standardised / argmax)."""
    torch.manual_seed(seed); X = torch.as_tensor(X, dtype=torch.float32, device=dev)
    if ce_classes is None:
        Y = torch.as_tensor(Y, dtype=torch.float32, device=dev); mu, sd = Y.mean(0), Y.std(0) + 1e-6; Yn = (Y - mu) / sd
        net = mlp(X.shape[1], Y.shape[1]).to(dev)
    else:
        Y = torch.as_tensor(Y, dtype=torch.long, device=dev); m = Y.shape[1]; net = mlp(X.shape[1], m * ce_classes).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=dev).manual_seed(seed)
    for s in range(steps):
        b = torch.randint(0, len(X), (bs,), device=dev, generator=g); out = net(X[b])
        loss = F.mse_loss(out, Yn[b]) if ce_classes is None else F.cross_entropy(out.view(-1, ce_classes), Y[b].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    net.eval()
    @torch.no_grad()
    def predict(Xq):
        Xq = torch.as_tensor(np.asarray(Xq, np.float32), device=dev); P = torch.cat([net(Xq[i:i + 4096]) for i in range(0, len(Xq), 4096)])
        return (P * sd + mu).cpu().numpy() if ce_classes is None else P.view(len(Xq), -1, ce_classes).argmax(-1).cpu().numpy()
    return predict


def fit(X, Y, Xte, dev, steps=3000, bs=512, ce_classes=None, seed=0):
    """Small MLP head; regression (MSE on standardised Y) or classification (Y int (n, m) -> logits (n, m, C))."""
    torch.manual_seed(seed); X = torch.as_tensor(X, dtype=torch.float32, device=dev); Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    if ce_classes is None:
        Y = torch.as_tensor(Y, dtype=torch.float32, device=dev); mu, sd = Y.mean(0), Y.std(0) + 1e-6; Yn = (Y - mu) / sd
        net = mlp(X.shape[1], Y.shape[1]).to(dev)
    else:
        Y = torch.as_tensor(Y, dtype=torch.long, device=dev); m = Y.shape[1]; net = mlp(X.shape[1], m * ce_classes).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator(device=dev).manual_seed(seed)
    for s in range(steps):
        b = torch.randint(0, len(X), (bs,), device=dev, generator=g); out = net(X[b])
        loss = F.mse_loss(out, Yn[b]) if ce_classes is None else F.cross_entropy(out.view(-1, ce_classes), Y[b].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        P = torch.cat([net(Xte[i:i + 4096]) for i in range(0, len(Xte), 4096)])
        return (P * sd + mu).cpu().numpy() if ce_classes is None else P.view(len(Xte), -1, ce_classes).argmax(-1).cpu().numpy()


def kmeans(X, k=16, iters=30, seed=0):
    rs = np.random.RandomState(seed); C = X[rs.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (a == j).any(): C[j] = X[a == j].mean(0)
    return C


def assign(X, C): return ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)


def token_bins(A_tr, A):
    lo, hi = np.quantile(A_tr, 0.01, 0), np.quantile(A_tr, 0.99, 0)
    return np.clip(((A - lo) / np.maximum(hi - lo, 1e-6) * (BINS - 1)).round(), 0, BINS - 1).astype(int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="*", default=["s1_acdc", "s1_mean"], help="Stage-I/III checkpoints (this repo's ckpt/ or legacy)")
    ap.add_argument("--no-univla", action="store_true"); ap.add_argument("--univla-arm", default="official_stage2", help="official_stage2 or a rala2 tag (s2mix_1ep = their recipe on our data / budget)"); ap.add_argument("--no-decoder", action="store_true", help="skip the +T (decoder-reuse) arms"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--out", default="libero_offline")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    dino = Fz.load_dino(dev); G, idx = frame_features(dino, dev); T_, E_ = steps_of(idx)
    is_val = idx["is_val"].astype(bool); tr, te = ~is_val[E_], is_val[E_]; task = idx["task_id"][E_]
    print(f"[libero] steps train {tr.sum()} test {te.sum()} ({time.time()-t0:.0f}s)", flush=True)
    A = idx["actions"]; chunk = np.stack([A[t:t + K * S].reshape(-1) for t in T_])                      # (n, 16*7)
    obs = Fz.quadrants(G[torch.from_numpy(T_)].float()).flatten(1).numpy()                              # (n, 3072) current-frame state
    onehot = np.eye(N_TASKS, dtype=np.float32)[task]; X_obs = np.concatenate([obs, onehot], 1)
    bins_te = token_bins(chunk[tr], chunk[te]); csd = chunk[tr].std(0) + 1e-6
    def action_metrics(P):
        return dict(tok_acc=float((token_bins(chunk[tr], P) == bins_te).mean()), l1=float((np.abs(P - chunk[te]) / csd).mean()))
    R = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "K": K, "S": S}
    R["nolatent"] = action_metrics(fit(X_obs[tr], chunk[tr], X_obs[te], dev, a.steps))
    print(f"[libero] no-latent floor: {R['nolatent']}", flush=True)
    latents = {}
    for tag in a.tags:
        p = os.path.join(CKPT, f"{tag}.pt"); p = p if os.path.exists(p) else legacy_path(tag)
        model, dstats, _ = load_ckpt(p, dev); model.eval(); latents[tag] = (our_latents(model, dstats, G, T_, dev), None, model.n_cells, model)
    if not a.no_univla:
        latents[f"univla_{a.univla_arm}"] = univla_latents(dev, T_, make_frame_reader(idx, dev), arm=a.univla_arm) + (K * 4, None)
    for name, (Z, I, ncell, model) in latents.items():
        r = {"dims": int(Z.shape[1])}
        # (D) action decoder with the TRUE latent
        XD = np.concatenate([Z, X_obs], 1); r["oracle"] = action_metrics(fit(XD[tr], chunk[tr], XD[te], dev, a.steps))
        # (P) latent policy, then the chain
        Zp = fit(X_obs[tr], Z[tr], X_obs[te], dev, a.steps)
        r["latent_cos"] = float(np.mean((Zp * Z[te]).sum(1) / (np.linalg.norm(Zp, axis=1) * np.linalg.norm(Z[te], axis=1) + 1e-9)))
        r["latent_r2"] = float(1 - ((Zp - Z[te]) ** 2).sum() / ((Z[te] - Z[tr].mean(0)) ** 2).sum())
        # latent accuracy: tokens = VQ indices (UniVLA) or k-means(16) per cell (ours)
        if I is not None:
            Ip = fit(X_obs[tr], I[tr], X_obs[te], dev, a.steps, ce_classes=16); r["latent_acc"] = float((Ip == I[te]).mean())
            maj = np.array([np.bincount(I[tr][:, j], minlength=16).argmax() for j in range(I.shape[1])]); r["latent_acc_majority"] = float((I[te] == maj).mean())
        else:
            d = Z.shape[1] // ncell; Zc, Zpc = Z.reshape(len(Z), ncell, d), Zp.reshape(len(Zp), ncell, d); acc, maj = [], []
            for c in range(ncell):
                C = kmeans(Zc[tr][:, c], 16); it, ip = assign(Zc[te][:, c], C), assign(Zpc[:, c], C)
                acc.append((it == ip).mean()); maj.append((it == np.bincount(assign(Zc[tr][:, c], C), minlength=16).argmax()).mean())
            r["latent_acc"], r["latent_acc_majority"] = float(np.mean(acc)), float(np.mean(maj))
        XDp = np.concatenate([Zp, X_obs[te]], 1); XDtr = np.concatenate([Z[tr], X_obs[tr]], 1)
        r["chain"] = action_metrics(fit(XDtr, chunk[tr], XDp, dev, a.steps))
        if model is not None and model.K == K and not a.no_decoder:
            # +T: the action head reads the transport decoder's hidden state instead of raw z (same z, this ckpt's decoder)
            Hz = decoder_feats(model, G, T_, Z, dev); Hp = decoder_feats(model, G, T_[te], Zp, dev)
            r["oracle+T"] = action_metrics(fit(np.concatenate([Hz[tr], X_obs[tr]], 1), chunk[tr], np.concatenate([Hz[te], X_obs[te]], 1), dev, a.steps))
            r["chain+T"] = action_metrics(fit(np.concatenate([Hz[tr], X_obs[tr]], 1), chunk[tr], np.concatenate([Hp, X_obs[te]], 1), dev, a.steps))
            # +E: action head on the decoder's OUTPUT effect.  oracle+E = true z through T; chain+E = policy's z through T.
            Ez = effect_feats(model, G, T_, Z, dev); Ep = effect_feats(model, G, T_[te], Zp, dev)
            r["oracle+E"] = action_metrics(fit(np.concatenate([Ez[tr], X_obs[tr]], 1), chunk[tr], np.concatenate([Ez[te], X_obs[te]], 1), dev, a.steps))
            r["chain+E"] = action_metrics(fit(np.concatenate([Ez[tr], X_obs[tr]], 1), chunk[tr], np.concatenate([Ep, X_obs[te]], 1), dev, a.steps))
            r["oracle+E_only"] = action_metrics(fit(Ez[tr], chunk[tr], Ez[te], dev, a.steps))          # effect alone, no f0: is dF^ itself action-readable?
        R[name] = r; print(f"[libero] {name}: {r}", flush=True)
        json.dump(R, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    print("%-18s %6s %11s %8s %13s %13s %13s %13s %13s %13s %13s" % ("latent", "dims", "latent_acc", "(maj)", "tok oracle", "tok chain", "l1 chain", "tok chain+T", "l1 chain+T", "l1 oracle+E", "l1 chain+E"))
    print("%-18s %6s %11s %8s %13.3f %13s %13s" % ("no latent", "-", "-", "-", R["nolatent"]["tok_acc"], "-", f"{R['nolatent']['l1']:.3f}"))
    for name, r in R.items():
        if isinstance(r, dict) and "oracle" in r:
            t = r.get("chain+T", {}); e = r.get("chain+E", {}); eo = r.get("oracle+E", {}); print("%-18s %6d %11.3f %8.3f %13.3f %13.3f %13.3f %13s %13s %13s %13s" % (name, r["dims"], r["latent_acc"], r["latent_acc_majority"], r["oracle"]["tok_acc"], r["chain"]["tok_acc"], r["chain"]["l1"],
                  f"{t['tok_acc']:.3f}" if t else "-", f"{t['l1']:.3f}" if t else "-", f"{eo['l1']:.3f}" if eo else "-", f"{e['l1']:.3f}" if e else "-"))
    print(f"[libero-done] {time.time()-t0:.0f}s", flush=True)


def make_frame_reader(idx, dev):
    """Global frame id -> resized frame tensor, via the hdf5 episodes (cached per episode)."""
    eps = libero.list_episodes(); starts = idx["starts"]; cache = {}
    def ep_of(g): return int(np.searchsorted(starts, g, side="right") - 1)
    def read(rows):
        out = torch.empty(rows.shape[0], rows.shape[1], 3, IMG, IMG, device=dev)
        for i, pair in enumerate(rows):
            for j, g in enumerate(pair):
                e = ep_of(g)
                if e not in cache:
                    if len(cache) > 4: cache.clear()
                    fr = torch.from_numpy(libero.load_episode_frames(eps[e])).to(dev).permute(0, 3, 1, 2).float().div_(255.)
                    cache[e] = F.interpolate(fr, size=(IMG, IMG), mode="bilinear", align_corners=False)
                out[i, j] = cache[e][int(g - starts[e])]
        return out
    return read


if __name__ == "__main__":
    main()
