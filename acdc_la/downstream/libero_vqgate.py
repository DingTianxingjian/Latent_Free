"""Go/no-go gate (review 2026-09-13): is a DISCRETE per-cell structured code of the EXISTING ACDC tokenizer a better
deployment target than its continuous posterior?  No tokenizer retraining.

  1. z* = s1_acdc_1ep code of every LIBERO step, 8 cells x 32; per-cell k-means with Kc codes fitted on train steps
  2. oracle      : residual head fed the DEQUANTISED code (centroids) -- must stay near the continuous oracle
  3. policy      : the same 4-frame history transformer, but a Kc-way classifier per cell (cross-entropy), same
                   oracle -> predicted curriculum; the head sees the MODE (straight-through argmax) when predicted
  4. chain       : head fed with the policy's mode (argmax centroid) or its expectation (softmax-weighted centroid)
Gate (PROTOCOL): chain L1 of a vq arm <= continuous `ours` of the same run - 0.015; otherwise no joint prior pretraining.
Arms: ours (continuous, same-run reference), vq16, vq64, vq256.   Output results/<out>.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from ..paths import RESULTS, CKPT
from ..common import features as Fz
from ..common.model import load_ckpt, legacy_path
from . import libero_offline as L
from .libero_offline import K, S, N_TASKS
from .libero_policy import HIST, hist_index, Policy, mlp, train_agent, predict


class VQAgent(nn.Module):
    def __init__(self, C):                                   # C (n_cells, Kc, d)
        super().__init__(); self.register_buffer("C", C); self.n_cells, self.Kc, d = C.shape
        self.bc = mlp(3072 + N_TASKS, K * S * 7); self.pi = Policy(self.n_cells * self.Kc); self.lat = mlp(3072 + self.n_cells * d, K * S * 7, zero_last=True)

    def logits(self, fq, task): return self.pi(fq, task).view(len(task), self.n_cells, self.Kc)
    def embed(self, p): return torch.einsum("bck,ckd->bcd", p, self.C).flatten(1)          # probs / one-hot -> dequantised code

    def forward(self, fq, task, z_for_head=None, mode="argmax"):
        onehot = F.one_hot(task, N_TASKS).float(); f_t = fq[:, -1].reshape(len(task), -1)
        a = self.bc(torch.cat([f_t, onehot], 1)); lg = self.logits(fq, task)
        if z_for_head is None:
            p = lg.softmax(-1)
            if mode == "argmax": p = F.one_hot(p.argmax(-1), self.Kc).float()
            z_for_head = self.embed(p)
        return a + self.lat(torch.cat([f_t, z_for_head], 1)), lg


def train_vq(feats, C, dev, a):
    tr = feats["tr"]; n_tr = int(tr.sum()); idx_tr = np.flatnonzero(tr)
    agent = VQAgent(torch.as_tensor(C, dtype=torch.float32)).to(dev)
    opt = torch.optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=0.01); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    chunk_mu = torch.as_tensor(feats["chunk"][tr].mean(0), device=dev); chunk_sd = torch.as_tensor(feats["chunk"][tr].std(0) + 1e-6, device=dev)
    g = torch.Generator().manual_seed(0); t0 = time.time(); log = {}
    for it in range(a.iters):
        b = idx_tr[torch.randint(0, n_tr, (a.batch,), generator=g).numpy()]
        fq = feats["FQ"][b].to(dev).float(); task = torch.as_tensor(feats["task"][b], device=dev)
        act = (torch.as_tensor(feats["chunk"][b], device=dev) - chunk_mu) / chunk_sd
        I = torch.as_tensor(feats["I"][b], device=dev); zq = torch.as_tensor(feats["Zstar"][b], device=dev)
        p_or = max(0.0, 1.0 - it / (0.6 * a.iters))
        lg = agent.logits(fq, task); ps = lg.softmax(-1)
        hard = F.one_hot(ps.argmax(-1), agent.Kc).float() + ps - ps.detach()               # straight-through mode
        use_star = (torch.rand(len(b), 1, device=dev) < p_or).float()
        z_mix = use_star * zq + (1 - use_star) * agent.embed(hard)
        pred, _ = agent(fq, task, z_for_head=z_mix)
        terms = dict(act=F.huber_loss(pred, act), code=F.cross_entropy(lg.reshape(-1, agent.Kc), I.reshape(-1)))
        loss = terms["act"] + a.lam_z * terms["code"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step(); sched.step()
        for k, v in terms.items(): log.setdefault(k, []).append(float(v))
        if (it + 1) % 1000 == 0: print(f"[vqgate:K{agent.Kc}] it {it+1} p={p_or:.2f} " + " ".join(f"{k}={np.mean(v[-1000:]):.4f}" for k, v in log.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
    agent.eval(); return agent, chunk_mu, chunk_sd


@torch.no_grad()
def predict_vq(agent, feats, sel, dev, mu, sd, mode="argmax", z_override=None, bs=2048):
    out, codes = [], []
    for b in np.array_split(np.flatnonzero(sel), max(1, sel.sum() // bs)):
        fq = feats["FQ"][b].to(dev).float(); task = torch.as_tensor(feats["task"][b], device=dev)
        z = None if z_override is None else torch.as_tensor(z_override[b], device=dev)
        pred, lg = agent(fq, task, z_for_head=z, mode=mode); out.append((pred * sd + mu).cpu().numpy()); codes.append(lg.argmax(-1).cpu().numpy())
    return np.concatenate(out), np.concatenate(codes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["ours", "vq64", "vq256", "pq4x16", "pq2x64", "pq4x64"]); ap.add_argument("--tokenizer", default="s1_acdc_1ep")
    ap.add_argument("--iters", type=int, default=8000); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--lam-z", type=float, default=1.0); ap.add_argument("--lam-e", type=float, default=0.0)
    ap.add_argument("--out", default="libero_vqgate")
    a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"; t0 = time.time()
    dino = Fz.load_dino(dev); G, idx = L.frame_features(dino, dev); T_, E_ = L.steps_of(idx)
    is_val = idx["is_val"].astype(bool); tr = ~is_val[E_]; te = is_val[E_]; task = idx["task_id"][E_].astype(np.int64)
    A = idx["actions"]; chunk = np.stack([A[t:t + K * S].reshape(-1) for t in T_]).astype(np.float32)
    H_idx = hist_index(T_, E_, idx); FQ = Fz.quadrants(G[torch.from_numpy(H_idx.reshape(-1))].float()).view(len(T_), HIST, 4, 768).half()
    p = os.path.join(CKPT, f"{a.tokenizer}.pt"); tok, dstats, _ = load_ckpt(p if os.path.exists(p) else legacy_path(a.tokenizer), dev); tok.eval()
    Z = L.our_latents(tok, dstats, G, T_, dev).astype(np.float32); n_cells0, d = tok.n_cells, tok.d
    feats = dict(FQ=FQ, F0=G[torch.from_numpy(T_)], task=task, chunk=chunk, tr=tr, te=te, Zstar=Z)
    csd = chunk[tr].std(0) + 1e-6
    def metrics(P): return dict(l1=float((np.abs(P - chunk[te]) / csd).mean()), tok_acc=float((L.token_bins(chunk[tr], P) == L.token_bins(chunk[tr], chunk[te])).mean()))
    R = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "iters": a.iters, "tokenizer": a.tokenizer}
    print(f"[vqgate] steps train {tr.sum()} test {te.sum()} code {n_cells0}x{d} ({time.time()-t0:.0f}s)", flush=True)
    for arm in a.arms:
        if arm == "ours":
            agent, mu, sd = train_agent("ours_noeff", feats, dev, a, None)
            r = dict(chain=metrics(predict(agent, feats, te, dev, mu, sd)), oracle=metrics(predict(agent, feats, te, dev, mu, sd, z_override=Z)))
            with torch.no_grad():
                zh = torch.cat([agent.pi(FQ[b].to(dev).float(), torch.as_tensor(task[b], device=dev)).cpu() for b in np.array_split(np.flatnonzero(te), 20)]).numpy()
            r["latent_r2"] = float(1 - ((zh - Z[te]) ** 2).sum() / ((Z[te] - Z[tr].mean(0)) ** 2).sum())
        else:
            n_cells = n_cells0
            # vq<K>: one K-way code per cell (8 tokens); pq<S>x<K>: each cell split into S sub-vectors of d/S dims, K-way each (8*S tokens)
            n_sub, Kc = (int(arm[2:].split("x")[0]), int(arm.split("x")[1])) if arm.startswith("pq") else (1, int(arm[2:]))
            n_cells_eff, d_eff = n_cells * n_sub, d // n_sub; Zc = Z.reshape(len(Z), n_cells_eff, d_eff); n_cells = n_cells_eff
            C = np.stack([L.kmeans(Zc[tr][:, c], Kc, iters=30, seed=c) for c in range(n_cells)])      # (n_tokens, Kc, d_eff) on train steps
            I = np.stack([L.assign(Zc[:, c], C[c]) for c in range(n_cells)], 1)                                            # (n, n_cells)
            Zq = np.stack([C[c][I[:, c]] for c in range(n_cells)], 1).reshape(len(Z), -1).astype(np.float32)
            qerr = float(((Zq - Z) ** 2).sum() / ((Z - Z[tr].mean(0)) ** 2).sum())
            maj = np.array([np.bincount(I[tr][:, c], minlength=Kc).argmax() for c in range(n_cells)])
            fv = dict(feats); fv["Zstar"], fv["I"] = Zq, I
            agent, mu, sd = train_vq(fv, C, dev, a)
            Pm, Im = predict_vq(agent, fv, te, dev, mu, sd, "argmax"); Pe, _ = predict_vq(agent, fv, te, dev, mu, sd, "expect")
            Po, _ = predict_vq(agent, fv, te, dev, mu, sd, z_override=Zq)
            r = dict(Kc=Kc, n_tokens=n_cells, quant_err_rel=qerr, oracle_dequant=metrics(Po), chain_mode=metrics(Pm), chain_expect=metrics(Pe),
                     code_acc=float((Im == I[te]).mean()), code_acc_majority=float((I[te] == maj).mean()),
                     code_acc_per_cell=[float((Im[:, c] == I[te][:, c]).mean()) for c in range(n_cells)])
        R[arm] = r; print(f"[vqgate] {arm}: {r}", flush=True)
        json.dump(R, open(os.path.join(RESULTS, f"{a.out}.json"), "w"), indent=1)
    print("%-8s %6s %6s %9s %11s %11s %11s %9s %9s" % ("arm", "Kc", "tokens", "q_err", "l1 oracle", "l1 mode", "l1 expect", "code_acc", "(maj)"))
    for arm, r in R.items():
        if not isinstance(r, dict): continue
        if arm == "ours": print("%-8s %6s %6s %9s %11.3f %11.3f %11s %9.3f %9s" % (arm, "-", "-", "-", r["oracle"]["l1"], r["chain"]["l1"], "-", r["latent_r2"], "R2"))
        else: print("%-8s %6d %6d %9.3f %11.3f %11.3f %11.3f %9.3f %9.3f" % (arm, r["Kc"], r["n_tokens"], r["quant_err_rel"], r["oracle_dequant"]["l1"], r["chain_mode"]["l1"], r["chain_expect"]["l1"], r["code_acc"], r["code_acc_majority"]))
    print(f"[vqgate-done] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
