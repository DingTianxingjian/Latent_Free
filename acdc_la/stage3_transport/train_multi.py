"""Rebuttal ablation A (2026-11-07): train SEVERAL tokenizer arms in one process on ONE data stream.

Every arm of Table 4 continues the same 65k checkpoint (tc_cont) for 15k steps with the same seed, so a single-arm run
reads exactly the same batch list (loaders.build is deterministic in the seed) -- the arms differ only in their loss
weights.  The single-arm jobs spent ~2.8 s/step mostly loading clips and running the frozen DINO featurizer, both of
which are identical across arms.  Here the batch, its DINO features and the one augmentation draw used by L_inv are
computed ONCE per step and every arm takes its own optimiser step on them.

Per arm: its own model, EMA teacher / donor buffer / whitening state (train_v2 ctx), AdamW + cosine schedule, warm-up
ramp, log, checkpoint ckpt/<prefix>_<arm>_s<seed>.pt, curve evaluations and final validation dump -- i.e. everything
run() does, kept per arm.  Not bit-identical to a single-arm run (the arms share the global RNG used by the
derangement / donor draws), statistically the same procedure.  A side effect that makes the comparison tighter: all
arms see the same clips AND the same nuisance draw at every step.

Arms (Table 4 rows):  recon = L_self (stage2_tc step)          noinv = EAT w/o L_inv       full = EAT (TERRA)
  a1 = L_self + L_inv       a2 = full - L_effect      a3 = full - L_rel      a4 = full - L_dist (MMD)
  a5 = L_effect -> head g(z) ~ Psi(dF), cosine       a6 = L_effect -> latent cycle, cosine to sg[z] (L_recover, l1 = 0)

    python -m acdc_la.stage3_transport.train_multi --tag eatabm_s0 --arms recon,noinv,full,a1,a2,a3,a4,a5,a6 \
        --init-from tc_cont --steps 15000 --warmup 4000 --batch 64 --lr 3e-4 --seed 0 --save-every 1000 --resume
"""
import argparse, copy, json, math, os, time
from types import SimpleNamespace
import numpy as np
import torch

from ..paths import RESULTS, CKPT
from ..common import features as Fz
from ..common.model import load_ckpt, save_ckpt, jl_projection
from ..common.pipeline import Featurizer
from ..common.dump import dump_val
from ..common import loaders
from ..common.train_loop import common_args, resolve_ckpt
from ..data.distractor import Distractor
from . import train_v2 as V2
from ..stage2_tc import train as TC

#            lam_effect lam_real lam_rel lam_inv lam_recover recover_l1 lam_head
ARMS = dict(
    recon=None,
    noinv=(1, 1, 0.5, 0, 0, 1.0, 0),
    full=(1, 1, 0.5, 1, 0, 1.0, 0),
    a1=(0, 0, 0, 1, 0, 1.0, 0),
    a2=(0, 1, 0.5, 1, 0, 1.0, 0),
    a3=(1, 1, 0, 1, 0, 1.0, 0),
    a4=(1, 0, 0.5, 1, 0, 1.0, 0),
    a5=(0, 1, 0.5, 1, 0, 1.0, 1),
    a6=(0, 1, 0.5, 1, 1, 0.0, 0),
)
RAMP = ("lam_effect", "lam_recover", "lam_real", "lam_rel", "lam_head")


def arm_args(a, name, prefix):
    """The namespace a single-arm train_v2 run of this arm would have had (same defaults, same flags)."""
    x = SimpleNamespace(**vars(a)); x.tag = f"{prefix}_{name}_s{a.seed}"; x.arm = name
    x.lam_var, x.probe_noise, x.recover_detach = 0.0, 0.1, False
    if ARMS[name] is None:                                           # recon: stage2_tc's objective, no aux terms
        x.lam_prior, x.lam_commit, x.lam_effect, x.lam_real, x.lam_rel, x.lam_inv, x.lam_recover, x.recover_l1, x.lam_head = 0, .25, 0, 0, 0, 0, 0, 1.0, 0
    else:
        x.lam_effect, x.lam_real, x.lam_rel, x.lam_inv, x.lam_recover, x.recover_l1, x.lam_head = ARMS[name]
    return x


def main():
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--arms", required=True, help="comma list from: " + ",".join(ARMS))
    ap.add_argument("--tau", type=float, default=0.99); ap.add_argument("--buffer", type=int, default=16)
    ap.add_argument("--rtc-groups", type=int, default=16); ap.add_argument("--psi-dim", type=int, default=512)
    a = ap.parse_args()
    names = a.arms.split(","); assert all(n in ARMS for n in names), names
    assert a.init_from, "every arm continues --init-from"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; assert dev == "cuda" or os.environ.get("ACDC_ALLOW_CPU"), "no GPU visible"
    prefix = a.tag                                                   # e.g. eatabm  ->  eatabm_<arm>_s<seed>
    A = {n: arm_args(a, n, prefix) for n in names}
    paths = {n: os.path.join(CKPT, f"{A[n].tag}.pt") for n in names}
    res = {n: torch.load(paths[n], map_location=dev) for n in names if a.resume and os.path.exists(paths[n])}
    steps_saved = {n: int(r.get("step", 0)) for n, r in res.items()}
    start = 0
    if res:
        assert len(res) == len(names) and len(set(steps_saved.values())) == 1, f"arms saved at different steps: {steps_saved}"
        start = next(iter(steps_saved.values())); print(f"[{prefix}] resuming all arms from step {start}", flush=True)
    srcs, ds, dv, dl, dl_val = loaders.build(a, start_step=start)
    loaders.describe(a, srcs, ds, dv, dev)
    dino = Fz.load_dino(dev); K = a.T - 1
    distractor = Distractor(dev)
    idx_c = dv.stratified_pair_indices(a.curve_per_source // 2, seed=321)
    dl_curve = torch.utils.data.DataLoader(torch.utils.data.Subset(dv, idx_c), batch_size=a.batch, shuffle=False,
                                           num_workers=min(4, a.workers), drop_last=True) if a.eval_every > 0 else None
    proj = jl_projection(K * 16 * 768, 512, 0, dev)                   # identical to run(): seed-0 projection

    R = {}                                                           # per-arm state
    for n in names:
        x = A[n]
        model, dstats, _ = load_ckpt(paths[n] if n in res else resolve_ckpt(a.init_from), dev)
        ctx = dict(proj=proj, distractor=distractor, dev=dev)
        params = [p for p in model.parameters() if p.requires_grad]
        if x.lam_head > 0:                                            # A5's predictor; trained with the arm, saved with it
            ctx["head"] = torch.nn.Sequential(torch.nn.Linear(model.n_tokens * model.d, 512), torch.nn.GELU(),
                                              torch.nn.Linear(512, a.psi_dim)).to(dev)
            if n in res and "head" in res[n]: ctx["head"].load_state_dict(res[n]["head"])
            params += list(ctx["head"].parameters())
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
        if n in res and "opt" in res[n]: opt.load_state_dict(res[n]["opt"]); sched.load_state_dict(res[n]["sched"])
        if n in res and "v2ctx" in res[n]:                            # EMA teacher, frozen T0, buffer, whitening: exact resume
            c = res[n]["v2ctx"]; ctx["teacher"] = copy.deepcopy(model).eval().requires_grad_(False); ctx["teacher"].load_state_dict(c["teacher"])
            ctx["T0"] = copy.deepcopy(model).eval().requires_grad_(False); ctx["T0"].load_state_dict(c["T0"])
            for k in ("zmu", "zvar", "psi_proj", "psi_mu", "psi_var", "b1"): ctx[k] = c[k].to(dev)
            ctx["buf"] = [(z.to(dev), s_, g_) for z, s_, g_ in c["buf"]]
        R[n] = dict(model=model, ctx=ctx, params=params, opt=opt, sched=sched, dstats=dstats, log={},
                    base={k: getattr(x, k) for k in RAMP}, step_fn=TC.step if ARMS[n] is None else V2.step)
        print(f"[{x.tag}] {'resumed' if n in res else 'loaded ' + a.init_from} | " +
              ", ".join(f"{k}={getattr(x, k)}" for k in ("lam_effect", "lam_real", "lam_rel", "lam_inv", "lam_recover", "recover_l1", "lam_head")), flush=True)
    dstats = R[names[0]]["dstats"]; assert all(abs(R[n]["dstats"] - dstats) < 1e-6 for n in names)
    feat = Featurizer(dino, K, a.pool, a.modes, dstats, dev)
    need_aug = any(A[n].lam_inv > 0 for n in names)

    def save(n, s):
        r = R[n]; c = r["ctx"]; extra = dict(sources=srcs, stage="EATAB", step=s, arm=n, opt=r["opt"].state_dict(), sched=r["sched"].state_dict())
        if "teacher" in c:
            extra["v2ctx"] = dict(teacher=c["teacher"].state_dict(), T0=c["T0"].state_dict(), buf=[(z.cpu(), s_, g_) for z, s_, g_ in c["buf"]],
                                  **{k: c[k] for k in ("zmu", "zvar", "psi_proj", "psi_mu", "psi_var", "b1")})
        if "head" in c: extra["head"] = c["head"].state_dict()
        for k, v in r["base"].items(): setattr(A[n], k, v)
        save_ckpt(paths[n], r["model"], vars(A[n]), dstats, extra=extra)

    t0 = time.time(); t_data = t_feat = 0.0; t_arm = {n: 0.0 for n in names}; tl = time.time()
    for s, batch in enumerate(dl, start=start):
        if s >= a.steps: break
        t_data += time.time() - tl; tf = time.time()
        ramp = min(1.0, max(0.0, (s - a.warmup) / max(1, a.warmup)))
        with torch.no_grad():
            b = feat(batch, None)
            if need_aug:
                g4a = feat.g4_of(V2._jitter(b["frames"])); b["_aug"] = (g4a[:, 0], Fz.dF_from_g4(g4a, feat.dstats))
        (torch.cuda.synchronize() if dev == "cuda" else None); t_feat += time.time() - tf
        for n in names:
            ta = time.time(); r = R[n]; x = A[n]
            for k, v in r["base"].items(): setattr(x, k, ramp * v)
            r["model"].train()
            loss, out = r["step_fn"](r["model"], feat, b, x, r["ctx"], train=True)
            r["opt"].zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(r["params"], 1.0); r["opt"].step(); r["sched"].step()
            for k, v in out.items(): r["log"].setdefault(k, []).append(float(v))
            (torch.cuda.synchronize() if dev == "cuda" else None); t_arm[n] += time.time() - ta
        if (s + 1) % 200 == 0 or s + 1 == a.steps:
            m = s + 1 - start
            print(f"[{prefix}] step {s+1} ramp={ramp:.2f} {(time.time()-t0)/m:.2f}s/step = data {t_data/m:.2f} + feat {t_feat/m:.2f} + "
                  + " + ".join(f"{n} {t_arm[n]/m:.2f}" for n in names), flush=True)
            for n in names:
                print(f"[{A[n].tag}] step {s+1} ramp={ramp:.2f} " + " ".join(f"{k}={np.mean(v[-200:]):.4f}" for k, v in R[n]["log"].items()), flush=True)
        if a.save_every and (s + 1) % a.save_every == 0 and s + 1 < a.steps:
            for n in names: save(n, s + 1)
        if dl_curve is not None and (s + 1) % a.eval_every == 0 and s + 1 < a.steps:
            from ..common.curve import curve_eval
            for n in names:
                try: curve_eval(R[n]["model"], feat, dl_curve, distractor, A[n].tag, s + 1, a.steps, dict(train_rec=float(np.mean(R[n]["log"]["rec"][-200:]))))
                except Exception as e: print(f"[{A[n].tag}] CURVE FAILED at step {s+1}: {type(e).__name__}: {e}", flush=True)
        tl = time.time()
    for n in names:
        save(n, a.steps); r = R[n]
        out = dump_val(r["model"], feat, dl_val, distractor, os.path.join(RESULTS, f"{A[n].tag}_val.npz"), a.val_batches)
        out.update(tag=A[n].tag, stage="EATAB", arm=n, seed=a.seed, steps=a.steps, seconds=int(time.time() - t0), train_clips=len(ds),
                   sources=len(srcs), final={k: float(np.mean(v[-200:])) for k, v in r["log"].items()}, args=vars(A[n]))
        json.dump(out, open(os.path.join(RESULTS, f"{A[n].tag}.json"), "w"), indent=1)
        print(f"[{A[n].tag}-done] rec={out['rec']:.4f} val_clips={out['n_val_clips']}", flush=True)


if __name__ == "__main__":
    main()
