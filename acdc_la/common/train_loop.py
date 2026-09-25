"""The training loop shared by the three stages.  A stage supplies `step_fn(model, feat, b, a, ctx, train)`
returning (loss, dict of scalar terms); everything else -- seeds, loaders, frozen DINO, feature scale,
checkpoint init / encoder freezing, warm-up ramp of the auxiliary weights, cosine schedule, logging,
checkpoint, validation dump -- is here, so an ablation differs from the method by its step function
and its flags only.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch

from ..paths import RESULTS, CKPT
from . import features as Fz
from .model import LAM, load_ckpt, save_ckpt, legacy_path, jl_projection
from .pipeline import Featurizer
from .dump import dump_val
from . import loaders
from ..data.distractor import Distractor

QUIET = ("xFormers", "warnings.warn", "Using cache", "UserWarning", "enable_nested")


def common_args(ap):
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sources", nargs="*", default=None, help="default: every complete cache")
    ap.add_argument("--T", type=int, default=5); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--pool", default="acdc", choices=["acdc", "mean", "endpoint"], help="Stage-I operator (mean/endpoint = pooled ablations)")
    ap.add_argument("--modes", type=int, default=2, help="temporal modes per quadrant for --pool acdc: 1 = DC only, 2 = DC+AC1, 3 = +AC2")
    ap.add_argument("--d", type=int, default=32); ap.add_argument("--d-u", type=int, default=8); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--beta-z", type=float, default=2e-4); ap.add_argument("--beta-u", type=float, default=1e-1)
    ap.add_argument("--warmup", type=int, default=1000, help="steps before auxiliary terms ramp in (ramp reaches 1 at 2*warmup)")
    ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--limit-eps", type=int, default=None)
    ap.add_argument("--val-per-source", type=int, default=400); ap.add_argument("--val-batches", type=int, default=10 ** 9)
    ap.add_argument("--init-from", type=str, default=None, help="checkpoint path, or a tag: ckpt/<tag>.pt here, else legacy claude_survey/ckpt/swaplam_<tag>.pt")
    ap.add_argument("--freeze-encoder", action="store_true", help="train the decoder only (E, U, cell_emb frozen)")
    ap.add_argument("--eval-only", action="store_true", help="load ckpt/<tag>.pt and write the validation dump only")
    ap.add_argument("--save-every", type=int, default=5000, help="periodic checkpoint (model + optimiser + step) for --resume")
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this many steps but keep the --steps cosine schedule, "
                    "so a run can match one that was cut short (tc_cont: --steps 128000 stopped at 65000); 0 = run to --steps")
    ap.add_argument("--resume", action="store_true", help="continue ckpt/<tag>.pt from its saved step (same seed, same batch list)")
    ap.add_argument("--eval-every", type=int, default=-1, help="steps between cheap proxy evaluations (Tier 0/1 of PROTOCOL.md) on a 100-clip-per-source val subset; -1 = steps//10 for runs >= 200k steps, else off")
    ap.add_argument("--curve-per-source", type=int, default=200)   # 100 in the leg-1 runs of 2026-09-12 (per-source ridge on ~33 test clips is noisy)
    return ap


def resolve_ckpt(spec):
    if os.path.exists(spec): return spec
    here = os.path.join(CKPT, f"{spec}.pt")
    return here if os.path.exists(here) else legacy_path(spec)


def run(a, step_fn, ramp_keys=(), pairs_for_train=False, stage="", build=None, on_step=None):
    """build(a, dev) -> (model, dstats): a stage that wraps an initial checkpoint in its own module (Stage IV)."""
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck_path = os.path.join(CKPT, f"{a.tag}.pt"); start = 0; ck_res = None
    if a.resume and os.path.exists(ck_path):
        ck_res = torch.load(ck_path, map_location=dev); start = int(ck_res.get("step", 0))
        if start >= a.steps and "opt" not in ck_res: print(f"[{a.tag}] already finished ({start} steps); nothing to resume", flush=True); return
        print(f"[{a.tag}] resuming from step {start}", flush=True)
    srcs, ds, dv, dl, dl_val = loaders.build(a, pairs_for_train=pairs_for_train, start_step=start)
    loaders.describe(a, srcs, ds, dv, dev)
    dino = Fz.load_dino(dev); K = a.T - 1
    n_cells = 4 * a.modes if a.pool == "acdc" else 4
    if ck_res is not None:
        model, dstats, _ = load_ckpt(ck_path, dev)
    elif build is not None and not a.eval_only:
        model, dstats = build(a, dev); a.d, a.d_u = model.d, model.d_u
        if dstats is None: dstats = Fz.estimate_dstats(dino, dl, dev)            # a stage that builds from scratch
        print(f"[{a.tag}] built {type(model).__name__} from {a.init_from} (dF rms {dstats:.4f})", flush=True)
    elif a.eval_only or a.init_from:
        model, dstats, ck0 = load_ckpt(ck_path if a.eval_only else resolve_ckpt(a.init_from), dev)
        assert model.K == K and (model.n_cells == n_cells or hasattr(model, "ckpt_meta")), (model.K, model.n_cells, K, n_cells)
        # a knob that is accepted and silently ignored is the bug class of 2026-08 (STATUS R8) and of the "u64" runs of
        # 2026-09-10 (d_u 64 requested, 8 loaded): the checkpoint fixes the architecture, so the flags must agree
        if hasattr(model, "ckpt_meta"): a.d, a.d_u = model.d, model.d_u
        assert (model.d, model.d_u) == (a.d, a.d_u), f"--init-from fixes d={model.d}, d_u={model.d_u}; flags say d={a.d}, d_u={a.d_u}. Train from scratch to change them."
        print(f"[{a.tag}] loaded {'own' if a.eval_only else a.init_from} checkpoint (dF rms {dstats:.4f})", flush=True)
    else:
        model = LAM(K, n_cells, a.d, a.d_u, layers=a.layers).to(dev)
        dstats = Fz.estimate_dstats(dino, dl, dev)                  # a constant of the frozen features
        print(f"[{a.tag}] dF rms {dstats:.4f}", flush=True)
    feat = Featurizer(dino, K, a.pool, a.modes, dstats, dev)
    distractor = Distractor(dev)
    if a.eval_every == -1: a.eval_every = a.steps // 10 if a.steps >= 200000 else 0
    dl_curve = None
    if a.eval_every:
        idx_c = dv.stratified_pair_indices(a.curve_per_source // 2, seed=321)
        dl_curve = torch.utils.data.DataLoader(torch.utils.data.Subset(dv, idx_c), batch_size=a.batch, shuffle=False, num_workers=min(4, a.workers), drop_last=True)
    ctx = dict(proj=jl_projection(K * 16 * 768, 512, 0, dev), distractor=distractor, dev=dev)
    dump_path = os.path.join(RESULTS, f"{a.tag}_val.npz")
    if a.eval_only:
        R = dump_val(model, feat, dl_val, distractor, dump_path, a.val_batches)
        R.update(tag=a.tag, stage=stage, eval_only=True, args=vars(a)); json.dump(R, open(os.path.join(RESULTS, f"{a.tag}.json"), "w"), indent=1)
        print(f"[{a.tag}-done] {R}", flush=True); return
    if a.freeze_encoder: model.freeze_encoder()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[{a.tag}] params {sum(p.numel() for p in model.parameters())/1e6:.1f}M trainable {sum(p.numel() for p in params)/1e6:.1f}M "
          f"pool={a.pool} modes={a.modes} n_cells={n_cells}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    if ck_res is not None and "opt" in ck_res:
        opt.load_state_dict(ck_res["opt"]); sched.load_state_dict(ck_res["sched"])
    base = {k: getattr(a, k) for k in ramp_keys}
    t0, log = time.time(), {}
    for s, batch in enumerate(dl, start=start):
        if s >= a.steps: break
        ramp = min(1.0, max(0.0, (s - a.warmup) / max(1, a.warmup)))
        for k, v in base.items(): setattr(a, k, ramp * v)
        b = feat(batch, distractor if getattr(a, "needs_patch", False) else None)
        loss, out = step_fn(model, feat, b, a, ctx, train=True)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        for k, v in out.items(): log.setdefault(k, []).append(float(v))
        if on_step is not None: on_step(model, s)
        if (s + 1) % 200 == 0 or s + 1 == a.steps:
            msg = " ".join(f"{k}={np.mean(v[-200:]):.4f}" for k, v in log.items())
            print(f"[{a.tag}] step {s+1} ramp={ramp:.2f} {msg} ({(time.time()-t0)/(s+1-start):.2f}s/step)", flush=True)
        if a.save_every and (s + 1) % a.save_every == 0 and s + 1 < a.steps:
            save_ckpt(ck_path, model, vars(a), dstats, extra=dict(sources=srcs, stage=stage, step=s + 1, opt=opt.state_dict(), sched=sched.state_dict()))
        if dl_curve is not None and (s + 1) % a.eval_every == 0 and s + 1 < a.steps:
            try:                                        # a proxy-curve failure must never kill a 40-hour pretraining run
                from .curve import curve_eval
                curve_eval(model, feat, dl_curve, distractor, a.tag, s + 1, a.steps, dict(train_rec=float(np.mean(log["rec"][-200:]))))
            except Exception as e:
                import traceback; print(f"[{a.tag}] CURVE FAILED at step {s+1}: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}", flush=True)
            model.train()
        if s + 1 >= a.steps: break
        if getattr(a, "stop_at", 0) and s + 1 >= a.stop_at: break
    for k, v in base.items(): setattr(a, k, v)
    done = min(a.steps, a.stop_at) if getattr(a, "stop_at", 0) else a.steps
    save_ckpt(ck_path, model, vars(a), dstats, extra=dict(sources=srcs, stage=stage, step=done,
                                                          opt=opt.state_dict(), sched=sched.state_dict()))
    R = dump_val(model, feat, dl_val, distractor, dump_path, a.val_batches)
    R.update(tag=a.tag, stage=stage, seed=a.seed, steps=done, seconds=int(time.time() - t0), train_clips=len(ds), sources=len(srcs),
             final={k: float(np.mean(v[-200:])) for k, v in log.items()}, args=vars(a))
    json.dump(R, open(os.path.join(RESULTS, f"{a.tag}.json"), "w"), indent=1)
    print(f"[{a.tag}-done] rec={R['rec']:.4f} kl_z={R['kl_z']:.2f} val_clips={R['n_val_clips']} -> {dump_path}", flush=True)
