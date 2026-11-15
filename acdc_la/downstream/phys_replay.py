"""Physical replay in LIBERO: execute a donor's real action chunk from a recipient's state and compare the resulting
REAL effect with what the latent transport predicts (reviewer request: Psi(Y_hat_{j<-i}) vs Psi(Y^real_{j<-i})).

    MUJOCO_GL=egl python -m acdc_la.downstream.phys_replay probe   # determinism + contact strata, sim only
    MUJOCO_GL=egl python -m acdc_la.downstream.phys_replay collect  # pairs -> results/phys_replay/<suite>.npz (frames)

Clip geometry is the tokenizer's: T=5 frames, stride 4 -> one chunk = 16 control steps (0.8 s at 20 Hz).
Donor (i, t): restore demo i's recorded MuJoCo state at t, execute actions[t:t+16].  Recipient (j, t'): restore demo
j's state at t', execute THE SAME donor actions.  Physical-action identity (same OSC deltas + gripper command), not
task-semantic identity.  Grasp chunks are aligned on the gripper-close command; the recipient is taken at its own
close moment plus a shift, so the arm-object relation varies naturally.  Contact strata at the end of the chunk:
  centered = both finger pads touch the donor's grasped object (robosuite _check_grasp)
  offset   = some gripper geom touches it, not a grasp
  outside  = no gripper contact with it
"""
import argparse, glob, json, os, sys
import numpy as np

H = 16                                  # control steps per clip  (T-1) * stride = 4 * 4
STRIDE = 4
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
HDF5 = "/lustre/fs1/home/ti874297/lf/latent_free/data/LIBERO_hdf5"
OUT = "/lustre/fs1/home/ti874297/lf/latent_free/acdc_la/results/phys_replay"


def tasks_of(suite):
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()[suite]()
    out = []
    for ti in range(s.n_tasks):
        t = s.get_task(ti); f = os.path.join(HDF5, suite, t.name + "_demo.hdf5")
        if os.path.exists(f): out.append((ti, t, f))
    return out


class Sim:
    def __init__(self, task, res=256):          # caches hold 256x256 frames; clip_to_frames resizes to 224 in training
        from experiments.robot.libero.libero_utils import get_libero_env
        self.env, self.lang = get_libero_env(task, "openvla", resolution=res)
        self.env.reset(); self.e = self.env.env
        self.grip = self.e.robots[0].gripper
        self.objs = {n: o for n, o in self.e.objects_dict.items()}

    def frame(self, obs):
        return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])      # upright, as the regenerated RLDS

    def restore(self, state):
        self.e.timestep, self.e.done, self.e.horizon = 0, False, 10 ** 9     # replays are not episodes: no horizon
        return self.env.set_init_state(state)

    def run(self, state, actions):
        """restore + execute; frames every STRIDE steps (T=5), plus the end state."""
        obs = self.restore(state); fr = [self.frame(obs)]
        for k, a in enumerate(actions):
            obs, _, _, _ = self.env.step(a.tolist())
            if (k + 1) % STRIDE == 0: fr.append(self.frame(obs))
        return np.stack(fr), self.e.sim.get_state().flatten().copy()

    def contact(self, name):
        o = self.objs[name]
        if self.e._check_grasp(self.grip, o): return "centered"
        if self.e.check_contact(self.grip, o): return "offset"
        return "outside"

    def grasped(self):
        for n, o in self.objs.items():
            if self.e._check_grasp(self.grip, o): return n
        return None

    def obj_pos(self):
        return {n: self.e.sim.data.body_xpos[self.e.obj_body_id[n]].copy() for n in self.objs}


def close_steps(act):
    g = act[:, 6]
    return [t for t in range(1, len(g)) if g[t - 1] < 0 <= g[t]]


def probe(a):
    import h5py
    rep = {}
    for suite in a.suites:
        for ti, task, f in tasks_of(suite)[:a.tasks]:
            sim = Sim(task); D = h5py.File(f, "r")["data"]; demos = sorted(D.keys(), key=lambda s: int(s.split("_")[1]))[:a.demos]
            det, strata, n_close = [], {}, 0
            for i, di in enumerate(demos):
                A, S = D[di]["actions"][()], D[di]["states"][()]
                for t in range(0, len(A) - H, max(1, (len(A) - H) // 4)):     # determinism: replay vs recording
                    _, s_end = sim.run(S[t], A[t:t + H])
                    p_rep = sim.obj_pos(); sim.restore(S[t + H]); p_rec = sim.obj_pos()
                    det.append(max(np.linalg.norm(p_rep[n] - p_rec[n]) for n in p_rep))
                cs = close_steps(A); n_close += len(cs)
                for tc in cs[:1]:
                    t0 = max(0, tc - a.lead); sim.run(S[t0], A[t0:t0 + H]); obj = sim.grasped()
                    if obj is None: strata.setdefault("donor_no_grasp", 0); strata["donor_no_grasp"] += 1; continue
                    for dj in demos:
                        if dj == di: continue
                        Aj, Sj = D[dj]["actions"][()], D[dj]["states"][()]; cj = close_steps(Aj)
                        if not cj: continue
                        for sh in a.shifts:
                            tj = cj[0] - a.lead + sh
                            if tj < 0 or tj >= len(Sj): continue
                            sim.run(Sj[tj], A[t0:t0 + H]); c = sim.contact(obj)
                            strata[f"{c}@{sh:+d}"] = strata.get(f"{c}@{sh:+d}", 0) + 1
            det = np.array(det)
            r = dict(lang=sim.lang, n_det=len(det), det_obj_err_cm_median=float(np.median(det) * 100), det_obj_err_cm_p95=float(np.percentile(det, 95) * 100),
                     det_obj_err_cm_max=float(det.max() * 100), close_events=n_close, strata=strata)
            rep[f"{suite}/{ti}"] = r; print(f"[probe] {suite}/{ti} {json.dumps(r)}", flush=True)
            sim.env.close()
            if a.save_png:
                from PIL import Image
                fr, _ = Sim(task).run(S[0], A[:H]); Image.fromarray(np.concatenate(list(fr), 1)).save(os.path.join(OUT, f"probe_{suite}_{ti}.png"))
    json.dump(rep, open(os.path.join(OUT, "probe.json"), "w"), indent=1)


# ---------------------------------------------------------------- experiment: pairs, model decodes, metrics
class LangT5:                                     # masked-mean pooled frozen t5-base = the tokenizer's `c`
    def __init__(self, dev):
        from transformers import T5EncoderModel, T5Tokenizer
        d = "/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA/t5-base"
        self.tok = T5Tokenizer.from_pretrained(d); self.enc = T5EncoderModel.from_pretrained(d).to(dev).eval(); self.dev = dev

    def __call__(self, text):
        import torch
        with torch.no_grad():
            e = self.tok([text], return_tensors="pt").to(self.dev); h = self.enc(**e).last_hidden_state[0]
            return h[e["attention_mask"][0].bool()].mean(0).float()


def obj_cell(sim, name, res=256):
    """4x4 grid cell (row-major, in the upright frame) that contains the object's centre."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    M = get_camera_transform_matrix(sim.e.sim, "agentview", res, res)
    r, c = project_points_from_world_to_camera(sim.obj_pos()[name][None], M, res, res)[0]
    r, c = res - 1 - r, res - 1 - c                                         # the frame is flipped both ways
    return int(np.clip(r * 4 // res, 0, 3) * 4 + np.clip(c * 4 // res, 0, 3))


def collect(a):
    """Sim pass: donor clip, recipient's real clip under the donor's actions, stratum, object cell -> frames npz."""
    import h5py, torch
    from ..common import features as Fz
    dino = Fz.load_dino("cuda")
    def g4(frames):                                   # list of (5,224,224,3) u8 -> (N,5,16,768) fp16
        out = []
        for b in range(0, len(frames), 32):
            from ..data.distractor import clip_to_frames
            x = clip_to_frames(torch.from_numpy(np.stack(frames[b:b + 32])), "cuda")         # the training transform
            B = x.shape[0]                            # fp32 (V100 has no bf16; training used bf16 autocast -- fp32 is the exact value)
            out.append(Fz.dino_g4(dino, x.reshape(B * 5, *x.shape[2:])).view(B, 5, 16, -1).half().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 5, 16, 768), np.float16)
    rng = np.random.default_rng(0)
    for suite in a.suites:
        out = os.path.join(OUT, f"pairs_{suite}.npz")
        if os.path.exists(out): print(f"[collect] {out} exists", flush=True); continue
        R = {k: [] for k in ("XD", "YR", "GX", "GY", "kind", "stratum", "shift", "cell", "task", "lang", "di", "dj", "det")}
        for ti, task, f in tasks_of(suite)[:a.tasks]:
            sim = Sim(task); D = h5py.File(f, "r")["data"]; demos = sorted(D.keys(), key=lambda s: int(s.split("_")[1]))[:a.demos]
            data = {d: (D[d]["actions"][()], D[d]["states"][()]) for d in demos}
            def add(kind, xd, yr, st, sh, cell, di, dj, det):
                for k, v in zip(("XD", "YR", "kind", "stratum", "shift", "cell", "task", "lang", "di", "dj", "det"),
                                (xd, yr, kind, st, sh, cell, ti, sim.lang, di, dj, det)): R[k].append(v)
            n0 = len(R["kind"])
            for di in demos:
                A, S = data[di]
                # ---- grasp chunks: the close command inside the chunk, donor's own replay ends in a grasp
                for tc in close_steps(A)[:1]:
                    obj = None
                    for lead in (2, 4, 6):
                        t0 = max(0, tc - lead); xd, _ = sim.run(S[t0], A[t0:t0 + H]); obj = sim.grasped()
                        if obj: break
                    if obj is None: continue
                    xd2, _ = sim.run(S[t0], A[t0:t0 + H]); det = float(np.abs(xd2.astype(int) - xd).max())   # same state+actions twice
                    for dj in demos:
                        if dj == di: continue
                        Aj, Sj = data[dj]; cj = close_steps(Aj)
                        if not cj: continue
                        for sh in a.shifts:
                            tj = cj[0] - lead + sh
                            if tj < 0 or tj + H > len(Sj): continue
                            yr, _ = sim.run(Sj[tj], A[t0:t0 + H])
                            add("grasp", xd, yr, sim.contact(obj), sh, obj_cell(sim, obj), di, dj, det)
                # ---- generic chunks (any phase): donor at a random t, recipient demo at the same progress fraction
                for _ in range(a.generic):
                    t0 = int(rng.integers(0, len(A) - H)); xd, _ = sim.run(S[t0], A[t0:t0 + H])
                    dj = demos[int(rng.integers(len(demos)))]
                    if dj == di: continue
                    Aj, Sj = data[dj]; tj = int(round(t0 / (len(A) - H) * (len(Aj) - H)))
                    yr, _ = sim.run(Sj[tj], A[t0:t0 + H])
                    add("generic", xd, yr, "generic", 0, -1, di, dj, -1.0)
            if not R.get("EX"): R["EX"] = [np.concatenate([x, y], 0) for x, y in zip(R["XD"][:12], R["YR"][:12])]
            R["GX"].append(g4(R["XD"])); R["GY"].append(g4(R["YR"])); R["XD"], R["YR"] = [], []   # frames -> g4 per task
            print(f"[collect] {suite}/{ti} pairs {len(R['kind']) - n0}  ({sim.lang})", flush=True); sim.env.close()
        for k in ("GX", "GY"): R[k] = np.concatenate(R[k])
        R["EX"] = np.stack(R["EX"]); del R["XD"], R["YR"]
        np.savez_compressed(out, **{k: np.array(v) for k, v in R.items()})


TRAIN_DEMOS = range(8, 28)        # demos 0-7 are the replay test pairs; never trained on
VAL_DEMOS = (26, 27)


def cache(a):
    """LIBERO fine-tuning data in the oxe_mix cache format (frames.npy 256x256 u8, index.npz, meta.json), source
    libero_sim_<suite>.  Every recorded state of demos 8-27 is RESTORED and rendered (no open-loop drift); actions are
    the recorded OSC deltas (action_kind delta)."""
    import h5py
    from ..paths import CACHE
    for suite in a.suites:
        d = os.path.join(CACHE, "oxe", f"libero_sim_{suite.replace('libero_', '')}"); os.makedirs(d, exist_ok=True)
        if os.path.exists(os.path.join(d, "meta.json")): print(f"[cache] {d} exists", flush=True); continue
        eps = []
        for ti, task, f in tasks_of(suite):
            D = h5py.File(f, "r")["data"]
            for di in TRAIN_DEMOS:
                if f"demo_{di}" in D: eps.append((ti, task, f, di, len(D[f"demo_{di}"]["actions"])))
        N = sum(e[4] for e in eps)
        fr = np.lib.format.open_memmap(os.path.join(d, "frames.npy.tmp"), mode="w+", dtype=np.uint8, shape=(N, 256, 256, 3))
        starts, lengths, is_val, acts, cur, row, langs = [], [], [], [], None, 0, []
        for ti, task, f, di, L in eps:
            if cur is None or cur[0] != ti:
                if cur is not None: cur[1].env.close()
                cur = (ti, Sim(task))
            sim = cur[1]; e = h5py.File(f, "r")["data"][f"demo_{di}"]; S, A = e["states"][()], e["actions"][()]
            for t in range(L): fr[row + t] = sim.frame(sim.restore(S[t]))
            starts.append(row); lengths.append(L); is_val.append(di in VAL_DEMOS); acts.append(A.astype(np.float32)); langs.append(sim.lang)
            row += L
            print(f"[cache] {suite} task {ti} demo {di} L={L} rows={row}/{N}", flush=True)
        cur[1].env.close(); fr.flush(); del fr
        os.replace(os.path.join(d, "frames.npy.tmp"), os.path.join(d, "frames.npy"))
        np.savez(os.path.join(d, "index.npz"), starts=np.array(starts), lengths=np.array(lengths), is_val=np.array(is_val), actions=np.concatenate(acts))
        json.dump(dict(name=os.path.basename(d), action_kind="delta", episodes_available=len(starts), frames=int(N), img_size=256,
                       demos=list(TRAIN_DEMOS), val_demos=list(VAL_DEMOS), instructions=sorted(set(langs)),
                       note="rendered by restoring recorded MuJoCo states (phys_replay cache); demos 0-7 held out for replay"),
                  open(os.path.join(d, "meta.json"), "w"), indent=1)


PREDICTORS = ["copy_donor", "zero"]


def evaluate(a):
    """Model pass on the collected pairs: per-pair Psi cosine, increment errors, object-cell errors per predictor."""
    import torch
    from ..common import features as Fz
    from ..common.model import load_ckpt
    from ..common.pipeline import Featurizer
    dev = "cuda"; dino = None
    models = {}
    for tag in a.models:
        m, dstats, ck = load_ckpt(os.path.join("/lustre/fs1/home/ti874297/lf/latent_free/acdc_la/ckpt", tag + ".pt"), dev)
        ar = ck.get("args", {}); models[tag] = (m.eval(), Featurizer(dino, 4, ar.get("pool", "acdc"), ar.get("modes", 2), dstats, dev))
    dst = {tag: float(f.dstats) for tag, (_, f) in models.items()}
    assert len(set(round(v, 6) for v in dst.values())) == 1, dst              # one dF scale -> one Y_real for all models
    feat = next(iter(models.values()))[1]
    for suite in a.suites:
        P = np.load(os.path.join(OUT, f"pairs_{suite}.npz"), allow_pickle=True); n = len(P["kind"])
        res = {}
        for b in range(0, n, a.batch):
            sl = slice(b, min(n, b + a.batch))
            gx, gy = (torch.from_numpy(P[k][sl]).to(dev).float() for k in ("GX", "GY"))
            dx, dy = Fz.dF_from_g4(gx, feat.dstats), Fz.dF_from_g4(gy, feat.dstats)     # (B,K,16,768)
            c = torch.zeros(len(dx), 768, device=dev)      # training had no lang_t5 cache in any source: c was always 0
            preds = {"copy_donor": dx, "zero": torch.zeros_like(dy)}
            with torch.no_grad():
                for tag, (m, fz) in models.items():
                    if hasattr(m, "encode_batch"): zmu = m.encode_batch(dict(f0=gx[:, 0], dF=dx), "clean")[0]   # donor evidence only
                    else: zmu = m.encode(fz.cells_of(dx))[0]
                    preds[tag] = m.decode(gy[:, 0], c, zmu, torch.zeros(len(dx), m.d_u, device=dev))  # u = 0: no recipient leak
                    zo = m.encode_batch(dict(f0=gy[:, 0], dF=dy), "clean")[0] if hasattr(m, "encode_batch") else m.encode(fz.cells_of(dy))[0]
                    preds[tag + "@ownz"] = m.decode(gy[:, 0], c, zo, torch.zeros(len(dx), m.d_u, device=dev))   # oracle: recipient's own code
            cells = torch.as_tensor(P["cell"][sl], device=dev)
            res.setdefault("_pair", {"dist": [], "real_vs_donor_cos": []})
            res["_pair"]["dist"].append((Fz.quadrants(gy[:, 0]) - Fz.quadrants(gx[:, 0])).flatten(1).norm(dim=1))
            res["_pair"]["real_vs_donor_cos"].append(torch.nn.functional.cosine_similarity(feat.cells_of(dy).flatten(1), feat.cells_of(dx).flatten(1), dim=1))
            psi = lambda d: feat.cells_of(d).flatten(1)
            for k, yh in preds.items():
                r = res.setdefault(k, {q: [] for q in ("cos", "err", "nrm", "cos_dc", "obj_err", "obj_nrm", "obj_cos", "res_cos")})
                cos = torch.nn.functional.cosine_similarity
                # the recipient-specific part: does (prediction - donor effect) point where (real - donor effect) points?
                r["res_cos"].append(cos(psi(yh - dx), psi(dy - dx), dim=1) if k != "copy_donor" else torch.zeros(len(dx), device=dev))
                r["cos"].append(cos(psi(yh), psi(dy), dim=1)); r["cos_dc"].append(cos(yh.sum(1).flatten(1), dy.sum(1).flatten(1), dim=1))
                r["err"].append((yh - dy).pow(2).flatten(1).sum(1)); r["nrm"].append(dy.pow(2).flatten(1).sum(1))
                idx = cells.clamp_min(0)[:, None, None, None].expand(-1, dy.shape[1], 1, dy.shape[-1])
                oy, oh = dy.gather(2, idx)[:, :, 0], yh.gather(2, idx)[:, :, 0]               # (B,K,768) object cell
                r["obj_err"].append((oh - oy).pow(2).flatten(1).sum(1)); r["obj_nrm"].append(oy.pow(2).flatten(1).sum(1))
                r["obj_cos"].append(cos(oh.flatten(1), oy.flatten(1), dim=1))
        out = {f"{k}/{q}": torch.cat(v).cpu().numpy() for k, r in res.items() for q, v in r.items()}
        for q in ("kind", "stratum", "shift", "cell", "task", "di", "dj", "det"): out[q] = P[q]
        np.savez_compressed(os.path.join(OUT, f"eval_{suite}.npz"), **out); print(f"[eval] {suite} {n} pairs", flush=True)


def summarize(a):
    rows = {}
    E = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(OUT, "eval_*.npz")))]
    cat = lambda k: np.concatenate([e[k] for e in E])
    kind, st = cat("kind"), cat("stratum")
    preds = sorted({k.split("/")[0] for k in E[0].files if "/" in k and not k.startswith("_")})
    groups = {"all": np.ones(len(kind), bool), "generic": kind == "generic", "grasp": kind == "grasp"}
    for s in ("centered", "offset", "outside"): groups[s] = st == s
    dist = cat("_pair/dist")
    for g in ("generic", "grasp"):
        m = groups[g]; med = np.median(dist[m]) if m.any() else 0
        groups[f"{g}_near"], groups[f"{g}_far"] = m & (dist <= med), m & (dist > med)
    for g, msk in groups.items():
        rows[g] = {"n": int(msk.sum()), "real_vs_donor_cos": float(cat("_pair/real_vs_donor_cos")[msk].mean()) if msk.any() else float("nan")}
        for p in preds:
            v = lambda q: cat(f"{p}/{q}")[msk]
            rows[g][p] = dict(res_cos=float(v("res_cos").mean()), cos=float(v("cos").mean()), cos_dc=float(v("cos_dc").mean()), nmse=float(v("err").sum() / v("nrm").sum()),
                              obj_nmse=float(v("obj_err").sum() / max(v("obj_nrm").sum(), 1e-9)) if not g.startswith("generic") else None,
                              obj_cos=float(v("obj_cos").mean()) if not g.startswith("generic") else None)
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    for g, r in rows.items():
        print(f"== {g}  n={r['n']}  real-vs-donor cos {r['real_vs_donor_cos']:.3f}")
        for p in preds:
            x = r[p]; print(f"  {p:26s} res_cos {x['res_cos']:+.3f}  cos {x['cos']:.3f}  cos_dc {x['cos_dc']:.3f}  NMSE {x['nmse']:.3f}" +
                             (f"  obj NMSE {x['obj_nmse']:.3f}  obj cos {x['obj_cos']:.3f}" if x["obj_nmse"] is not None else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "collect", "eval", "summary", "cache"])
    ap.add_argument("--generic", type=int, default=6, help="generic (non-grasp) donor chunks per demo")
    ap.add_argument("--models", nargs="+", default=[f"eatabm_{m}_s{s}" for m in ("full", "recon") for s in (0, 1, 2)])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--tasks", type=int, default=10); ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--lead", type=int, default=6, help="chunk starts this many steps before the close command")
    ap.add_argument("--shifts", type=int, nargs="+", default=[-8, -4, 0, 4, 8])
    ap.add_argument("--save-png", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"cache": cache, "probe": probe, "collect": collect, "eval": evaluate, "summary": summarize}[a.mode](a)


if __name__ == "__main__":
    main()
