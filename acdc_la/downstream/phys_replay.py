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
ROOT = os.environ.get("LF_ROOT", "/lustre/fs1/home/ti874297/lf/latent_free")   # Newton; override on other hosts
HDF5 = os.environ.get("LF_LIBERO_HDF5", os.path.join(ROOT, "data/LIBERO_hdf5"))
OUT = os.environ.get("LF_PHYS_OUT", os.path.join(ROOT, "acdc_la/results/phys_replay"))
CKPT = os.environ.get("LF_CKPT", os.path.join(ROOT, "acdc_la/ckpt"))
T5 = os.environ.get("LF_T5", os.path.join(ROOT, "third_party/UniVLA/t5-base"))


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

    def restore(self, state, past=None):
        """Restore a recorded state EXACTLY.  Two pieces of state live outside the MuJoCo state vector and must be
        rebuilt, or the same state + actions do not reproduce (2026-11-15 test: object 1.4 cm off, gripper fingers
        differ): (1) solver warm-start -> mj_resetData first; (3, below) the OSC goal; (2) the Panda gripper's integrated command
        `current_action` (+-0.01 per sim substep toward sign(cmd), clipped) -> replayed from the demo's own command
        history `past` (actions[:t]); zeros at episode start, as robosuite's reset."""
        self.e.timestep, self.e.done, self.e.horizon = 0, False, 10 ** 9     # replays are not episodes: no horizon
        self.e.sim.reset()
        ca = np.zeros(self.grip.dof)
        if past is not None and len(past):
            n_sub = int(round(self.e.control_timestep / self.e.model_timestep)); step = np.array([-1.0, 1.0]) * self.grip.speed
            for g in np.sign(np.asarray(past)[:, -1]):
                ca = np.clip(ca + step * g * n_sub, -1.0, 1.0)      # n_sub same-sign increments: clip once == clip each
        self.grip.current_action = ca
        obs = self.env.set_init_state(state)
        # (3) the OSC controller keeps goal_ori across calls and only refreshes it on a non-zero rotation command, so a
        # stale goal from the PREVIOUS replay steered the first steps (first run != later runs, robot state 0.11 apart).
        for r in self.e.robots:
            r.controller.update(force=True); r.controller.reset_goal()
        return obs

    def run(self, state, actions, past=None):
        """restore + execute; frames every STRIDE steps (T=5), plus the end state."""
        obs = self.restore(state, past); fr = [self.frame(obs)]
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
            det, det2, strata, n_close = [], [], {}, 0
            for i, di in enumerate(demos):
                A, S = D[di]["actions"][()], D[di]["states"][()]
                for t in range(0, len(A) - H, max(1, (len(A) - H) // 4)):
                    # TWO different quantities, kept apart (2026-11-15, workstation): (A) replay vs replay is the
                    # experiment's own noise floor and must be 0 after the exact-restore fix; (B) replay vs the
                    # RECORDED state at t+H is a property of LIBERO's demos in this robosuite build (they were not
                    # produced by this code path: solver warm-start and the gripper's integrated command evolved
                    # along the whole original episode).  (B) does NOT enter the metric -- both X_D and Y_real are
                    # replayed -- so it must not be used to gate collection.  Measured: (A) 0.0000 cm on 56/56
                    # chunks, (B) median 0.12 / p95 1.30 / max 9.71 cm.
                    _, s_end = sim.run(S[t], A[t:t + H], A[:t]); p_rep = {k: v.copy() for k, v in sim.obj_pos().items()}
                    sim.run(S[t], A[t:t + H], A[:t]); p_rep2 = sim.obj_pos()
                    det2.append(max(np.linalg.norm(p_rep[n] - p_rep2[n]) for n in p_rep))
                    sim.restore(S[t + H], A[:t + H]); p_rec = sim.obj_pos()
                    det.append(max(np.linalg.norm(p_rep[n] - p_rec[n]) for n in p_rep))
                cs = close_steps(A); n_close += len(cs)
                for tc in cs[:1]:
                    t0 = max(0, tc - a.lead); sim.run(S[t0], A[t0:t0 + H], A[:t0]); obj = sim.grasped()
                    if obj is None: strata.setdefault("donor_no_grasp", 0); strata["donor_no_grasp"] += 1; continue
                    for dj in demos:
                        if dj == di: continue
                        Aj, Sj = D[dj]["actions"][()], D[dj]["states"][()]; cj = close_steps(Aj)
                        if not cj: continue
                        for sh in a.shifts:
                            tj = cj[0] - a.lead + sh
                            if tj < 0 or tj >= len(Sj): continue
                            sim.run(Sj[tj], A[t0:t0 + H], Aj[:tj]); c = sim.contact(obj)
                            strata[f"{c}@{sh:+d}"] = strata.get(f"{c}@{sh:+d}", 0) + 1
            det = np.array(det)
            det2 = np.array(det2)
            r = dict(lang=sim.lang, n_det=len(det),
                     replay_vs_replay_cm_max=float(det2.max() * 100), replay_vs_replay_zero_frac=float((det2 < 1e-11).mean()),
                     vs_recording_cm_median=float(np.median(det) * 100), vs_recording_cm_p95=float(np.percentile(det, 95) * 100),
                     vs_recording_cm_max=float(det.max() * 100), close_events=n_close, strata=strata)
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
        d = T5
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
        R = {k: [] for k in ("XD", "YR", "XD2", "XD2_key", "GX", "GY", "GX2", "kind", "stratum", "shift", "cell", "task", "lang", "di", "dj", "det", "lead")}
        for ti, task, f in tasks_of(suite)[:a.tasks]:
            sim = Sim(task); D = h5py.File(f, "r")["data"]; demos = sorted(D.keys(), key=lambda s: int(s.split("_")[1]))[:a.demos]
            data = {d: (D[d]["actions"][()], D[d]["states"][()]) for d in demos}
            def add(kind, xd, yr, st, sh, cell, di, dj, det, lead=99):
                for k, v in zip(("XD", "YR", "kind", "stratum", "shift", "cell", "task", "lang", "di", "dj", "det", "lead"),
                                (xd, yr, kind, st, sh, cell, ti, sim.lang, di, dj, det, lead)): R[k].append(v)
            n0 = len(R["kind"])
            for di in demos:
                A, S = data[di]
                # ---- grasp chunks: the close command inside the chunk, donor's own replay ends in a grasp
                for tc in close_steps(A)[:1]:
                    obj = None
                    # The chunk is anchored at t0 = tc - lead and must END in a grasp.  In the slow-grasp
                    # tasks the grasp only forms 4-12 steps AFTER the close command, so a sweep over positive
                    # leads alone drops those donors -- systematically, by whole task (libero_spatial/7:
                    # 0/8 donors with (2,4,6), 8/8 once 0/-4/-8 are allowed, first qualifying lead -8/-4/0).
                    # --leads keeps (2,4,6) as the default so a run already in flight is unchanged, and the
                    # extended list is a strict SUPERSET in that order: donors that qualified still qualify
                    # at the same lead, so an extended collection can be pooled with an old one.
                    for lead in a.leads:
                        t0 = max(0, tc - lead); xd, _ = sim.run(S[t0], A[t0:t0 + H], A[:t0]); obj = sim.grasped()
                        if obj: break
                    if obj is None: continue
                    xd2, _ = sim.run(S[t0], A[t0:t0 + H], A[:t0]); det = float(np.abs(xd2.astype(int) - xd).mean())   # same state+actions twice
                    R["XD2"].append(xd2); R["XD2_key"].append(f"{ti}_{di}")             # replay noise floor, one per donor
                    for dj in demos:
                        if dj == di: continue
                        Aj, Sj = data[dj]; cj = close_steps(Aj)
                        if not cj: continue
                        for sh in a.shifts:
                            tj = cj[0] - lead + sh
                            if tj < 0 or tj + H > len(Sj): continue
                            yr, _ = sim.run(Sj[tj], A[t0:t0 + H], Aj[:tj])
                            add("grasp", xd, yr, sim.contact(obj), sh, obj_cell(sim, obj), di, dj, det, lead)
                # ---- generic chunks (any phase): donor at a random t, recipient demo at the same progress fraction
                for _ in range(a.generic):
                    t0 = int(rng.integers(0, len(A) - H)); xd, _ = sim.run(S[t0], A[t0:t0 + H], A[:t0])
                    dj = demos[int(rng.integers(len(demos)))]
                    if dj == di: continue
                    Aj, Sj = data[dj]; tj = int(round(t0 / (len(A) - H) * (len(Aj) - H)))
                    yr, _ = sim.run(Sj[tj], A[t0:t0 + H], Aj[:tj])
                    add("generic", xd, yr, "generic", 0, -1, di, dj, -1.0)
            if not R.get("EX"): R["EX"] = [np.concatenate([x, y], 0) for x, y in zip(R["XD"][:12], R["YR"][:12])]
            R["GX"].append(g4(R["XD"])); R["GY"].append(g4(R["YR"])); R["XD"], R["YR"] = [], []   # frames -> g4 per task
            R["GX2"].append(g4(R["XD2"])); R["XD2"] = []
            print(f"[collect] {suite}/{ti} pairs {len(R['kind']) - n0}  ({sim.lang})", flush=True); sim.env.close()
        for k in ("GX", "GY", "GX2"): R[k] = np.concatenate(R[k])
        R["EX"] = np.stack(R["EX"]); del R["XD"], R["YR"], R["XD2"]
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


# Nulls (LEDGER 2026-11-15, transport_nulls.py --residual): the raw residual cosine cos(Psi(Y_hat - Y_d), Psi(Y_r - Y_d))
# is NOT 0 for predictors that ignore the recipient -- every a * Y_d with a < 1 (zero included) scores the same positive
# value, because cosine is scale-invariant and Y_r, Y_d are near-orthogonal.  So the reported quantities are:
#   proj_cos   cos(P Psi(Y_hat), P Psi(Y_real)),  P = I - u u^T,  u = Psi(Y_donor)/|.|   -> the whole a * Y_donor family = 0
#   @wrongz    the same decoder with a WRONG donor's code (another task's donor, fixed per donor group): what the
#              recipient's first frame alone buys.  The claim is matched - wrong, not distance from 0.
#   DiD        same donor, two recipients B, C: cos(Y_hat_B - Y_hat_C, Y_real_B - Y_real_C); the donor term cancels
#              exactly (copy / scaled copy / zero are undefined = 0).  Matched vs wrong code again.
# Pairs sharing a donor episode are not independent: every CI is a cluster bootstrap over donor episodes (task, demo).
def evaluate(a):
    import torch
    from ..common import features as Fz
    from ..common.model import load_ckpt
    from ..common.pipeline import Featurizer
    dev = "cuda"; cos = torch.nn.functional.cosine_similarity
    models = {}
    for tag in a.models:
        m, dstats, ck = load_ckpt(os.path.join(CKPT, tag + ".pt"), dev)
        ar = ck.get("args", {}); models[tag] = (m.eval(), Featurizer(None, 4, ar.get("pool", "acdc"), ar.get("modes", 2), dstats, dev))
    dst = {tag: float(f.dstats) for tag, (_, f) in models.items()}
    assert len(set(round(v, 6) for v in dst.values())) == 1, dst              # one dF scale -> one Y_real for all models
    feat = next(iter(models.values()))[1]
    rng = np.random.default_rng(0)

    def code(m, fz, g, d):
        return m.encode_batch(dict(f0=g[:, 0], dF=d), "clean")[0] if hasattr(m, "encode_batch") else m.encode(fz.cells_of(d))[0]

    for suite in a.suites:
        P = np.load(os.path.join(OUT, f"pairs_{suite}.npz"), allow_pickle=True); n = len(P["kind"])
        task, kind = P["task"].astype(int), P["kind"]
        di = np.array([int(str(x).split("_")[-1]) for x in P["di"]])
        cluster = task * 1000 + di                                             # donor episode
        gkey = np.where(kind == "grasp", cluster, 10 ** 6 + np.arange(n))       # grasp: one donor clip per (task, demo)
        groups, gid = np.unique(gkey, return_inverse=True)
        first = np.array([np.flatnonzero(gid == g)[0] for g in range(len(groups))])
        gtask, gkind = task[first], kind[first]
        wrong = np.empty(len(groups), int)                                     # a donor of the same kind from ANOTHER task
        for g in range(len(groups)):
            cand = np.flatnonzero((gtask != gtask[g]) & (gkind == gkind[g]))
            if not len(cand): cand = np.flatnonzero(gtask != gtask[g])
            wrong[g] = rng.choice(cand)
        wsrc = first[wrong[gid]]                                               # pair index whose donor clip is the wrong donor
        out, keep = {}, {}
        for b in range(0, n, a.batch):
            sl = np.arange(b, min(n, b + a.batch))
            T = lambda k, idx: torch.from_numpy(P[k][idx]).to(dev).float()
            gx, gy, gw = T("GX", sl), T("GY", sl), T("GX", wsrc[sl])
            dx, dy, dw = (Fz.dF_from_g4(g, feat.dstats) for g in (gx, gy, gw))
            c = torch.zeros(len(sl), 768, device=dev)                         # training had no lang_t5 cache: c was always 0
            u0 = lambda m: torch.zeros(len(sl), m.d_u, device=dev)             # u = 0: the recipient's real clip cannot leak
            preds = {"copy_donor": dx, "zero": torch.zeros_like(dy)}
            with torch.no_grad():
                for tag, (m, fz) in models.items():
                    preds[tag] = m.decode(gy[:, 0], c, code(m, fz, gx, dx), u0(m))
                    preds[tag + "@wrongz"] = m.decode(gy[:, 0], c, code(m, fz, gw, dw), u0(m))
                    preds[tag + "@ownz"] = m.decode(gy[:, 0], c, code(m, fz, gy, dy), u0(m))   # ceiling: its own code
            psi = lambda d: feat.cells_of(d).flatten(1)
            py, px = psi(dy), psi(dx); uu = px / px.norm(dim=1, keepdim=True).clamp_min(1e-8)
            proj = lambda v: v - (v * uu).sum(1, keepdim=True) * uu
            cells = torch.as_tensor(P["cell"][sl], device=dev).clamp_min(0)
            idx = cells[:, None, None, None].expand(-1, dy.shape[1], 1, dy.shape[-1]); oy = dy.gather(2, idx)[:, :, 0]
            for k, yh in preds.items():
                ph = psi(yh); oh = yh.gather(2, idx)[:, :, 0]
                vals = dict(cos=cos(ph, py, dim=1), proj_cos=cos(proj(ph), proj(py), dim=1),
                            res_cos=cos(ph - px, py - px, dim=1), cos_dc=cos(yh.sum(1).flatten(1), dy.sum(1).flatten(1), dim=1),
                            err=(yh - dy).pow(2).flatten(1).sum(1), nrm=dy.pow(2).flatten(1).sum(1),
                            obj_err=(oh - oy).pow(2).flatten(1).sum(1), obj_nrm=oy.pow(2).flatten(1).sum(1),
                            obj_cos=cos(oh.flatten(1), oy.flatten(1), dim=1))
                for q, v in vals.items(): out.setdefault(f"{k}/{q}", []).append(v.cpu())
                if k in models or k.endswith("@wrongz") or k == "_real":
                    keep.setdefault(k, []).append(ph.half().cpu())
            keep.setdefault("_real", []).append(py.half().cpu())
            out.setdefault("_pair/dist", []).append((Fz.quadrants(gy[:, 0]) - Fz.quadrants(gx[:, 0])).flatten(1).norm(dim=1).cpu())
            out.setdefault("_pair/real_vs_donor_cos", []).append(cos(py, px, dim=1).cpu())
        res = {k: torch.cat(v).numpy() for k, v in out.items()}
        if "GX2" in P.files and len(P["GX2"]):                               # replay noise floor (same state + actions twice)
            key = np.array([f"{t}_{d}" for t, d in zip(P["task"], P["di"])]); first_of = {key[i]: i for i in np.flatnonzero(P["kind"] == "grasp")[::-1]}
            i1 = np.array([first_of[k] for k in P["XD2_key"]])
            g1, g2 = torch.from_numpy(P["GX"][i1]).float().to(dev), torch.from_numpy(P["GX2"]).float().to(dev)
            d1, d2 = Fz.dF_from_g4(g1, feat.dstats), Fz.dF_from_g4(g2, feat.dstats)
            res["_noise/nmse"] = ((d2 - d1).pow(2).flatten(1).sum(1) / d1.pow(2).flatten(1).sum(1)).cpu().numpy()
            res["_noise/cos"] = cos(feat.cells_of(d2).flatten(1), feat.cells_of(d1).flatten(1), dim=1).cpu().numpy()
        keep = {k: torch.cat(v).float() for k, v in keep.items()}
        # ---- difference-in-differences over recipient pairs of the same grasp donor
        D = {"did_cluster": [], "did_sb": [], "did_sc": []}
        for g in range(len(groups)):
            ii = np.flatnonzero(gid == g)
            if gkind[g] != "grasp" or len(ii) < 2: continue
            b_, c_ = np.triu_indices(len(ii), 1); B, C = ii[b_], ii[c_]
            dr = keep["_real"][B] - keep["_real"][C]
            for k in keep:
                if k == "_real": continue
                D.setdefault(f"did/{k}", []).append(cos(keep[k][B] - keep[k][C], dr, dim=1).numpy())
            D["did_cluster"].append(cluster[B]); D["did_sb"].append(P["stratum"][B]); D["did_sc"].append(P["stratum"][C])
        for k, v in D.items(): res[k] = np.concatenate(v) if v else np.zeros(0)
        for q in ("kind", "stratum", "shift", "cell", "task", "di", "dj", "det", "lead"):
            if q in P: res[q] = P[q]
        res["cluster"] = cluster
        np.savez_compressed(os.path.join(OUT, f"eval_{suite}.npz"), **res)
        print(f"[eval] {suite} {n} pairs, {len(groups)} donor groups, {len(res['did_cluster'])} DiD recipient pairs", flush=True)


def cboot(x, cl, B=2000, seed=0):
    """mean of x and a 95% cluster-bootstrap CI (clusters = donor episodes)."""
    x = np.asarray(x, float); ok = np.isfinite(x); x, cl = x[ok], cl[ok]
    if not len(x): return (float("nan"),) * 3
    u, inv = np.unique(cl, return_inverse=True); s = np.bincount(inv, x); c = np.bincount(inv).astype(float)
    r = np.random.default_rng(seed).integers(0, len(u), (B, len(u)))
    bs = s[r].sum(1) / c[r].sum(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize(a):
    E = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(os.path.join(OUT, "eval_*.npz")))]
    cat = lambda k: np.concatenate([e[k] for e in E])
    kind, st, cl, dist = cat("kind"), cat("stratum"), cat("cluster"), cat("_pair/dist")
    tags = sorted({k.split("/")[0] for k in E[0].files if "/" in k and not k.startswith(("_", "did"))})
    import re
    if all("_noise/nmse" in e.files for e in E):
        nn_, nc_ = cat("_noise/nmse"), cat("_noise/cos")
        print(f"replay noise floor (donor executed twice, {len(nn_)} donors): NMSE median {np.median(nn_):.4f} max {nn_.max():.4f}; Psi cos median {np.median(nc_):.4f} min {nc_.min():.4f}")
    arm = lambda t: re.sub(r"_s\d+", "", t)                                 # seeds of one arm are averaged per pair
    arms = {}
    for t in tags: arms.setdefault(arm(t), []).append(t)
    groups = {"generic": kind == "generic", "grasp": kind == "grasp"}
    for s in ("centered", "offset", "outside"): groups[s] = st == s
    for g in ("generic", "grasp"):
        m = groups[g]; med = np.median(dist[m]) if m.any() else 0
        groups[f"{g}_near"], groups[f"{g}_far"] = m & (dist <= med), m & (dist > med)
    V = lambda A, q: np.mean([cat(f"{t}/{q}") for t in arms[A]], 0)
    rows = {}
    fmt = lambda r: f"{r[0]:+.3f} [{r[1]:+.3f},{r[2]:+.3f}]"
    for g, msk in groups.items():
        R = rows[g] = {"n": int(msk.sum()), "clusters": int(len(np.unique(cl[msk]))),
                       "real_vs_donor_cos": float(cat("_pair/real_vs_donor_cos")[msk].mean()) if msk.any() else None}
        print(f"\n== {g}  n={R['n']} donor episodes={R['clusters']}  cos(real, donor)={R['real_vs_donor_cos']}")
        print(f"  {'predictor':30s} {'proj_cos [95% CI]':>26s} {'raw res_cos':>11s} {'cos':>7s} {'NMSE':>7s} {'objNMSE':>8s}")
        for A in arms:
            pc = cboot(V(A, "proj_cos")[msk], cl[msk]); nm = V(A, "err")[msk].sum() / V(A, "nrm")[msk].sum()
            on = V(A, "obj_err")[msk].sum() / max(V(A, "obj_nrm")[msk].sum(), 1e-9)
            R[A] = dict(proj_cos=pc, res_cos=float(V(A, "res_cos")[msk].mean()), cos=float(V(A, "cos")[msk].mean()), nmse=float(nm),
                        obj_nmse=None if g.startswith("generic") else float(on))
            print(f"  {A:30s} {fmt(pc):>26s} {R[A]['res_cos']:>+11.3f} {R[A]['cos']:>7.3f} {nm:>7.3f} " +
                  (f"{on:>8.3f}" if R[A]["obj_nmse"] is not None else ""))
        for A in arms:                                                       # the claim: matched - wrong code
            if A.endswith(("@wrongz", "@ownz")) or A in ("copy_donor", "zero"): continue
            gap = cboot(V(A, "proj_cos")[msk] - V(A + "@wrongz", "proj_cos")[msk], cl[msk])
            R[A]["gap_vs_wrongz"] = gap; print(f"  GAP {A:26s} matched - wrong donor, proj_cos: {fmt(gap)}")
    # DiD
    dcl, sb, sc = cat("did_cluster"), cat("did_sb"), cat("did_sc")
    if len(dcl):
        Dv = lambda A: np.mean([cat(f"did/{t}") for t in arms[A]], 0)
        cats = {"all": np.ones(len(dcl), bool), "both centered": (sb == "centered") & (sc == "centered"),
                "both outside": (sb == "outside") & (sc == "outside"), "centered vs outside": ((sb == "centered") & (sc == "outside")) | ((sb == "outside") & (sc == "centered")),
                "any offset": (sb == "offset") | (sc == "offset")}
        rows["did"] = {}
        print("\n== DiD: same donor, two recipients: cos(Y_hat_B - Y_hat_C, Y_real_B - Y_real_C)")
        for name, msk in cats.items():
            print(f"  [{name}] recipient pairs={int(msk.sum())} donor episodes={len(np.unique(dcl[msk]))}")
            rows["did"][name] = {}
            for A in arms:
                if A.endswith("@wrongz") or A.endswith("@ownz") or f"did/{arms[A][0]}" not in E[0].files: continue
                m_ = cboot(Dv(A)[msk], dcl[msk]); w_ = cboot(Dv(A + "@wrongz")[msk], dcl[msk]); gp = cboot(Dv(A)[msk] - Dv(A + "@wrongz")[msk], dcl[msk])
                rows["did"][name][A] = dict(matched=m_, wrong=w_, gap=gp)
                print(f"    {A:26s} matched {fmt(m_)}  wrong {fmt(w_)}  gap {fmt(gp)}")
    json.dump(rows, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["probe", "collect", "eval", "summary", "cache"])
    ap.add_argument("--generic", type=int, default=6, help="generic (non-grasp) donor chunks per demo")
    ap.add_argument("--models", nargs="+", default=[f"eatabm_{m}_s{s}" for m in ("full", "recon") for s in (0, 1, 2)])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--suites", nargs="+", default=SUITES)
    ap.add_argument("--tasks", type=int, default=10); ap.add_argument("--demos", type=int, default=8)
    ap.add_argument("--lead", type=int, default=6, help="probe only: chunk starts this many steps before the close command")
    ap.add_argument("--leads", type=int, nargs="+", default=[2, 4, 6],
                    help="collect: anchor offsets tried in order, first whose chunk ends in a grasp wins. "
                         "Negative = chunk starts AFTER the close command; 6 4 2 0 -4 -8 -12 -16 recovers the "
                         "slow-grasp tasks and is a strict superset of the default")
    ap.add_argument("--shifts", type=int, nargs="+", default=[-8, -4, 0, 4, 8])
    ap.add_argument("--save-png", action="store_true")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    {"cache": cache, "probe": probe, "collect": collect, "eval": evaluate, "summary": summarize}[a.mode](a)


if __name__ == "__main__":
    main()
