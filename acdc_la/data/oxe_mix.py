"""Multi-source clip dataset over the OXE caches, sampled the way UniVLA samples.

Each source is a cache dir with frames.npy (N,256,256,3 u8) + index.npz (starts/lengths/
is_val/actions) -- `bridge_big` and every `cache/oxe/<name>` written by oxe_convert.py.
Clips are T frames `stride` apart inside one episode, exactly as rala2.ClipDS builds them,
so a T=2 clip is the pair the audit probes.

Sampling follows UniVLA's loader (`rlds/dataset.py`, balance_weights=True):
    p_i = w_i * n_transitions_i / sum_j w_j * n_transitions_j
where n_transitions_i is the number of clips the converted cache actually provides (after
the per-source episode cap), NOT the size of the original dataset.  Both the requested
weights and the realised probabilities are written to the manifest so the deviation from
their full-scale mixture is a number, not a footnote.

Returns (clip u8 (T,256,256,3), global_episode_id, lang_row).  lang_row indexes a T5 cache;
sources without one return row 0 (the all-masked row) and are listed in the manifest as
`lang=False`, so a language arm trained on the mix sees language only where a cache exists.
"""
import os, json
import numpy as np
import torch

from ..paths import CACHE

# UniVLA `omni_magic_soup_plus_plus` weights (mixtures.py); bridge_oxe -> our bridge_big.
UNIVLA_W = {
    "bridge_big": 1.0, "fractal20220817_data": 0.5, "kuka": 0.1, "taco_play": 2.0, "jaco_play": 1.0,
    "berkeley_cable_routing": 1.0, "roboturk": 2.0, "viola": 2.0, "berkeley_autolab_ur5": 2.0, "toto": 1.0,
    "language_table": 0.1, "stanford_hydra_dataset_converted_externally_to_rlds": 2.0,
    "austin_buds_dataset_converted_externally_to_rlds": 1.0, "nyu_franka_play_dataset_converted_externally_to_rlds": 3.0,
    "furniture_bench_dataset_converted_externally_to_rlds": 0.1, "ucsd_kitchen_dataset_converted_externally_to_rlds": 2.0,
    "austin_sailor_dataset_converted_externally_to_rlds": 1.0, "austin_sirius_dataset_converted_externally_to_rlds": 1.0,
    "dlr_edan_shared_control_converted_externally_to_rlds": 1.0, "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 1.0,
    "utaustin_mutex": 1.0, "berkeley_fanuc_manipulation": 2.0, "cmu_stretch": 1.0, "bc_z": 0.2, "fmb": 1.0, "dobbe": 0.2,
    "berkeley_gnm_recon": 2.0, "berkeley_gnm_cory_hall": 2.0, "berkeley_gnm_sac_son": 2.0,
}

# Bridge V2 train split in the RAIL copy (data/bridge_raw/dataset_info.json); our bridge_big holds 5,000 of these.
BRIDGE_EPISODES_AVAILABLE = 53192

FULL = os.environ.get("ACDC_FULL_DATA", "0") == "1"   # 1 = use the uncapped caches (cache/oxe_full, cache/bridge_full) where they exist


def source_dir(name):
    capped = os.path.join(CACHE, "bridge_big" if name == "bridge_big" else os.path.join("oxe", name))
    if FULL:
        full = os.path.join(CACHE, "oxe_full", "bridge_oxe" if name == "bridge_big" else name)   # bridge_big -> UniVLA's bridge_oxe, uncapped
        if os.path.exists(os.path.join(full, "meta.json")): return full
    return capped

def available_sources():
    """Every source whose cache is complete (meta.json for oxe caches, index.npz for bridge_big)."""
    out = []
    for n in UNIVLA_W:
        d = source_dir(n)
        if n == "bridge_big" and os.path.exists(os.path.join(d, "index.npz")): out.append(n)
        elif os.path.exists(os.path.join(d, "meta.json")): out.append(n)
    return out

class MixDS(torch.utils.data.Dataset):
    def __init__(self, sources, T, stride, split="train", weights=None, limit_eps=None, with_actions=False):
        """with_actions=True additionally returns, per clip:
             act   (T-1, 7) f32  displacement over each stride segment -- the SUM of the raw
                                 per-step actions for delta sources, a[t+stride]-a[t] for
                                 sources whose action is an absolute pose (meta.json
                                 action_kind), xyz only trusted; gripper = last value.
             lang  (768,) f32    masked-mean pooled frozen-T5 embedding of the episode's
                                 instruction, zeros if the source has no cache or the
                                 episode has no instruction.
             src   int           source index (order of `sources`)."""
        self.T, self.stride = T, stride
        self.with_actions = with_actions
        span = (T - 1) * stride
        want_val = (split == "val")
        self.src, self.frames, self.lang, self.langvec, self.actions, self.abs_action, per = [], [], [], [], [], [], []
        self.paths, self.offsets, self.fbytes, self._fds = [], [], [], {}
        rows, ep_off = [], 0
        for si, name in enumerate(sources):
            d = source_dir(name); z = np.load(os.path.join(d, "index.npz"))
            starts, lengths, is_val = z["starts"], z["lengths"], z["is_val"].astype(bool)
            fr = np.load(os.path.join(d, "frames.npy"), mmap_mode="r")
            self.frames.append(fr)                                   # kept for shape/dtype and external readers
            # frame access goes through os.pread, not the mmap: random touches into 25 mapped files
            # totalling ~600 GB on NFS trigger multi-MB kernel readahead per frame, and 14 workers
            # sat in D-state at 10-13 GB RSS each with the GPU idle (job 20436140).  A positional
            # read of exactly one 196,608-byte frame has no such amplification.
            self.paths.append(os.path.join(d, "frames.npy")); self.offsets.append(int(fr.offset)); self.fbytes.append(int(np.prod(fr.shape[1:])))
            self.actions.append(z["actions"].astype(np.float32))
            lc = os.path.join(d, "lang_t5.npz")
            if os.path.exists(lc):
                L = np.load(lc); self.lang.append(L["ep2row"])
                e, m = L["emb"].astype(np.float32), L["mask"].astype(np.float32)[..., None]
                self.langvec.append((e * m).sum(1) / np.maximum(m.sum(1), 1.0))      # (n_rows, 768); row 0 -> 0
            else:
                self.lang.append(None); self.langvec.append(None)
            eps = [e for e in range(len(starts)) if bool(is_val[e]) == want_val]
            if limit_eps: eps = eps[:limit_eps]
            n0 = len(rows)
            for e in eps:
                s, L = int(starts[e]), int(lengths[e])
                for t in range(0, L - span): rows.append((si, s + t, ep_off + e, e))
            meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
            self.abs_action.append(meta.get("action_kind") == "absolute")
            per.append(dict(source=name, cache=d, w=float((weights or UNIVLA_W)[name]), episodes=len(eps), clips=len(rows) - n0,
                            lang=self.lang[-1] is not None, action_kind=meta.get("action_kind", "delta" if name == "bridge_big" else "unknown"),
                            episodes_all=int(len(starts)), frames=int(lengths.sum()),
                            episodes_available=meta.get("episodes_available", BRIDGE_EPISODES_AVAILABLE if name == "bridge_big" else None)))
            self.src.append(name); ep_off += len(starts)
        self.index = np.array(rows, dtype=np.int64)           # (n, 4): source, first_row, global_ep, local_ep
        # UniVLA's balanced sampling is p_i ~ w_i * n_transitions_i over the FULL datasets.  Our
        # caches cap each source at 5000 episodes, and the cap binds very unevenly (bridge 5k of
        # 53k episodes; dobbe 5k of 5.2k but 219 frames each), so p ~ w * n_clips_converted would
        # give dobbe MORE samples than Bridge -- the opposite of their mix.  So: estimate each
        # source's full size as episodes_available x mean converted episode length, take UniVLA's
        # probabilities from that, and spread each source's probability uniformly over the clips
        # we actually hold.  The cap then limits within-source diversity only.  Both this and the
        # cache-count version are written to the manifest.
        for p in per:
            mean_len = p["frames"] / max(1, p["episodes_all"])
            p["n_transitions_full_est"] = int(round((p["episodes_available"] or p["episodes_all"]) * max(0.0, mean_len - span)))
        Zf = sum(p["w"] * p["n_transitions_full_est"] for p in per); Zc = sum(p["w"] * p["clips"] for p in per)
        for p in per:
            p["p_univla"] = round(float(p["w"] * p["n_transitions_full_est"] / Zf), 5)      # used
            p["p_univla_on_cache_counts"] = round(float(p["w"] * p["clips"] / Zc), 5)        # what the cap alone would give
            p["p_uniform_clips"] = round(p["clips"] / len(rows), 5)
        per_item = np.array([per[si]["p_univla"] / max(1, per[si]["clips"]) for si in self.index[:, 0]])
        self.item_w = torch.as_tensor(per_item, dtype=torch.double)
        self.manifest = dict(T=T, stride=stride, split=split, n_sources=len(sources), n_clips=int(len(rows)),
                             sampling="p_i = w_i * n_transitions_i(full, est. = episodes_available x mean converted episode length) / Z; "
                                      "spread uniformly over the converted clips of source i (UniVLA rlds/dataset.py balance_weights=True)",
                             sources=per)

    def __getstate__(self):
        d = self.__dict__.copy(); d["_fds"] = {}; return d

    def stratified_indices(self, per_source, seed=123):
        """`per_source` clip indices drawn uniformly WITHIN each source (all of a source's clips if it
        has fewer).  For readouts: a per-source probe needs a floor of clips per source, which the
        UniVLA-weighted sampler does not give (ucsd_kitchen is 0.1% of it)."""
        rs = np.random.RandomState(seed); out = []
        for si in range(len(self.src)):
            rows = np.flatnonzero(self.index[:, 0] == si)
            out.append(rs.choice(rows, min(per_source, len(rows)), replace=False))
        return np.concatenate(out)

    def source_batches(self, batch, num_batches, seed=0):
        """Source-HOMOGENEOUS batches: draw the source per batch by its UniVLA probability, then
        `batch` clips uniformly within that source (with replacement across batches).  Marginal
        source frequencies are identical to `sampler()`; what changes is that every in-batch
        partner shares embodiment, camera and action units -- required for the swap realism term
        (a within-source distribution match), for the oracle's nearest-action donor (job 20442294:
        mixed-source batches made oracle transport no better than random pairing) and for the
        control to be the same objective with a wrong partner rather than a wrong embodiment."""
        rs = np.random.RandomState(seed)
        p = np.array([q["p_univla"] for q in self.manifest["sources"]], np.float64); p /= p.sum()
        rows_by_src = [np.flatnonzero(self.index[:, 0] == si) for si in range(len(self.src))]
        for _ in range(num_batches):
            si = rs.choice(len(p), p=p)
            yield rs.choice(rows_by_src[si], batch, replace=len(rows_by_src[si]) < batch).tolist()

    # ---- action-matched pairing (grounded swap) -------------------------------------------------
    def _action_seq(self, rows):
        """(n, K*3) per-segment xyz displacement for clip index rows (same recipe as __getitem__)."""
        K, S = self.T - 1, self.stride; out = np.zeros((len(rows), K * 3), np.float32)
        for n, r in enumerate(rows):
            si, g, _, _ = self.index[r]; A = self.actions[si]
            if self.abs_action[si]:
                rr = g + np.arange(self.T) * S; out[n] = (A[rr[1:], :3] - A[rr[:-1], :3]).ravel()
            else:
                out[n] = np.stack([A[g + k * S: g + (k + 1) * S, :3].sum(0) for k in range(K)]).ravel()
        return out

    def build_action_index(self, cache_path=None, max_per_source=100000, seed=0):
        """Per source: the action sequences of (up to) max_per_source train clips, standardised per
        source, and for EVERY such clip its nearest-action partner from a DIFFERENT episode of the
        same source (the Bridge oracle's recipe -- dataset-wide, not nearest-of-32).  Cached."""
        import os, hashlib
        # the cache is only valid for the exact dataset it was built from: key it by a signature of
        # (sources, T, stride, n_clips, per-source clip counts, max_per_source) and validate on load
        sig = hashlib.md5(("|".join(self.src) + f"|T{self.T}|s{self.stride}|n{len(self.index)}|" +
                           ",".join(str(int((self.index[:, 0] == si).sum())) for si in range(len(self.src))) + f"|m{max_per_source}").encode()).hexdigest()[:10]
        if cache_path:
            root, ext = os.path.splitext(cache_path); cache_path = f"{root}_{sig}{ext}"
        if cache_path and os.path.exists(cache_path):
            z = np.load(cache_path, allow_pickle=True)
            if int(z["n_index"]) == len(self.index) and z["rows"].max() < len(self.index) and z["partner"].max() < len(self.index):
                self.pair_rows, self.pair_partner, self.pair_dist, self.act_std = z["rows"], z["partner"], z["dist"], z["act_std"]
                return
        rs = np.random.RandomState(seed); rows_all, part_all, dist_all, std_all = [], [], [], []
        for si in range(len(self.src)):
            rows = np.flatnonzero(self.index[:, 0] == si)
            if len(rows) > max_per_source: rows = np.sort(rs.choice(rows, max_per_source, replace=False))
            A = self._action_seq(rows); sd = A.std(0) + 1e-6; An = A / sd; ge = self.index[rows, 2]
            partner = np.full(len(rows), -1); dist = np.full(len(rows), np.inf, np.float32)
            # chunked brute force on standardised sequences, excluding same-episode candidates; on the
            # GPU when there is one (100k x 100k x 12 per source is seconds there, ~15 min in numpy)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            At = torch.as_tensor(An, device=dev); get = torch.as_tensor(ge, device=dev); sq = (At ** 2).sum(1)
            for a0 in range(0, len(rows), 4096):
                q = At[a0:a0 + 4096]
                d2 = (q ** 2).sum(1, keepdim=True) + sq[None, :] - 2 * q @ At.T
                d2[get[a0:a0 + 4096, None] == get[None, :]] = float("inf")
                j = d2.argmin(1); partner[a0:a0 + 4096] = rows[j.cpu().numpy()]
                dist[a0:a0 + 4096] = d2.gather(1, j[:, None])[:, 0].clamp_min(0).sqrt().cpu().numpy()
            rows_all.append(rows); part_all.append(partner); dist_all.append(dist); std_all.append(sd)
        self.pair_rows, self.pair_partner, self.pair_dist = np.concatenate(rows_all), np.concatenate(part_all), np.concatenate(dist_all)
        self.act_std = np.stack(std_all)
        if cache_path: np.savez(cache_path, rows=self.pair_rows, partner=self.pair_partner, dist=self.pair_dist, act_std=self.act_std, n_index=len(self.index))

    def stratified_pair_indices(self, per_source, seed=123, max_dist=None):
        """For readouts: per source, `per_source` anchors with their nearest-action partners, laid out
        [a0,p0,a1,p1,...] and grouped by source so batches stay source-pure."""
        assert hasattr(self, "pair_rows"), "call build_action_index first"
        rs = np.random.RandomState(seed); src_of = self.index[self.pair_rows, 0]; out = []
        for si in range(len(self.src)):
            b = np.flatnonzero(src_of == si)
            if max_dist is not None: b = b[self.pair_dist[b] <= max_dist]
            k = rs.choice(b, min(per_source, len(b)), replace=False)
            o = np.empty(2 * len(k), np.int64); o[0::2] = self.pair_rows[k]; o[1::2] = self.pair_partner[k]; out.append(o)
        return np.concatenate(out)

    def pair_batches(self, batch, num_batches, seed=0, max_dist=None):
        """Source-homogeneous batches of batch/2 (anchor, nearest-action partner) pairs laid out as
        [a0, p0, a1, p1, ...].  Anchors drawn by UniVLA source probability; pairs with standardised
        action distance > max_dist are skipped (the loss also weights by distance)."""
        rs = np.random.RandomState(seed)
        p = np.array([q["p_univla"] for q in self.manifest["sources"]], np.float64); p /= p.sum()
        src_of = self.index[self.pair_rows, 0]
        by_src = [np.flatnonzero(src_of == si) for si in range(len(self.src))]
        if max_dist is not None: by_src = [b[self.pair_dist[b] <= max_dist] for b in by_src]
        half = batch // 2
        for _ in range(num_batches):
            si = rs.choice(len(p), p=p)
            if len(by_src[si]) < half: continue
            k = rs.choice(by_src[si], half, replace=False)
            out = np.empty(2 * half, np.int64); out[0::2] = self.pair_rows[k]; out[1::2] = self.pair_partner[k]
            yield out.tolist()

    def sampler(self, num_samples=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.utils.data.WeightedRandomSampler(self.item_w, num_samples or len(self), replacement=True, generator=g)

    def __len__(self): return len(self.index)

    def _read_frames(self, si, rows):
        fd = self._fds.get(si)
        if fd is None:                                               # one descriptor per source per worker
            fd = self._fds[si] = os.open(self.paths[si], os.O_RDONLY)
        nb, shape = self.fbytes[si], self.frames[si].shape[1:]
        out = np.empty((len(rows),) + tuple(shape), np.uint8)
        for k, r in enumerate(rows):
            buf = os.pread(fd, nb, self.offsets[si] + int(r) * nb)
            assert len(buf) == nb, (self.paths[si], r, len(buf))
            out[k] = np.frombuffer(buf, np.uint8).reshape(shape)
        return out

    def __getitem__(self, i):
        si, g, ge, le = self.index[i]
        rows = g + np.arange(self.T) * self.stride
        clip = self._read_frames(si, rows)
        lrow = int(self.lang[si][le]) if self.lang[si] is not None else 0
        if not self.with_actions:
            return torch.from_numpy(clip.copy()), int(ge), lrow
        A = self.actions[si]; K = self.T - 1; S = self.stride
        if self.abs_action[si]:
            act = A[rows[1:]] - A[rows[:-1]]                                  # pose difference per segment
            act[:, 6] = A[rows[1:], 6]
        else:
            act = np.stack([A[g + k * S: g + (k + 1) * S].sum(0) for k in range(K)])   # summed deltas
            act[:, 6] = A[rows[1:] - 1, 6]
        lv = self.langvec[si][lrow] if self.langvec[si] is not None else np.zeros(768, np.float32)
        return (torch.from_numpy(clip.copy()), int(ge), lrow,
                torch.from_numpy(act.astype(np.float32)), torch.from_numpy(lv.astype(np.float32)), int(si))
