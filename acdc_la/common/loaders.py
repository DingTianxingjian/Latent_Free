"""Train / val loaders over the OXE mixture, shared by the three stages.

Train batches are SOURCE-HOMOGENEOUS (one source per batch, drawn by UniVLA's mixture probability):
the swap realism term is a within-source distribution match, and a wrong partner must be a wrong
partner, not a wrong embodiment.  Val is a fixed stratified set of (anchor, nearest-action partner)
pairs per source, laid out [a0,p0,a1,p1,...] so the matched-donor readers (E_correct) have a partner
at index i^1; the pair index is cached under results/.
"""
import os, json
import numpy as np
import torch

from ..paths import RESULTS
from ..data import oxe_mix


def build(a, pairs_for_train=False, start_step=0):
    srcs = a.sources or oxe_mix.available_sources()
    ds = oxe_mix.MixDS(srcs, a.T, a.stride, "train", limit_eps=a.limit_eps, with_actions=True)
    dv = oxe_mix.MixDS(srcs, a.T, a.stride, "val", limit_eps=a.limit_eps, with_actions=True)
    ds.manifest["batch_mode"] = "source"
    json.dump(ds.manifest, open(os.path.join(RESULTS, f"{a.tag}_manifest.json"), "w"), indent=1)
    if pairs_for_train:
        ds.build_action_index(cache_path=os.path.join(RESULTS, "oxe_action_pairs_train.npz"))
        sampler = list(ds.pair_batches(a.batch, a.steps + 8, seed=a.seed, max_dist=getattr(a, "pair_max_dist", None)))
    else:
        sampler = list(ds.source_batches(a.batch, a.steps + 8, seed=a.seed))
    sampler = sampler[start_step:]                                    # resume: the batch list is deterministic in seed
    dl = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    dv.build_action_index(cache_path=os.path.join(RESULTS, "oxe_action_pairs_val.npz"))
    val_idx = dv.stratified_pair_indices(a.val_per_source // 2, seed=123)
    assert a.batch % 2 == 0
    dl_val = torch.utils.data.DataLoader(torch.utils.data.Subset(dv, val_idx), batch_size=a.batch, shuffle=False,
                                         num_workers=min(4, a.workers), drop_last=True)
    return srcs, ds, dv, dl, dl_val


def describe(a, srcs, ds, dv, dev):
    print(f"[{a.tag}] sources={len(srcs)} train_clips={len(ds)} val_clips={len(dv)} T={a.T} stride={a.stride} K={a.T-1} "
          f"batch={a.batch} steps={a.steps} dev={dev}", flush=True)
    print(f"[{a.tag}] ACDC_FULL_DATA={os.environ.get('ACDC_FULL_DATA', '0')} (1 = uncapped caches where converted)", flush=True)
    for p in ds.manifest["sources"]:
        print(f"[{a.tag}]   {p['source'][:44]:44s} clips={p['clips']:8d} eps={p['episodes_all']:6d} p_univla={p['p_univla']:.4f} cache={p.get('cache', '?').split('/cache/')[-1]}", flush=True)
