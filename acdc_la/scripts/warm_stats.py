"""Compute and cache the RLDS dataset statistics for acdc_oxe29 in ONE process, before any DDP run.

Each DDP rank otherwise computes them independently (same scan, same accumulation, twice over) on top of its own 7B
model, which OOM-killed the 2-rank smoke on this 62 GB host after 100 minutes.  Statistics are cached next to the data
as dataset_statistics_<hash>.json and are then read, not recomputed, by every later run.
"""
import json, os, sys, time
from acdc_la.paths import LF, IMG
from prismatic.vla.datasets import RLDSDataset

t0 = time.time()
ds = RLDSDataset(os.path.join(LF, "data", "oxe_raw"), "acdc_oxe29", lambda b: None, resize_resolution=(IMG, IMG),
                 shuffle_buffer_size=256, image_aug=False, window_size=14, training_phase="post-training")
print(f"[warm] statistics ready for {len(ds.dataset_statistics)} sources in {int(time.time()-t0)} s", flush=True)
