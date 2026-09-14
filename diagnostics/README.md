# Diagnostics — the completed label-free-LAM probing study

Archived one-off scripts for the finished diagnostic paper (see top-level
`SUMMARY.md`). All results are in `../results/*.csv`; features are cached under
`../cache/`. These are kept for provenance and are **not** on the Phase-1 path.

They import the top-level shared modules (`data`, `probes`, `probes_torch`,
`features`, `features_vjepa`, `features_wan`), so to re-run one, run it from the
repo root with the root on the path, e.g.:

```bash
cd /anvil/scratch/x-tding2/latent_free
PYTHONPATH=. envs/latent_free/bin/python diagnostics/exp6b_vjepa_temporal.py
```

`run/` holds the matching SLURM wrappers (their `cd` already points at the repo
root; they were written to be submitted from anywhere).

Map of what each answers is in `SUMMARY.md`; the two gating experiments that
green-lit Phase 1 are `exp5c_framecount_null.py` (Wan fusion ≠ frame count) and
`exp6b_vjepa_temporal.py` (V-JEPA2 fusion is general and stronger).
