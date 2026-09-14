# UniVLA dataset readiness audit — 2026-09-06

Read-only acquisition/status audit. Scheduler/process snapshot: 2026-09-06 05:44 UTC
(01:44 EDT), login01.anvil.rcac.purdue.edu. No downloader, converter, job, or training
configuration was changed. Sizes below use decimal GB/TB. Recommendations are not
implemented changes or empirical model results.

## Bottom line

The target is the local `omni_magic_soup_plus_plus`: 26 manipulation sources, 3 GNM
navigation sources, and one curated Ego4D collection represented by 4 split entries.
That is 33 mixture entries, not 33 independent underlying datasets.

The current downloader covers only 24 of 28 public-bucket sources, omitting four large
sources; it also excludes Bridge and Ego4D. Twelve sources have all expected shard names
present: 677 nonempty TFRecord files totaling 64.317927 GB. Their logs report zero failed
downloads. This is a file-count/size audit, NOT a complete checksum or record-decoding audit.

The medium/large download log is empty, the other 16 public-source directories are absent,
and no matching downloader was visible on the inspected login host. A process elsewhere
cannot be ruled out. Conversion job 20432961 is PENDING (Priority), with no runtime yet.
Do not interpret the old 'fetching' ledger entry as verified current progress.

## Source inventory

Expected sizes come from the existing GCS metadata inventory
`claude_survey/results/oxe_mix_sizes.json`; they are approximate transfer sizes, not
decoded-image storage. Default version is 0.1.0; `dobbe` and `fmb` are 0.0.1.
Names ending in `…_rlds` below abbreviate `_dataset_converted_externally_to_rlds`.

| Source | Expected GB | Local status |
| --- | ---: | --- |
| fractal20220817_data | 119.259 | Listed; no source directory |
| kuka | 835.396 | Omitted from downloader |
| taco_play | 51.289 | Listed; no source directory |
| jaco_play | 9.921 | 136/136 shards present |
| berkeley_cable_routing | 5.012 | 68/68 shards present |
| roboturk | 48.733 | Listed; no source directory |
| viola | 11.167 | 88/88 shards present |
| berkeley_autolab_ur5 | 82.023 | Listed; no source directory |
| toto | 137.071 | Listed; no source directory |
| language_table | 429.355 | Omitted from downloader |
| stanford_hydra…_rlds | 77.828 | Listed; no source directory |
| austin_buds…_rlds | 1.604 | 16/16 shards present |
| nyu_franka_play…_rlds | 5.565 | 48/48 shards present |
| furniture_bench…_rlds | 123.479 | Omitted from downloader |
| ucsd_kitchen…_rlds | 1.423 | 16/16 shards present; cache checked |
| austin_sailor…_rlds | 20.236 | Listed; no source directory |
| austin_sirius…_rlds | 7.032 | 64/64 shards present |
| dlr_edan_shared_control…_rlds | 3.316 | 29/29 shards present |
| iamlab_cmu_pickup_insert…_rlds | 54.003 | Listed; no source directory |
| utaustin_mutex | 22.326 | Listed; no source directory |
| berkeley_fanuc_manipulation | 9.504 | 124/124 shards present |
| cmu_stretch | 0.763 | 8/8 shards present |
| bc_z | 86.476 | Listed; no source directory |
| fmb | 1285.731 | Omitted from downloader |
| dobbe | 22.652 | Listed; no source directory |
| berkeley_gnm_recon | 20.108 | Listed; no source directory |
| berkeley_gnm_cory_hall | 1.490 | 16/16 shards present |
| berkeley_gnm_sac_son | 7.519 | 64/64 shards present |
| bridge_oxe | Not verified for exact target | Partial RAIL replacement only |
| ego4d_split_1 through ego4d_split_4 | Curated manifest size unverified | Not in this downloader; no acquisition verified |

Bridge replacement details: `latent_free/data/bridge_raw` contains 97 train shards
(11.102946 GB), versus metadata declaring 1024 train + 128 validation shards; no validation
shards or `.part` files were found there. Metadata describes 132.485966 GB total for this
RAIL copy. `cache/bridge_big/index.npz` contains 5000 episodes / 187587 frames, including
1000 validation episodes. This is neither full Bridge coverage nor proof of equivalence to
UniVLA's `bridge_oxe` encoding. The older '99 shards' count is not the verified TFRecord count.

UCSD conversion: 150/150 train episodes, 0 skipped, 3970 frames, 15 held-out episodes,
frames shape `(3970,256,256,3)`, actions shape `(3970,7)`, and summed episode lengths match
both arrays. Logged conversion duration is 32 seconds. Other OXE caches were not found.

## Conditional download ETA

Twelve completed source logs total 295 seconds, approximately 218 MB/s across 64.3 GB.
Per-source logged rates range approximately 157–264 MB/s. These are short historical
transfers, not an observed ongoing transfer of the missing sources.

| Scope | Total expected GB | Remaining GB | At sustained 150–250 MB/s |
| --- | ---: | ---: | --- |
| Current reduced 24-source downloader | 806.321 | 742.003 | 0.82–1.37 hours |
| All 28 public-bucket sources | 3480.281 | 3415.964 | 3.80–6.33 hours |
| Four omitted sources alone | 2673.961 | 2673.961 | 2.97–4.95 hours |

Rows overlap and must not be added together. Estimates start when an active correctly
scoped transfer runs, exclude retries/stalls, and exclude Bridge, Ego4D, conversion, feature
extraction and queue delay. A whole-pipeline completion timestamp is not currently defensible.

UniVLA's Ego4D recipe requests **v2 FHO clips + annotations**, then selects interaction
segments using `info_clips.json`, samples at 2 FPS, and builds four RLDS split entries.
It does not require the entire Ego4D archive. Resolve the exact clip manifest and byte sum
before estimating this part. Official access documentation says initial credentials take
about 48 hours and expire after 14 days; access state here was not inspected. Licensing is
a user/institution action, not something an agent should accept on the user's behalf.

Sources: [UniVLA Ego4D recipe](https://github.com/OpenDriveLab/UniVLA/blob/main/vla-scripts/extern/ego4d_rlds_dataset_builder/ego4d/README.md),
[Ego4D access instructions](https://ego4d-data.org/docs/start-here/).

The UCSD conversion observation is about 124 frames/second in one small dataset. Linear
extrapolation would be 2.24 hours per million frames for that serial throughput, but this is
NOT a validated cross-dataset conversion estimate. Benchmark representative large/video
sources and CPU parallelism before assigning a full conversion budget. Feature extraction
requires its own measured GPU throughput and does not appear in the network ETA.

## Corrections to earlier assumptions

### Sampling weights are multipliers, not direct source probabilities

`prismatic/vla/datasets/datasets.py` explicitly sets `balance_weights=True`.
`rlds/dataset.py` multiplies each weight by its recorded `num_transitions`, then normalizes:

`p_i = w_i * N_i / sum_j(w_j * N_j)`.

The current local mixture's raw weights sum to **40.2**, not 44.3. More importantly,
normalizing raw weights alone is not this sampler. Prior claims that four omitted sources
constitute '3% of samples' and Ego4D '9%' are retracted. Actual probabilities require the
dataset statistics and runtime loader audit; filtering/caps also change effective coverage.
Do not estimate training importance from compressed bytes, episode counts alone, or raw w.
Save both requested and observed source frequencies, frame/window counts and filtering rates.

### Pretraining windows currently contain endpoints only

`rlds/traj_transforms.py::chunk_act_obs` gathers exactly the first and last observations.
`rlds/dataset.py` selects this path for LAM and policy pretraining, and adjusts intervals by
source. Ego4D uses an interval of two frames. `num_frames: 16` in a YAML does not by itself
make this loader return 16 frames.

AC/DC over a transition sequence requires intermediate observations: at least two genuine
temporal deltas to have a nontrivial AC coefficient, and more samples for useful profiles.
Define and implement a real sequence-window loader; validate returned tensor shapes,
timestamps, duration and padding, including short Ego4D clips. Never fabricate a temporal
profile by repeating endpoints. The original endpoint-only baseline may remain a reference,
but the mechanism control must share our sequence input and effective exposure.

### Existing conversion is a pilot cache, not a full-mixture protocol

`oxe_convert.py` defaults to at most 5000 episodes from a seeded shard-shuffled stream.
It reads only the selected train split (or first available split), makes a new random 10%
episode holdout, saves only the first language instruction, and exports no timestamp/FPS.
Its `fps` local variable contains file paths, not frame rates. Missing file paths can also
leave insufficient original-episode provenance. Existing Bridge uses a 20% holdout instead.

The `|action| / |difference(action)| > 5` rule is only an absolute-command heuristic. It does
not validate units, axis conventions, action/state timing, or whether command differences
equal realized motion. Blindly differencing Euler angles is not a valid rotation treatment.
Padded navigation or human action dimensions are not measured 7-DoF manipulation targets.

## Recommended data contract and use

1. **Freeze the manifest first.** Record source alias, URL, version, licence/access,
   expected files/bytes, checksums where available, train/validation/test provenance,
   original episode/video IDs, embodiment, camera, original/effective FPS and instructions.
   Record transformations and code revision. Keep raw files recoverable until validated.
2. **Validate before marking ready.** Check every expected shard's size/checksum, read
   TFRecord records and decode representative images from every source, and reconcile raw,
   accepted, skipped and filtered counts. Make interrupted conversion distinguishable from
   completion. Current shell conversion uses no `pipefail`, so its final DONE echo is not
   sufficient evidence that each Python process succeeded. `meta.json` is useful evidence,
   but should be backed by array/count checks and a versioned completion manifest.
3. **Separate splits before deriving windows.** Preserve official test data; use stable
   episode/video-group assignments for additional validation. Keep overlapping windows,
   duplicate Bridge encodings, and multiple Ego4D clips from one original video in the same
   split. Retain instruction changes over time. Fit feature/action normalization on training
   data only; do not tune on the final action-probe or rollout test set.
4. **Preserve real temporal and spatial evidence.** Use the intended primary camera per
   source and record any crop/resize geometry. Retain intermediate frames, timestamps,
   padding masks and actual horizon. Resample or duration-condition AC/DC rather than
   treating an equal frame count as equal elapsed time across robot and human datasets.
   If 2-FPS Ego4D clips are too short, explicitly choose a supported horizon/profile mask or
   a higher-FPS re-extraction; the latter is a disclosed deviation from the original recipe.
5. **Use all domains for video/latent pretraining.** Factual reconstruction/flow, nuisance
   SSL, and the separately real-data-trained dynamics prior can consume their relevant
   video/language streams. Recorded robot actions remain outside the action-free training
   input/loss/pair selection, and are separately available for controlled diagnostics and
   downstream robot-policy supervision. Human videos do not supply robot motor commands.
6. **Use compatible destinations for swap.** Begin within embodiment/camera/horizon and
   task-feasibility groups while varying state. This does not assert that two recorded
   actions match. Do not initially swap arbitrary human walking and arm manipulation
   transitions. Apply temporally shared nuisance augmentations; do not erase navigational
   geometry or task-relevant objects as 'background'. Evaluate donor semantics independently.
7. **Match the full data path across comparisons.** Original UniVLA reference, continuous
   reconstruction-only control, and our method need recorded actual exposure and a shared
   split contract. Explicitly select `omni_magic_soup_plus_plus` in both LAM and policy stages:
   the supplied policy config currently selects `omni_magic_soup_plus` without Ego4D. Sample
   streams using the intended probabilities, not sequential per-dataset training. Report
   changed caps, filters, sequence information and domain-balanced alternatives as changes.
8. **Keep policy stages distinct.** Infer latent targets from training transitions for
   language-conditioned latent-policy pretraining; adapt the head for continuous codes.
   Final motor-action learning and rollout evaluation use the selected target-robot task
   datasets and valid action conventions. Do not merge human/navigation dummy or padded
   actions into one global seven-dimensional motor-action regression target.

## Storage and throughput cautions

`myquota` reports scratch 557.7 GB / 100 TB and 443.9k / 1 million files; home is already
19.1 / 25 GB. Use scratch with explicit output paths, not tool defaults under home. The
file-count allowance can become a problem well before the byte allowance: prefer RLDS/video
or other bounded shards over millions of individual JPEG files.

Decoded uint8 RGB at 256x256 costs **196.6 GB per million frames**. A 4x4 grid of 768-D FP16
features costs **24.6 GB per million frames**, whereas a 16x16 grid at the same width costs
**393.2 GB per million frames**. These are layout calculations, not measured final corpus
sizes. The current converter temporarily holds both `frames.raw` and `frames.npy`, doubling
that dataset's RGB space during copying. Budget raw + cache + features + temporary copies;
retain recoverable source data while checking new representations. Stream/mmap sharded data
and separately benchmark reading, image decode, feature extraction and training consumption.

Frozen-feature caching is useful only for the chosen frozen backbone/preprocessing; do not
silently reuse it after changing crop, temporal sampling, backbone or nuisance intervention.
Feature-space perturbations are not automatically equivalent to augmenting input images.

## Immediate next actions (not executed)

Confirm Claude's active downloader host/log and intended full manifest; include the four
omitted sources; resolve full Bridge coverage/equivalence; confirm Ego4D access and curated
clip list. Before scaling conversion, implement the sequence/time/provenance contract and
benchmark representative sources. Then run a small per-domain end-to-end data smoke test,
including source frequencies and shapes, while remaining data transfers continue. No need
to wait for every byte to test the loader, but do not label a pilot as full-mixture training.

## Local evidence

- `claude_survey/results/oxe_mix_sizes.json`
- `latent_free/logs/oxe_fetch_small.out`, `latent_free/logs/oxe_fetch_big.out`
- `latent_free/data/oxe_raw/*/*/{dataset_info.json,FETCH_DONE}` and shard file stats
- `latent_free/data/bridge_raw/dataset_info.json` and shard file stats
- `latent_free/cache/bridge_big/index.npz`
- `latent_free/cache/oxe/ucsd_kitchen_dataset_converted_externally_to_rlds/{meta.json,index.npz,frames.npy}`
- `latent_free/phase1/oxe/{oxe_fetch.py,oxe_convert.py,run_oxe_convert.sh}`
- `latent_free/third_party/UniVLA/prismatic/vla/datasets/{datasets.py,rlds/dataset.py,rlds/traj_transforms.py}`
- `latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/{mixtures.py,configs.py,transforms.py}`
- `latent_free/third_party/UniVLA/prismatic/conf/vla.py`
- `latent_free/third_party/UniVLA/vla-scripts/extern/ego4d_rlds_dataset_builder/ego4d/README.md`
