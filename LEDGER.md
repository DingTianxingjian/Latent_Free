# Latent-free training: master experiment summary

Last updated: 2026-09-06.

This is the project-level, continuously updated summary for `latent_free/`,
`claude_survey/`, and `codex_survey/`. Detailed tables and implementation notes remain in
the source directories, but no experiment or method decision should exist only in a chat.

## Mandatory logging contract

After every experiment finishes (including a failed or invalid run), append a short entry
to this file before reporting the result. Each entry must contain:

1. date, run/configuration, data split, seed count, and budget;
2. the hypothesis or decision being tested;
3. the shortest sufficient numerical result, including the relevant control;
4. one conclusion with one of: **confirmed**, **provisional**, **negative**, **invalid**,
   **retracted**, or **planned**;
5. paths to the raw artifact and code;
6. the resulting method decision or next kill criterion.

Never silently replace an old conclusion. Mark it retracted/invalid and state what new
control changed the interpretation. Plans must not be reported as experimental results.

## Current method in one paragraph

The current **planned** candidate follows preservation → invariance/structure → transport:
four spatial quadrants × temporal **DC + AC1**, a continuous effect latent, explicit decoder
state `F0` and task condition, same-transition nuisance invariance, locally compositional
algebra, and cross-state swaps using a flow decoder with a frozen dynamics-prior regularizer
and code cycle. Single-axis spatial/temporal preservation and oracle-positive transport
have supporting measurements; the hybrid and combined new objective do not. The completed
photo-VICReg, frozen-Wan decoder, and cycle-only screens did not produce the hoped-for
action/transport benefit. Prior + cycle does not by itself identify shared action semantics.
Define temporal-basis-aware algebra and test semantic transport independently. Real-pair
mining remains an optional expansion subject to a precision–coverage gate, not a prerequisite
assumed by the new pair-free proposal. A separate future-derived exogenous residual remains
deferred; any stochastic residual must be controlled when testing donor-code influence.

Detailed current audit: [`latent_free/EXPERIMENT_REVIEW_2026-09-06.md`](latent_free/EXPERIMENT_REVIEW_2026-09-06.md).

## Experiment ledger

### 2026-08-05 — IRLA same-transition SSL positive pairs

- **Test.** Test whether a clean transition and the same transition with a synthetic
  exogenous moving patch can serve as an exact positive pair, with InfoNCE preserving
  transition discrimination while enforcing nuisance invariance.
- **Setup.** Bridge, episode-disjoint train/validation windows, 16,945 training windows,
  continuous 8-slot IDM/FDM, batch 128, 8,000 steps, one seed; matched `lambda_inv=0`
  reconstruction control versus `lambda_inv=1`.
- **Result.** `lambda_inv=0 -> 1` changes clean `xyz R2` from `0.0166 -> 0.1739`, clean
  `cont6 R2` from `0.0096 -> 0.1007`, and moving-patch drop from `40.6% -> 6.3%`.
  The frozen feature-difference control drops `69.0%` under the same patch.
- **Control.** The patched-view self-probe reaches `0.0919`, close to the clean-trained
  probe's patched-view `0.0943`, so the robustness result is not only a clean-probe
  coordinate mismatch. One seed and one synthetic nuisance family remain unresolved.
- **Conclusion — provisional.** A paired-by-construction SSL objective can jointly improve
  action decodability and invariance for one synthetic nuisance on Bridge. This does not
  establish invariance to lighting, true background replacement, or real exogenous motion.
- **Artifacts.** [`claude_survey/results/irla_bridge_inv0.csv`](claude_survey/results/irla_bridge_inv0.csv),
  [`claude_survey/results/irla_bridge_inv1.csv`](claude_survey/results/irla_bridge_inv1.csv),
  [`claude_survey/logs/cs_irla_19694534.out`](claude_survey/logs/cs_irla_19694534.out),
  [`claude_survey/irla.py`](claude_survey/irla.py).
- **Decision.** Reuse exact same-transition positive pairs for the registered photometric
  and static-style intervention, but compare InfoNCE with a negative-free VICReg-style
  objective and require multi-seed, multi-nuisance held-out evaluation.

### 2026-08-05 — IRLA `lambda_inv=3` run-status audit

> **Correction, 2026-09-06:** The failure attribution below is retracted. This was a
> bundled job: lambda=3 completed and printed the CSV values; the later lambda=10 arm
> timed out. See the replacement audit entry dated 2026-09-06.

- **Test.** Test a stronger invariance weight on the same Bridge moving-patch objective.
- **Setup.** Bridge, job `19690719`, intended 10,000-step run with `lambda_inv=3`.
- **Result.** The scheduler log reaches step 9,000 and then records cancellation at the time
  limit; it contains no final evaluation or completion marker.
- **Control.** A file named `irla_bridge_inv3.csv` exists, but its provenance cannot be
  reconciled with the incomplete log, so its numbers are not accepted as a completed run.
- **Conclusion — retracted.** The original claim that the lambda=3 result was incomplete
  misattributed the subsequent lambda=10 timeout. The historical description above is
  superseded by the 2026-09-06 raw-log audit.
- **Artifacts.** [`claude_survey/logs/cs_irla_19690719.out`](claude_survey/logs/cs_irla_19690719.out),
  unverified [`claude_survey/results/irla_bridge_inv3.csv`](claude_survey/results/irla_bridge_inv3.csv).
- **Decision.** Do not use `lambda_inv=3` in comparisons unless it is rerun with a unique
  run identifier and completion marker.

### 2026-07-22 onward — frozen-feature spatial pooling audit

- **Test.** DINOv2 features on episode-disjoint LIBERO-spatial; compare CLS, patch mean,
  4x4 spatial layout projected to 768 dimensions, and the full 4x4 descriptor.
- **Result.** Stride-1 translation R2 for temporal difference: `0.194 / 0.288 / 0.508 /
  0.612`. At stride 4 the full spatial difference reaches `0.839` versus `0.626` for patch
  mean. V-JEPA2 independently loses 51% (stride 1) and 27% (stride 4) of difference
  decodability under mean pooling, close to DINOv2's 53%/25%.
- **Conclusion — confirmed.** Global pooling loses where motion occurred; the effect is
  not specific to DINO or explained by width alone. Opposing local motions can cancel.
- **Artifacts.** [`latent_free/SUMMARY.md`](latent_free/SUMMARY.md), especially Exp 4 and
  Exp 6.
- **Decision.** Never use globally mean-pooled `delta-f` as an information ceiling. Keep
  a spatial effect map until the learned bottleneck.

### 2026-08-18 — Bridge `df` ceiling correction

- **Test.** Compare globally mean-pooled DINO difference with 4x4 spatial difference on
  held-out Bridge.
- **Result.** `df=0.3340`, `df_g4=0.4303`. The latter is still 4x4 pooled; full 16x16 is
  not estimable with the available probe rows.
- **Conclusion — confirmed with scope.** Mean pooling is lossy, but neither number is the
  information ceiling; the probe is capacity limited.
- **Artifacts.** [`claude_survey/results/ceiling.csv`](claude_survey/results/ceiling.csv),
  [`claude_survey/STATUS.md`](claude_survey/STATUS.md).

### 2026-08-20/21 — slot count and redundancy

- **Test.** UniVLA-style learned global query bottlenecks with 1, 2, 4, 8, and 16 slots;
  matched held-out evaluation.
- **Result.** Action R2 is `0.1955 / 0.2600 / 0.3215 (0.3025 repeat) / 0.198--0.229 /
  0.0770`. With four slots, the best single slot carries 72% of all-slot post-VQ
  decodability and 88% pre-VQ; same-code agreement is 2.1x the independent expectation.
- **Conclusion — confirmed.** Four slots are not too few at this budget. Adding slots
  produces noisier/redundant summaries rather than using more information.
- **Artifacts.** [`claude_survey/results/rala2_eval_slots.csv`](claude_survey/results/rala2_eval_slots.csv),
  [`claude_survey/results/slots.csv`](claude_survey/results/slots.csv).
- **Decision.** Keep a compact slot count. Do not treat more slots or a generic diversity
  penalty as the pooling fix.

### 2026-08-20 — SeqDelta-REPA and bottleneck type

- **Test.** Faithful V-JEPA2 clip-level SeqDelta alignment with continuous beta-VAE,
  pre-VQ alignment, and post-VQ alignment.
- **Result.** Continuous latent improves `0.3463 -> 0.4698`. Post-VQ placement changes by
  `+0.0006`; pre-VQ placement decreases `0.5134 -> 0.4250`. A DINO pair target reaches only
  `0.3441`.
- **Conclusion — provisional.** The faithful clip effect anchor shapes a continuous latent,
  but a 16-entry quantizer discards that directional shaping. This does not establish that
  the same anchor transfers to every embodied domain.
- **Artifacts.** [`claude_survey/results/rala2_eval_v.csv`](claude_survey/results/rala2_eval_v.csv).
- **Decision.** Validate effect structure in a continuous/factorized bottleneck before
  returning to discrete action tokens.

### 2026-08-21 — natural cross-scene pair audit

- **Test.** Instruction+phase, DINO/V-JEPA differences, and action-oracle pairing on held-out
  cross-scene transitions.
- **Result.** Action-bin agreement (chance `0.167`): instruction+phase `0.294`, mean DINO
  difference L2 `0.215`, spatial DINO L2 `0.134`, spatial DINO cosine `0.385`, V-JEPA2
  cosine `0.324`, action oracle `0.903`.
- **Conclusion — confirmed.** Existing natural hard-pair rules are too inaccurate for a
  swap objective. The best rule also selects unusually similar state appearance.
- **Artifacts.** [`claude_survey/results/pairq.csv`](claude_survey/results/pairq.csv).
- **Decision.** Use paired-by-construction interventions first; real-pair training requires
  a measured precision--coverage gate.

### 2026-08-21 — transport decomposition

- **Test.** Action-matched same-scene and cross-scene code transplantation with matched
  action-pair quality.
- **Result.** For `s2_base`, cross-scene transport is `0.4719` and same-scene `0.5694`.
  Scene change explains about 18% of the deficit from perfect transport; about 82% remains
  inside one scene. Released UniVLA weights show a much larger cross-scene loss.
- **Conclusion — confirmed.** Cross-context non-identifiability is real but is not the only,
  or largest, failure on Bridge. The representation's effect is weak even within a scene.
- **Artifacts.** [`claude_survey/results/transport2.csv`](claude_survey/results/transport2.csv).

### 2026-08-26 — quick temporal DC/AC audit

- **Test.** Three episode-disjoint repeats, each with 12k train/5k validation windows.
  Decompose four **spatially mean-pooled** feature deltas into temporal DC and AC
  components. This test intentionally removes the 4x4 spatial layout to isolate temporal
  averaging; it is not intended to recover the spatial-pooling loss measured above.
- **Setup.** The two prediction targets are different: `net = sum_t action_t`, whereas
  `profile = flatten(action_t - mean_t(action))`. The profile target explicitly removes the
  large window-average/net component and asks only for within-window deviations. Its R2 is
  therefore not numerically comparable to the earlier single-step/net translation R2 values
  (`0.5--0.9`) measured with spatial layout preserved.
- **Result.** Adding aligned AC1 to DC improves action-profile R2 by `+0.0391` over DC and
  `+0.0498` over equal-width shuffled AC1. Aligned-vs-shuffled net-action gain is `+0.0496`.
  AC2 is weak and AC3 is at chance. Per repeat, DC profile R2 is `0.0655--0.0699` and
  DC+AC1 is `0.1042--0.1099`; the paired absolute gain is `0.0387--0.0400`, or roughly
  56--61% of the DC baseline R2. This relative percentage describes the probe score, not
  a percentage of action information.
- **Control.** Equal-width shuffled AC1 scores `0.0541--0.0597`, below both aligned AC1 and
  DC alone. All three repeats use episode-disjoint train/validation data, but resample
  windows from the same episode split; they are not three independent dataset splits, and
  no episode-level bootstrap confidence interval has yet been computed.
- **Conclusion — confirmed.** In this frozen-feature, linear-probe instrument, AC1 contains
  a reproducible nonzero action-profile signal that temporal DC pooling discards and extra
  shuffled width does not recover. The practical magnitude for a learned latent, pair
  mining, transport, or downstream policy remains **provisional**.
- **Artifacts.** [`codex_survey/QUICK_DCT.md`](codex_survey/QUICK_DCT.md),
  [`codex_survey/results/quick_dct_effects.csv`](codex_survey/results/quick_dct_effects.csv).
- **Decision.** The candidate real-pair descriptor is spatially factorized `DC + AC1`.
  This experiment supports the descriptor substrate; it has **not yet** shown that DC+AC1
  produces sufficiently precise real pairs. Before claiming a useful improvement, run
  held-out-episode bootstrap/repeated split controls and the trained hybrid/downstream tests.

### 2026-08-27/29 — complete pooling x reconstruction matrix (30/30)

- **Test.** Five pooling types x endpoint/sequence reconstruction x three seeds, Bridge-big,
  continuous 32-dimensional bottleneck. As in the quick audit, `net` predicts the summed
  four-step action and `profile` predicts the mean-removed per-step residual trajectory;
  values from those two columns must not be compared as the same target.

| pooling | endpoint net R2 | endpoint profile R2 | sequence net R2 | sequence profile R2 | endpoint/sequence cross-slot R2 |
|---|---:|---:|---:|---:|---:|
| global | 0.200 | 0.080 | 0.225 | 0.089 | 0.959 / 0.994 |
| spatial | **0.257** | 0.078 | **0.272** | 0.082 | 0.435 / 0.536 |
| temporal | 0.239 | **0.109** | 0.220 | **0.100** | 0.076 / 0.062 |
| DCT | 0.217 | 0.102 | 0.216 | 0.099 | 0.009 / 0.011 |
| learned attention | 0.208 | 0.080 | 0.253 | 0.092 | 0.954 / 0.990 |

- **Result.** Spatial beats global net-action in all three seeds (`+0.0566` endpoint,
  `+0.0465` sequence). Temporal/DCT improves profile by about `+0.014` but loses roughly
  `0.03--0.05` net R2. Learned attention largely collapses to global pooling. Sequence
  reconstruction does not remove the spatial/temporal trade-off; the registered target x
  pooling interaction is negative.
- **Conclusion — confirmed.** Spatial and slow temporal structure contain complementary
  action information. None of the tested single-axis poolers preserves both.
- **Artifacts.** [`codex_survey/results/`](codex_survey/results/),
  [`codex_survey/README.md`](codex_survey/README.md).
- **Decision.** Next architecture must explicitly combine spatial quadrants with DC+AC1;
  neither raw spatial pooling nor pure temporal/DCT pooling is sufficient. Initial success
  gate: net R2 about `>=0.27` and profile R2 about `>=0.10` in the same model.

### 2026-08-28 — Language/TI/TC conditioning

- **Test.** No language, language in encoder, and language in encoder+decoder; held-out
  code prediction with majority controls.
- **Result.** TC language-only improvement over majority is `+0.0067 / +0.0005 / -0.0059`.
  Adding language to `f_t` changes accuracy by approximately `-0.0050 / +0.0029 / +0.0036`.
  TC uses 15--16/16 codes, while TI uses 1/1 effective code in these runs.
- **Conclusion — negative.** Concatenating a T5 embedding does not make TC task-centric.
  Code usage is not the explanation for the TC result.
- **Artifacts.** [`claude_survey/results/latacc_lang.csv`](claude_survey/results/latacc_lang.csv).
- **Decision.** Task should explicitly condition policy/decoder selection of a shared effect;
  do not require the physical effect code itself to classify task identity.

### 2026-08-30/09-01 — operational transport objective test

- **Test.** Train matched transport arms with actual-action oracle positives, oracle-positive
  ranking, random/control pairing, cycle, and the best then-available pseudo pairs.

| arm | xs action-match transport | matched cross-scene | matched same-scene | gain over base |
|---|---:|---:|---:|---:|
| base | 0.5070 | 0.4719 | 0.5694 | 0.0000 |
| oracle-positive | **0.6558** | 0.6052 | 0.6561 | **+0.1488** |
| oracle | **0.6628** | **0.6121** | **0.6688** | **+0.1558** |
| random/control | 0.5114 | 0.4728 | 0.5575 | +0.0044 |
| current pseudo pairs | 0.5080 | 0.4702 | 0.5465 | +0.0010 |

- **Conclusion — confirmed for the objective, negative for current pseudo pairs.** A correct
  off-diagonal transport signal materially improves transport, while the control does not.
  The present pseudo-pair rule yields no benefit. Therefore transport has not failed as a
  method; pair precision/coverage is the binding problem.
- **Artifacts.** [`claude_survey/results/transport_operational_summary.md`](claude_survey/results/transport_operational_summary.md),
  `claude_survey/results/transport_s2_tr*.csv`.
- **Decision.** Continue the transport line, but do not train from low-precision hard pairs.
  Improve the descriptor and/or create exact intervention pairs.

### 2026-09-01 — large-pool DINO pseudo-pair audit

- **Test.** DINO 4x4 descriptor projected to 512 dimensions, cross-scene mutual nearest
  neighbours and top-50% margin filtering over 134,464 training transitions.
- **Result.** One-way pairing gives bin agreement `0.449`. Mutual pairs give `0.553` over
  9,856 rows. Selected top-margin pairs give `0.560` bin agreement and `0.631` action cosine
  over 4,928 rows, only `3.66%` coverage. The empirical cross-scene null is `0.185` bin
  agreement and `0.0066` cosine.
- **Conclusion — provisional/negative for training.** Mutual+margin filtering clearly
  enriches action similarity beyond chance, but about 44% bin disagreement at 3.7% coverage
  is not accurate enough for hard swap training.
- **Artifacts.** [`claude_survey/results/transport_dino_audit.json`](claude_survey/results/transport_dino_audit.json).
- **Decision.** Use this as the baseline that spatial DC+AC1 and multi-representation
  consensus must beat on a precision--coverage curve.

### 2026-09-01 — DINO pseudo-pair transport training

- **Test.** Determine whether the best large-pool DINO mutual+margin pairs provide a useful
  off-diagonal transport training signal.
- **Setup.** Bridge-big, one 3,000-step arm (`s2_trdino`), 4,928 mutual pairs from 134,464
  transitions (`3.66%` coverage); donor action-bin agreement `0.5601`, xyz cosine `0.6305`.
- **Result.** `xs_actmatch` transport is `0.4678`, below base `0.5070` by `-0.0392` and far
  below oracle-positive `0.6558`. Matched cross-scene transport is `0.4483` versus base
  `0.4719`; matched same-scene is `0.5341` versus base `0.5694`. The random/control arm is
  `0.5114`, so ordinary training variation does not explain an improvement.
- **Control.** Verified action oracle and action-far conditions remain responsive:
  `xs_actmatch` pairing agreement is `0.903`, while `xs_actfar` gives transport `-0.0634`.
- **Conclusion — negative.** DINO pairs at 56% bin agreement are not merely insufficient;
  this one-seed arm degrades transport. The pairing-quality gate must be substantially
  stricter before real-pair training is retried.
- **Artifacts.** [`claude_survey/results/transport_s2_trdino.csv`](claude_survey/results/transport_s2_trdino.csv),
  [`claude_survey/logs/tr_dino_pos_20271889.out`](claude_survey/logs/tr_dino_pos_20271889.out).
- **Decision.** Do not train additional hard-swap arms from this DINO pair set. Test
  paired-by-construction intervention and spatial DC+AC1 precision--coverage first.

### 2026-09-01 — SAM3 background-intervention gate

- **Test.** Check whether generic SAM3 masks can preserve the action-carrying foreground
  while replacing the background with a donor scene.
- **Setup.** Eight held-out Bridge scene pairs, stride 4, generic SAM3 checkpoint, prompts
  for robot arm/gripper/held object, 7-pixel dilation and feathered compositing.
- **Result.** Mean foreground union area `0.146`, background replaced `0.854`, motion-support
  recall `0.753`, and temporal mask IoU only `0.344`. One of eight future masks is empty;
  gripper and held-object prompts are usually empty. Visual inspection shows inconsistent
  endpoint masks and donor images that retain their own robot, producing two-robot or
  implausible counterfactuals.
- **Control.** The gate records endpoint RGB-difference support and separate prompt masks;
  high replacement area alone therefore cannot hide missed action-support pixels.
- **Conclusion — negative.** The current SAM3+raw-donor compositing pipeline is not a safe
  action-preserving intervention. This is a failure of the present prompting/donor pipeline,
  not evidence that offline SAM3 masks are unusable in principle.
- **Artifacts.** [`claude_survey/results/sam3_bg_gate_20272597/summary.json`](claude_survey/results/sam3_bg_gate_20272597/summary.json),
  [`claude_survey/results/sam3_bg_gate_20272597/metrics.csv`](claude_survey/results/sam3_bg_gate_20272597/metrics.csv),
  sample mosaics in `claude_survey/results/sam3_bg_gate_20272597/`.
- **Decision.** Do not use these composites for training. Remove/inpaint donor robots, add
  temporal mask propagation/union and motion-connected object support, then repeat a larger
  gate before any intervention loss is trained.

## Data: UniVLA's pretraining mixture, reproduced as far as it is reproducible

### 2026-09-05/06 — OXE mixture fetch + conversion (in progress; corrected 09-06 after Codex's audit)

- **Goal.** Train the swap-transport method and a matched-budget UniVLA stage-2 baseline on
  the data UniVLA used, so the comparison is on their distribution, not only on Bridge (where
  the scene axis is 18% of the transport deficit, C8).
- **What UniVLA used.** `omni_magic_soup_plus_plus`: 26 OXE manipulation sources + 3 Berkeley
  GNM navigation sources + one curated Ego4D collection as 4 split entries; LAM at 256x256 on
  each source's `primary` camera. **Sampling** uses `balance_weights=True`
  (`rlds/dataset.py`): p_i ~ w_i x num_transitions_i, so raw weights alone say nothing about a
  source's share. **Windows** come from `traj_transforms.chunk_act_obs`, which gathers only the
  first and last observation of each window -- their LAM sees two frames per sample, with a
  per-source interval (Ego4D: 2 frames at 2 FPS).
- **What is reachable.** 28 of 33 entries are in the public bucket `gs://gresearch/robotics`
  (3.48 TB; per-source sizes in `claude_survey/results/oxe_mix_sizes.json`).
- **Deviations.** (1) **Ego4D is not fetched**: not in the bucket; licence (48 h credential
  turnaround, user/institution action) + UniVLA's curated FHO-clip RLDS builder. The one real
  gap; its share of their samples is unknown until per-source transition counts exist.
  (2) **Bridge** is our RAIL `bridge_dataset` copy: 97 train tfrecord shards of 1024 (11.1 GB
  of 132.5 GB), 5,000 episodes in `bridge_big`, 20% holdout; not verified equivalent to the
  `bridge_oxe` re-encode. (3) **Per-source cap of 5,000 whole episodes** from a seeded
  shard-shuffled stream, 10% episode holdout; the cap binds on fractal, bc_z, kuka,
  language_table, fmb, furniture_bench. Sampling weights are applied at training time.
- **Retracted (09-06).** The 09-05 entry excluded kuka / language_table / fmb /
  furniture_bench as "3% of their samples" and called Ego4D "9%". Both numbers normalised raw
  w and ignored `balance_weights`; they are wrong and withdrawn. The four sources are now in
  the fetch manifest (last). The same entry said the large sources were "fetching": they were
  not -- the chained launcher's `pgrep` matched its own command line and never exited. The
  fetch was started directly on 09-06 (`logs/oxe_fetch_big.out`).
- **Conversion.** `latent_free/phase1/oxe/oxe_convert.py` -> `cache/oxe/<name>/{frames.npy,
  index.npz, scene_meta.npz, meta.json}`, the layout every existing loader reads; **all frames
  of each episode are kept**, so T>=4 windows are available without a new loader. Actions are
  standardised with UniVLA's `OXE_STANDARDIZATION_TRANSFORMS` (vendored verbatim). Pilot-cache
  limits Codex listed and which stand: train split only, first instruction only, no
  timestamp/FPS field, per-source holdout drawn anew.
- **Caveat on actions.** Several sources' "action" is an **absolute pose** that UniVLA also
  labels `EEF_POS` (ucsd_kitchen: mm/degrees). `meta.json` records |a| / |a[t+1]-a[t]| (12.7
  ucsd_kitchen, 1.47 Bridge) and `action_looks_absolute` (threshold 5). This is a heuristic for
  the command's form only: it does not validate units, axis conventions or whether commands
  equal realised motion, and differencing Euler angles is not a rotation treatment. Probes on
  flagged sources use finite differences on xyz only until per-source conventions are checked.
- **Status 09-06 (evening).** Fetch complete: 28/28 sources, 3.2 TB, 0 failed shards.
  Converted: 24 sources + bridge_big (login node for the small/medium ones; job `20435188` for
  the large ones at ~2,000 f/s vs ~150 f/s on the throttled login node). Failed in that job and
  resubmitted as `20441462`: iamlab_cmu and furniture_bench (UniVLA's transforms call
  `tensorflow_graphics` quaternion->Euler; vendored as `_tfg_shim.py` with an import-time
  self-check), kuka (tfds decoding of a mistyped non-image field; converter now uses
  `SkipDecoding` and decodes only the primary camera, as the OXE loaders do -- the job
  byte-compares a reconverted ucsd_kitchen against the existing cache before proceeding), fmb
  (never attempted: its fetch finished after the job's source list was built).
- **Action-kind decisions so far.** description: austin_buds, bc_z, language_table (delta),
  ucsd_kitchen (absolute); override: jaco_play, 3x GNM (delta); statistic: cmu_stretch,
  dlr_edan, dobbe and others (delta), **toto (absolute, no description -- to verify against the
  toto card before any probe on it is read)**.
- **Status 09-07 00:00 — conversion complete.** 28/28 public-bucket sources + bridge_big in
  `cache/oxe/` (~1.5 TB); the last four via `20448435` / `20456520` (kuka 74k frames; fmb 670k,
  action kind *unknown*; iamlab 146k, *absolute* by statistic; furniture_bench 3.86M frames /
  759 GB, delta by description). Ego4D remains the only gap. Sources whose action kind is
  `unknown`/statistic-only (fmb, toto, iamlab, cmu_stretch, dlr_edan, dobbe) need their
  dataset cards checked before per-source probe values on them are quoted.
- **Audit.** `latent_free/UNIVLA_DATASET_AUDIT_2026-09-06.md` (Codex, read-only).
- **Repo health (09-07).** `latent_free/.git` has **110 missing objects** (`git fsck`; last commit
  predates the Aug 23 incident) and cannot build a tree, so nothing under `latent_free/` has been
  committable since. The OXE pipeline is mirrored and versioned at `claude_survey/oxe_pipeline/`.
  Repairing or re-initialising `latent_free`'s history is a user decision.

## Registered method: Swap-Transport LAM (pre-registration, 2026-09-06)

### 2026-09-06 — five-arm experiment, registered before any arm has run

- **Question (P3).** Can cross-state transport be trained without paired counterfactual targets?
  Known: the transport objective works with correct off-diagonal pairs (oracle +0.156 transport,
  +0.086 `z_q` R2) and hurts with 56%-agreement pairs (-0.039); no label-free pairing rule
  clears the threshold (C15). The candidate replaces pairs with two constraints on the swapped
  transition `D(F0_j, c_j, z_i, u_j)`: realism (distribution-matched to real transitions, MMD
  after a fixed random projection) and cycle (re-encodes to `z_i`).
- **Model.** Frozen DINOv2 ViT-B/14 -> 4x4 feature grid; effect cells = 4 spatial quadrants x
  {DC = summed displacement, AC1} of the K=4 feature-difference sequence (T=5 frames, stride 4:
  the same 16-frame window UniVLA's loader spans while returning only its endpoints); Gaussian
  `z` (8 cells x 32), Gaussian `u` (8-d, beta_u = 0.1, never swapped); decoder = 4-layer
  transformer over [16 state tokens, 8 z tokens, u, pooled-T5 c] predicting the 4x4 x K
  difference sequence. `claude_survey/swaplam.py`.
- **Arms** (same seed, sources, steps=10k, batch 32; realism/cycle ramp in after 1k steps):
  `rec` (reconstruction + KL only; floor), `control` (+ MSE transport to a RANDOM donor; the wrong
  objective, mirrors `s2_trctl`), `swap` (+ MMD realism + cycle), `swapalg` (+ ALAM-style
  composition/reversal on the DC cells of half-clips), `oracle` (+ MSE transport to the in-batch
  NEAREST-action donor; uses labels; ceiling).
- **Readouts** (held-out episodes; probes are the only place labels enter): `nmse_net`,
  `nmse_prof` (linear-probe NMSE, the LAOM/MaskLAM number), `patch_drop_pct`,
  **`transport_action_r2`** (decode `z_i` on a random `j`, re-encode, probe: does the swapped
  transition carry the DONOR's action -- the pairing-free operational transport number),
  `swap_sens` (empty-z detector), cycle / MMD / algebra errors, KL.
- **Pre-registered reading.**
  1. `oracle` - `control` on `transport_action_r2` is the room the objective has; if it is
     within the seed floor, the readout has no power and nothing below is read.
  2. **Method holds** if `swap` beats `control` on `transport_action_r2` by more than the
     3-seed floor (to be measured on `rec`) AND `nmse_net` does not worsen beyond that floor
     AND `swap_sens` is not below `rec`'s. Realism/cycle converging while `transport_action_r2`
     stays at `control` = the identifiability gap (risk #1) is real: the method is dead as
     specified, and the result is reported as such.
  3. `swapalg` is NOT a transport ablation: read its algebra errors first; algebra down with
     transport unchanged is a valid finding ("structured latent geometry alone is
     insufficient for semantic transport"), not a failed run.
  4. Nothing here is a claim about UniVLA until the matched-budget stage-2 retrain on the same
     caches exists; the released checkpoint is a reference row, not the baseline.
- **Data.** Every complete cache at job start (the manifest is written per run:
  `results/swaplam_<tag>_manifest.json`). **Sampling corrected 09-06 before any job ran**: source
  probabilities follow UniVLA's rule on FULL dataset sizes (episodes_available x mean converted
  episode length), spread uniformly over the clips we hold; on the cache counts alone the
  5000-episode cap would have given dobbe (5k/5.2k eps, 219 frames each, w=0.2) more samples
  than Bridge (5k/53k eps, w=1.0). Consequence to keep in view: capped sources are revisited
  more often per epoch than uncapped ones (Bridge ~8x dobbe), i.e. lower within-source diversity
  at the same source frequency. Both distributions are in every manifest.
- **Known limits at registration.** MMD on batch 32 has little power even after projection
  (upgrade path: flow-matching decoder + dynamics-prior score regularisation); realism cannot
  reject transported camera/exogenous motion (that is the `pairpatch` arm's job, not run in this
  set); `u` vs `z` capacity is set by `beta_u` without a principled rule; single seed per arm in
  the first run -- directions are read only after the 3-seed floor exists.
- **Artifacts.** `claude_survey/swaplam.py`, `claude_survey/run_swaplam.sh`,
  `claude_survey/oxe_mix.py`; results to `claude_survey/results/swaplam_<arm>.json`.
- **Jobs (ai partition).** `20435188` converted the large sources (done; 4 failures resubmitted
  as `20441462`); `20436140` (five arms) started 09-06 ~10:30 and was **cancelled at 0% GPU**:
  its 14 loader workers sat in I/O wait at 10-13 GB RSS each -- random mmap touches into 25
  files (~600 GB) on NFS trigger multi-MB readahead per frame; fixed with positional reads
  (`oxe_mix.py`, byte-identical), resubmitted as **`20442294`**; `20436363` = `s2mix_base`,
  UniVLA stage 2 (`rala2.py --data-mix all --lang none`, T=2, 10k steps, batch 16) on the
  same caches with the same sampler -- the matched-data baseline row that item 4 requires.
  Each job records the source manifest it actually trained on.

### 2026-09-06 — matched-data UniVLA stage-2 baseline trained on the mixture (`s2mix_base`)

- **Run.** `rala2.py --data-mix all --lang none`, T=2, stride 4, 10k steps, batch 16, seed 0,
  25 sources (manifest `claude_survey/results/s2mix_base_manifest.json`), 0.11 s/step with the
  positional-read loader. Training mse 0.147 at 10k (Bridge-only `s2_base`: 0.120; the mixture is
  harder, as expected). Job `20436363`. No readout yet.
- **How baseline and method will be compared (`eval_mix.py`, job `20444100`).** One val
  sampler, one probe, one split rule, three checkpoints: `s2mix_base` (matched data),
  `s2_base` (same architecture, Bridge only -- isolates what the mixture did to the code),
  `official_stage2` (released weights, reference row). Each T=5 stride-4 val clip is encoded as
  UniVLA's four consecutive pairs. **Like-for-like target = `nmse_seg1`**: one UniVLA code vs
  the method's z, both probed for the first 4-step displacement. `nmse_net` (16-step) uses the
  four UniVLA codes concatenated, i.e. 4x the bits of one code -- reported, but not the headline.
  `transport_action_r2` for UniVLA uses its own decoder and a feature-level re-encode
  (`cycle.py` paths). Both `z_q` and `emb` rows.
- **Conclusion — planned.** Nothing is read until both jobs return and the `rec` arm's seed
  floor exists.

### 2026-09-06 — first mixture readout was invalid: pooled probe over heterogeneous action units

- **What happened.** `eval_mix` job `20444100` returned NMSE > 1 for `s2mix_base`, `s2_base` and
  `official_stage2` -- and **for the frozen-feature reference too** (1.117; ~0.64 on Bridge).
  A metric that fails on the reference is a fault in the instrument, not the models.
- **Cause.** One ridge fitted over clips pooled across 25 sources whose action units differ by
  orders of magnitude (Bridge metres ~0.03/segment, ucsd_kitchen millimetres in the hundreds,
  GNM waypoint offsets, absolute-pose differences): the fit is dominated by the largest-unit
  source and predicts nothing for the rest. This is the failure Codex's audit named in
  advance ("do not merge ... into one global seven-dimensional regression target").
- **Fix (`claude_survey/probes.py`).** Probes are fitted and read WITHIN each source
  (episode-disjoint inside the source); the report carries the per-source table plus
  clip-weighted and unweighted means over sources with >= 80 held-out clips. The same fitted
  probe reads the moving-patch view and the re-encoded swap. Validation clips are drawn
  stratified, 400 per source, instead of by the UniVLA weights (which give ucsd_kitchen 0.1%).
  Synthetic check on three unit scales: pooled R2 0.15 vs per-source 0.996.
- **Conclusion — invalid (instrument), retracted before any reading.** The three pooled result
  files are renamed `eval_mix_POOLED_INVALID_*.json` and kept. Both readout paths (swaplam,
  eval_mix) now use `probes.py`; eval resubmitted as `20444721`; the five-arm job `20442294`
  (still pending) picks up the same code.

### 2026-09-06 — swap-transport v1: partial readouts, an instrument finding, and the v1.1 fix

- **Status.** Job `20442294`: rec, control, swap, swapalg, oracle all trained (10k steps, 0.11
  s/step, batch 32, 25 sources, seed 0). rec/control readouts crashed on a module-name collision
  (`probes.py` shadowed by `latent_free/probes.py`; renamed `mixprobes.py`) and are re-read from
  their checkpoints in `20449049` together with all other arms, now with per-source swap
  sensitivity and a raw-array dump.
- **First valid numbers (single seed; NOT read for direction until rec/control/oracle exist).**
  per-source probes, 400 val clips/source, 23 sources averaged:

| arm | clean R2 net (z) | frozen dF ref | seg1 | patch drop | transport_action_r2 | algebra err |
|---|---:|---:|---:|---:|---:|---:|
| swap | 0.322 | 0.299 | 0.286 | 21.6% | 0.253 | -- |
| swapalg | 0.336 | 0.299 | 0.299 | 20.1% | 0.233 | 0.0016 / 0.0018 |

  The 256-d continuous z reads slightly above the 12,288-d frozen difference on the same
  clips. Algebra is satisfied almost exactly and transport did not move -- the pre-registered
  "geometry alone is insufficient" reading, pending the other arms.
- **Unexplained, structural.** Transport is ~0.00 (to three decimals) on the same sources in
  both arms -- berkeley_autolab_ur5 (clean 0.71), nyu_franka (0.60), fanuc, stanford_hydra,
  gnm_recon -- and 0.6-0.9 on ucsd_kitchen, fractal, austin_buds, austin_sirius. Values that
  close to zero look like a constant prediction (the decoder ignoring z on those embodiments).
  Tested and rejected: language conditioning (every source has a T5 cache; instruction count
  does not track it: nyu 1 instruction -> 0.00, austin_buds 1 -> 0.74). Per-source swap
  sensitivity and the array dump in `20449049` are there to settle it before anything is read.
- **Instrument finding -> v1.1.** The oracle arm's transport loss (0.95) is no better than the
  control's (0.94): a batch of 32 drawn by mixture weight spans ~15 sources, so the
  nearest-xyz donor in mixed units is rarely a same-embodiment partner, the realism term
  compares mixed bags, and control differs from swap by embodiment as well as pairing. v1.1 =
  **source-homogeneous batches** (`oxe_mix.source_batches`: source per batch by p_univla, 32
  clips within it; marginal frequencies verified unchanged), the audit's rec. #6. Five-arm job
  `20450559` (tags `*_src`) queued behind the v1 readouts. v1 stays on record as the
  mixed-batch condition.

### 2026-09-06 — swap-transport v1 complete (mixed batches): architecture transports, the swap objective does not add to it

- **Five arms, one seed, 25 sources, per-source probes on 9,753 stratified val clips (jobs
  `20442294` + eval-only `20449049`).** Baselines on the same instrument from `eval_mix` job
  `20444721`.

| row | clean R2 net (z) | seg1 | patch drop | transport_action_r2 | KL(z) | cycle | swap sens |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen 12,288-d dF (reference) | 0.299 | -- | -- | -- | -- | -- | -- |
| UniVLA stage 2, released | 0.342* | 0.252 | **14%** | 0.113 | -- | -- | 0.54 |
| UniVLA stage 2, our 10k retrain on the mixture (`s2mix_base`) | 0.224* | 0.143 | 22% | 0.042 | -- | -- | 0.25 |
| **rec** (our architecture, reconstruction only) | 0.353 | 0.308 | 22% | **0.236** | 31 | 0.27 | 0.35 |
| control (+ MSE transport to a random donor) | 0.360 | 0.302 | 20% | **0.027** | 26 | 0.63 | 0.17 |
| swap (+ MMD realism + cycle) | 0.322 | 0.286 | 22% | 0.253 | **79** | 0.008 | 0.49 |
| swapalg (+ algebra) | 0.336 | 0.299 | 20% | 0.233 | 59 | 0.004 | 0.77 |
| oracle (+ MSE transport to nearest-action donor) | 0.363 | 0.313 | 22% | 0.052 | 25 | 0.67 | 0.15 |

  *UniVLA's 16-step number concatenates four codes (4x the bits); its `emb` rows are 0.339 /
  0.336. UniVLA transport is on the first pair with its own decoder.
- **Readings (pre-registration items).**
  1. Oracle - control = 0.025: **the oracle was not a ceiling in v1** -- mixed-source batches
     made its nearest-xyz donor effectively random (its training transport loss equalled
     control's). Item 1 is not evaluable in v1. Power of the readout is shown instead by
     rec - control = +0.21: the wrong objective destroys transport while leaving the clean
     probe untouched (the Bridge `s2_trctl` signature).
  2. swap - rec = +0.018, swapalg - rec = -0.003: **the swap objective adds nothing measurable
     over the architecture** (single seed; the `z_q` floor on Bridge was 0.03 sd). Realism and
     cycle both converged (MMD ~0 in training, cycle 0.008), the decoder became more
     z-sensitive, z carries 2.5x the information (79 vs 31 nats) and none of it is action
     (clean R2 0.353 -> 0.322). This is the pre-registered identifiability failure (risk #1):
     realism + cycle were satisfied by non-action content. For the v1 configuration the method
     as specified does not work; recorded as such.
  3. Algebra: errors 0.0016 / 0.0018, transport unchanged -> "structured latent geometry alone
     is insufficient for semantic transport" holds.
  4. **What does hold, and is the reportable result of this round:** the architecture alone
     (fixed quadrant x {DC,AC1} basis, continuous cells, frozen DINO, decoder given state and
     task) gives transport 0.236 = 5.6x our matched UniVLA retrain and 2.1x the released
     weights, with a like-for-like probe (seg1 0.308 vs 0.143 / 0.252) above both. The released
     weights remain clearly more robust to the moving patch (14% vs 22%).
- **Per-source anomaly, dissected.** Transport is ~0.00 in EVERY arm (incl. rec) on
  nyu_franka, stanford_hydra, fanuc, gnm_recon, and 0.03 on UR5 (clean 0.73), while ucsd_kitchen
  reads 0.79. On the failing sources the re-encoded swap predicts neither the donor's nor the
  recipient's action (UR5: -1.40 vs recipient), with full variance -- action-scrambled, not
  constant. Rejected: language conditioning (all sources have caches; instruction count does not
  track it); off-manifold re-encodings (spread and NN distance to the clean cloud are the same in
  failing and passing sources). Remaining hypothesis = state-relative semantics: the same code
  means a different motion on a different state where states vary (UR5, franka play) and the
  same one where they do not (ucsd: one kitchen). Test queued: per-pair transport error vs
  donor-recipient state distance (F0 dumped in `20451518`).
- **Artifacts.** `claude_survey/results/swaplam_{rec,control,swap,swapalg,oracle}.json` (+
  `_readout_arrays.npz`), `results/eval_mix_mix_*.json`, logs `cs_swaplam_20442294`,
  `cs_swaplam_eval_20449049`, `cs_eval_mix_20444721`.
- **Decision.** v1.1 (source-homogeneous batches, job `20450559`) is the proper test of the
  swap objective and of the oracle ceiling; it does not change the architecture result. If v1.1
  also shows swap = rec, the swap objective is dropped from the method and the paper's method
  claim is the architecture + factorisation, with transport-vs-state-distance as the analysis.

### 2026-09-06 (late) — the v1 "zero-transport sources" were an instrument artifact; all transport numbers above are superseded

- **Finding.** Dissecting the rec arm's dumped arrays: on every "zero" source ONE held-out row
  carried 60-90% of the transport target's variance (top-1 share of SST 0.74-0.89, kurtosis
  130-190), while within-source net displacements are not heavy-tailed (max/median 2-5x). The
  rows were **donors from a different source**: ~3% of readout pairs straddle two sources'
  stratified blocks, and a donor with foreign action units (ucsd millimetres, austin_sailor
  ~10-20) alone sets a source's R2 to ~0 when it lands in the test third -- which is why the
  per-source numbers moved between re-pairings of the same checkpoint. Rejected on the way:
  language conditioning, off-manifold re-encodings, state-relative semantics (transport error is
  uncorrelated with donor-recipient state distance within source, |r| <= 0.31).
- **Fix.** Transport rows use same-source donors only (`mixprobes` masked extras; swaplam and
  eval_mix). Cross-embodiment transport is a separate question and is NOT what
  `transport_action_r2` measures. Offline re-read of the two arms with dumps:

| v1 | transport, same-source donors | old (mixed donors, INVALID) | clean R2 | swap sens |
|---|---:|---:|---:|---:|
| rec | **0.441** | 0.274 | 0.336 | 0.35 |
| control | **0.106** | 0.027 | 0.360 | 0.17 |

  Former zero sources now 0.36-0.66, tracking their clean R2 (fanuc 0.47, nyu_franka 0.66,
  roboturk 0.51, dlr_edan 0.50); dobbe (wrist camera) and toto stay ~0 with clean ~0.
- **Status of the readings in the previous entry.** Items 2 and 3 (swap adds nothing over rec;
  algebra insufficient) are re-read when the corrected re-read of swap/swapalg/oracle lands
  (`20454376`); item 4 (architecture >> UniVLA) is re-read against the corrected UniVLA rows
  (`20454377`). The rec-vs-control gap grew (0.34), so the instrument has more power than
  reported. The old `transport_action_r2` values are kept in the result files under
  `..._v1_mixed_donors_INVALID`.

### 2026-09-07 — swap-transport v1 + v1.1 complete on the corrected instrument: the architecture transports; the swap objective does not add to it

- **Instrument (final form).** Per-source ridge probes, 400 stratified held-out clips per source,
  transport rows restricted to same-source donors (`mixprobes.py`). 28 sources in the v1.1
  reads (kuka, fmb added), 25-26 in v1. Single seed per arm; rec seeds 1/2 queued for the floor.

| row | seg1 probe (like-for-like) | net probe | transport_action_r2 | patch drop | swap sens |
|---|---:|---:|---:|---:|---:|
| frozen 12,288-d dF (reference) | -- | 0.283 | -- | -- | -- |
| UniVLA released (z_q / emb) | 0.242 / 0.276 | 0.322 / 0.328 | 0.240 / 0.210 | **11% / 14%** | 0.55 |
| UniVLA stage 2, 10k retrain on the mixture | 0.134 / 0.246 | 0.202 / 0.313 | 0.072 / -0.025 | 16% / 28% | 0.25 |
| UniVLA arch, Bridge only | 0.117 / 0.200 | 0.149 / 0.276 | 0.094 / -0.064 | 42% / 99% | 0.15 |
| **rec** v1 / v1.1 | 0.291 / 0.296 | 0.336 / 0.340 | **0.441 / 0.452** | 27% / 23% | 0.35 / 0.36 |
| control v1 / v1.1 | 0.300 / 0.292 | 0.348 / 0.340 | 0.094 / 0.034 | 26% / 25% | 0.17 / 0.11 |
| swap v1 / v1.1 | 0.288 / 0.277 | 0.313 / 0.284 | 0.510 / 0.476 | 33% / 26% | 0.50 / 0.61 |
| swapalg v1 / v1.1 | 0.286 / 0.282 | 0.324 / 0.304 | 0.470 / 0.479 | 22% / 30% | 0.78 / 0.92 |
| oracle v1 / v1.1 | 0.303 / 0.303 | 0.348 / 0.341 | 0.088 / 0.155 | 22% / 26% | 0.15 / 0.12 |

- **Reading 1 -- the reportable result.** A reconstruction-only LAM with the fixed quadrant x
  {DC, AC1} basis, continuous cells, frozen DINO and a state/task-conditioned decoder transports
  at 0.44-0.45 on the pairing-free readout: 1.9x the released UniVLA weights (0.24; ~100x the
  compute, trained on these sources) and 6x the matched-data UniVLA retrain (0.07). Its
  like-for-like probe is a small edge (0.29 vs 0.24-0.28). The released weights are clearly more
  robust to the moving patch (11-14% vs 23-27%); that gap is real and unaddressed. The wrong
  objective (control) destroys transport (0.03-0.09) without touching the clean probe, in both
  batch regimes: the readout has power.
- **Reading 2 -- the swap objective, as specified, does not work.** swap - rec = +0.07 (v1) and
  +0.02 (v1.1); swapalg - rec = +0.03 / +0.03. Under v1.1's within-embodiment batching the
  realism term is satisfied almost exactly (MMD 0.002-0.01), cycle is ~0.005, z carries 2.4x the
  information (80 vs 33 nats), the decoder is 1.7x more z-sensitive -- and clean R2 falls
  0.34 -> 0.28, reconstruction rises above 1.0 (worse than predicting zero), patch drop worsens.
  Realism + cycle were bought with non-action content and at the diagonal's expense: the
  pre-registered identifiability failure (risk #1), confirmed in the configuration designed to
  give the objective its best chance. Single seed; +0.02 is below any plausible floor.
- **Reading 3 -- algebra.** Composition/reversal errors 0.002-0.003 in both regimes; transport
  unchanged relative to swap. "Structured latent geometry alone is insufficient for semantic
  transport" stands.
- **Reading 4 -- the oracle arm is not a ceiling, in either regime.** Its training transport loss
  stays at/above the predict-zero level (0.95-1.07) even with same-embodiment nearest-action
  donors: the same action on a different state yields a different feature difference, so
  MSE-to-recipient-dF is unlearnable, and a nearest-of-32 donor is a weak action match (the
  Bridge oracle searched thousands at 0.90 agreement). Item 1 of the pre-registration is
  therefore not evaluable with this construction; a proper ceiling needs dataset-wide
  action-matched donors decoded and scored the way transport.py does.
- **Side result.** The UniVLA architecture trained on Bridge only loses 42% (z_q) / 99% (emb) of
  its action readout under the moving patch when read on the mixture; trained on the mixture it
  loses 16% / 28%. UniVLA's robustness comes from the training distribution, not the architecture.
- **Retractions inside this line.** The 09-06 "zero-transport sources" and the state-relative
  hypothesis (instrument artifact: foreign-unit donors); the 09-06 transport values (mixed
  donors); "swap adds nothing" was stated on the invalid numbers and is now stated on valid ones.
- **Artifacts.** `claude_survey/results/swaplam_{rec,control,swap,swapalg,oracle}{,_src}.json`,
  `results/eval_mix_mix_*.json`, jobs `20442294`, `20454376`, `20450559`, `20454377`.
- **Decision.** The swap objective (MMD realism + cycle) is dropped from the method. The method
  claim becomes the architecture + factorisation (P1 + P2-by-construction), with the transport
  readout as the P3 measurement. Next: (a) 3-seed floor (rec seeds 1/2 queued, `20455766/7`);
  (b) a valid ceiling (dataset-wide action-matched donors); (c) the robustness gap to the
  released weights -- the moving-patch intervention pair (`pairpatch` arm) is the registered
  lever and has not been run; (d) discretise the cells per source-agnostic small codebook and
  re-read, since VLAs consume tokens.

### 2026-09-07 — three-seed floor for the transport readout; v1.1 arms re-read against it

- **Floor (rec, v1.1 batching, seeds 0/1/2, 27 sources).** transport_action_r2 `0.452 / 0.439 /
  0.434` -> **0.441 +- 0.009**; clean R2 net `0.334 +- 0.006`; seg1 `0.289 +- 0.011`; profile
  `0.121 +- 0.004`; patch drop `26.4 +- 2.9` pp. The continuous latent is stable across seeds
  (compare the 0.03 sd of UniVLA's `z_q` on Bridge, C11).
- **v1.1 arms against it (single seed each; rec sd as the unit).**

| arm | transport | vs rec | clean R2 vs rec |
|---|---:|---:|---:|
| control | 0.034 | -0.41 (-44 sd) | +0.006 |
| oracle (nearest-of-32 donor) | 0.155 | -0.29 (-31 sd) | +0.007 |
| swap | 0.476 | **+0.034 (+3.7 sd)** | **-0.049 (-9 sd)** |
| swapalg | 0.479 | **+0.038 (+4.1 sd)** | **-0.029 (-5 sd)** |

- **Reading.** With the floor known, the swap terms do move transport: +0.03-0.04, about four rec
  standard deviations (the swap arms' own seed variance is not measured and their training is
  less stable, so treat ~4 sd as an upper bound on the confidence). The price is a clean-probe
  loss of 5-9 sd and reconstruction above the predict-zero level. So the corrected statement is
  not "no effect" but **"a small transport gain bought with action information and
  reconstruction"** -- the trade an identifiability failure would produce, and not a usable one.
  Decision unchanged: the swap objective as specified is dropped; the architecture result
  (0.44 vs 0.24 released / 0.07 matched UniVLA) is the claim, now with an error bar.
- **Artifacts.** `claude_survey/results/swaplam_rec_src{,_s1,_s2}.json`; jobs `20455766/7`.

### 2026-09-07 — review finding: `transport_action_r2` is circular; headline downgraded pending an independent reader

- **Finding (external review, accepted).** The readout is P(E(D(c_j, z_i))). Cycle training makes
  E(D(., z_i)) ~ z_i by construction, and D∘E autoencoding does so approximately even without it,
  so the number tracks P(z_i) whether or not the generated motion is the donor's. An explicit
  counter-model (per-context sign flip: E_B = -E_A, D_B = -D_A) has zero reconstruction, zero
  MMD, zero cycle and 0% correct transfer. This is C18's distinction ("the code survives
  transplantation; its effect does not") re-created under a new name.
- **Consequence.** Every "transport" number in the entries above is **round-trip retention of
  action information through E(D(.))**, renamed `roundtrip_retention_r2`. The headline
  "architecture transports at 0.44 vs UniVLA 0.24" becomes "architecture retains 0.44 vs 0.24
  through decode->re-encode on another state" -- a comparable quantity across the two models,
  but **not evidence for P3**. The control/oracle collapse (0.03-0.15) is then read as "the
  decoder stops writing z into its output", also not a transport statement.
- **Also accepted from the review.** (a) `u_j` sees the recipient's future and could carry its
  action (a bypass); in every run so far KL(u) = 0.000, so the bypass did not occur, but the
  design permits it. (b) MMD-to-observed-futures has the wrong reference distribution
  (behaviour policy vs donor distribution), so a correct intervention can be penalised.
  (c) Summing raw per-step actions is only a displacement for position-increment sources;
  velocity-controlled sources (jaco_play) need dt; rotation should not be summed.
  (d) DC/AC1 is a real information bottleneck (gripper events, reversals) to be checked.
- **Independent reader (implemented, jobs `20468363` swaplam x12, `20468364` UniVLA x3).** A
  per-source probe fitted ONLY on real frozen dF -> action is applied to the decoder's OUTPUT:
  `indep_self` = G(D(c_j,z_j)) vs a_j (can the output be read at all); `indep_transfer` =
  G(D(c_j,z_i)) vs a_i (donor's action realised on j -- the P3 number); `indep_replay` =
  G(D(c_j,z_i)) vs a_j (or is j's own motion replayed). Same-source donors. Neither E nor D
  enters the probe, so E(D(.)) ~ z cannot satisfy it. Limit: G is a linear reader in frozen
  feature space and may fail off the real-dF distribution; a decode->simulator counterfactual
  would be the true ground truth and is not available here.
- **Registered next arms (not run).** Following the review's design: A = rec; B = rec + MMD/cycle
  (done: swap); **C = grounded swap** -- action-matched pairs within embodiment / controller /
  timescale (labels, weak supervision), loss on the REAL recipient future through a control
  branch F_ctrl(c_j, z_i) that never sees u_j; **D** = C with the donor correspondence shuffled
  (same source/duration distribution); **E** = A + the same label budget as direct auxiliary
  supervision. Sanity first: F_GT(c_j, a_j) -> Y_j: if the true action cannot predict the effect
  target, fix the target before judging swap. Main metric: E_correct vs E_wrong vs E_self on real
  recipient futures, held-out contexts. Cycle and MMD become diagnostic columns.

### 2026-09-07 — independent effect reader: transfer is real for the architecture, and the swap objective improves it

- **Instrument.** Per-source ridge fitted ONLY on real frozen dF -> net xyz (never on E or D
  outputs), applied to the decoder's output. `indep_transfer` = G(D(c_j, z_i)) vs a_i (donor's
  action realised on recipient j; same-source donors); `indep_replay` = same vs a_j;
  `indep_self` = G(D(c_j, z_j)) vs a_j. 28 sources, 400 paired val clips/source. Job `20469450`
  (swaplam x12), `20468364` (UniVLA x3, first-pair construct).

| row | indep_transfer | indep_replay | indep_self | retention (old, circular) | clean R2 |
|---|---:|---:|---:|---:|---:|
| UniVLA released (first pair; its ref = 0.224) | 0.089 | -0.01 | 0.231 | 0.221 | 0.242 |
| UniVLA 10k retrain on the mixture | 0.008 | 0.007 | 0.108 | 0.071 | 0.130 |
| **rec** (4 runs: v1, v1.1 s0/s1/s2; ref = 0.302) | **0.245 / 0.253 / 0.252 / 0.248** | -0.00 | 0.23 | 0.45 | 0.34 |
| control (v1 / v1.1) | 0.095 / 0.065 | +0.08 | 0.20 | 0.06 / 0.03 | 0.36 |
| swap (v1 / v1.1) | **0.296 / 0.308** | -0.08 | 0.22 | 0.52 / 0.48 | 0.34 / 0.29 |
| swapalg (v1 / v1.1) | **0.339 / 0.335** | -0.15 | 0.26 | 0.48 / 0.49 | 0.33 / 0.32 |
| oracle (v1 / v1.1) | 0.102 / 0.126 | +0.07 | 0.20 | 0.11 / 0.14 | 0.35 |

- **Reading 1.** For the reconstruction-only architecture a donor's code decoded on another
  state realises 0.25 of the donor's action variance under an independent reader, with no
  replay of the recipient's motion (0.00). Relative to each model's own frozen-feature
  reference: ours 0.81, UniVLA released 0.40, UniVLA matched retrain 0.04. **This is the P3
  measurement that the retention metric was not**, and the architecture result survives it.
- **Reading 2 -- correction of 09-07's verdict on the swap objective.** On the independent
  reader swap adds +0.05 and swapalg **+0.09** over rec, against a four-run rec spread of
  0.004. The "adds nothing" verdict was read on the circular retention number and was wrong in
  the other direction; the review's point (retention != transfer) cut both ways. Costs stand:
  clean R2 -0.05 (v1.1), reconstruction above the predict-zero level, and the swap arms'
  decodes are farther from the real recipient future in raw MSE (E_correct 1.34 vs rec 1.03)
  -- directionally right, off the real feature distribution in scale. The swap objective is
  reinstated as a candidate, not as the method.
- **Reading 3.** control and oracle (MSE-to-recipient-future) partly replay the recipient
  (+0.07/+0.08) and transfer 0.07-0.13: decoders that learned to ignore z.
- **Weak instruments, noted.** E_self / E_correct / E_wrong as relative MSE on the real future
  all sit near 1.0 (feature differences are mostly unpredictable texture); donor discrimination
  is 0.58-0.62 for every arm. They are kept as diagnostics; the independent reader is the
  transfer number. The reader's own limit: a linear probe in frozen-feature space may read
  decodes off the real-dF distribution imperfectly; a decode->simulator counterfactual would be
  the ground truth and is unavailable here.
- **Pending.** Grounded-swap set (`20468964`: F_GT sanity, grounded, shuffled pairs, auxiliary
  label) -- the review's arms C/D/E -- read on the same instrument.

### 2026-09-07 — grounded set (review arms C/D/E + F_GT) and scale-free readings

- **Grounded set, job `20468964`, same instrument.** grounded (MSE cross-reconstruction of the
  recipient's real future from the nearest-action donor's code, u-free branch, dataset-wide pairs
  with 0.875 sign agreement): swap sensitivity 0.10, transfer 0.134, replay +0.05 -- the
  decoder damps z, as control/oracle did: the real feature difference is mostly unpredictable
  from any code, so the MSE-optimal decoder predicts from state. shuffled pairs: 0.034 (pairing
  matters, +0.10, but both below rec). auxlabel (same label budget on z): clean probe **0.412**
  (best) but transfer 0.228 = rec -- knowing the action better in the code does not make the
  decoder realise it elsewhere. Donor discrimination (E_correct < E_wrong) 0.58-0.69 everywhere;
  E_* relative MSEs sit near 1.0 and are kept as diagnostics only.
- **Scale-free check (cosine between predicted and true donor action), job `20472045`.**
  transfer cos: rec 0.449, swap 0.492, swapalg 0.485, grounded 0.257, auxlabel 0.408; replay cos
  0.05-0.15. The swap arms' lead over rec is +0.04 on cosine vs +0.08 on R2: about half the R2
  advantage was scale (their decodes are off the real-future distribution in magnitude,
  E_correct 1.48). A direction gain remains.
- **F_GT is not a ceiling for this reader.** Decoding from the TRUE 12-number action sequence
  gives self 0.165 / transfer 0.115 (cos 0.23) -- below the z-path. The visual difference
  carries more readable content than xyz (object, gripper, image position); the sanity passes
  (target predictable from the true action) but no ceiling exists on this instrument.
- **Standing order on the independent reader.** swap ~ swapalg > rec > auxlabel ~ F_GT(z-path)
  > grounded(MSE) > oracle ~ control > shuffled. The architecture result holds; the swap
  objective's gain is real in direction and smaller than its R2 suggested.
- **Registered and queued (`20472049`).** Rank form of the grounded loss (review sec. 5.3): on the
  same recipient state the matched donor's code must explain the real future better than a
  wrong same-source donor's by a margin (negatives with indistinguishable actions excluded);
  alone (`grank`) and on top of swapalg (`swapalg_grank`). Goal: keep the transfer gain, remove
  the scale/clean-probe cost.

### 2026-09-07 — rank-form grounded loss (review sec. 5.3): scale fixed, direction not gained

- **Job `20472049`, same instrument, single seed.** `grank` = rec + rank margin (matched donor's
  code must explain the recipient's real future better than a wrong same-source donor's; final
  rank_acc 0.74). `swapalg_grank` = swapalg + the same term.

| arm | transfer R2 | transfer cos | replay cos | clean R2 | E_correct | patch drop % | KL(z) |
|---|---:|---:|---:|---:|---:|---:|---:|
| rec (4 runs) | 0.245-0.253 | 0.449 | 0.10 | 0.34-0.36 | 1.04 | 23-24 | 33 |
| swap | 0.308 | 0.492 | 0.06 | 0.29 | 1.18 | 22 | 80 |
| swapalg | 0.335 | 0.485 | 0.05 | 0.32 | 1.48 | 34 | 63 |
| **grank** | **0.319** | 0.451 | 0.00 | **0.360** | 1.20 | 24.6 | 37 |
| swapalg_grank | 0.299 | 0.438 | 0.03 | 0.272 | 1.20 | 31 | 36 |

- **Reading 1 -- grank.** Transfer R2 +0.066 over rec (rec four-run spread 0.008), clean probe
  0.360 = the top of rec's 0.34-0.36 range (not above it), replay cosine 0.00, patch drop and KL
  at rec level; none of swapalg's costs. Transfer cosine unchanged (0.451 vs rec 0.449). Recorded
  as: **squared-error transfer improved, mean direction similarity not improved.** Whether the
  gain is magnitude calibration or a magnitude-weighted redistribution of direction error is NOT
  decided by two means (a mean cosine can stay fixed while large-action directions improve and
  small-action ones degrade); the per-sample radial/angular decomposition and the val-fitted
  scalar calibration (`diag_transfer.py`) decide it. E_correct 1.20 vs rec 1.04 is a remaining
  cost on the real-future error, not a scale statement.
- **Reading 1b -- how the rank margin was met (review sec. 4).** The negative branch carries
  gradient (no stop-grad, `swaplam.py` grank block). Positives and negatives BOTH got worse on
  the real recipient future: E_correct 1.04 -> 1.20, E_wrong 1.07 -> 1.25; the gap widened
  0.033 -> 0.050 and rank_acc reached 0.74. The margin was earned mainly by pushing wrong donors
  away, not by bringing the correct donor's decode closer -- the review's zero-loss counterexample
  in the mild form. An absolute anchor on the correct decode is missing from this loss.
- **Reading 2 -- swapalg_grank.** Below grank on transfer (0.299), cosine (0.438) and clean
  (0.272); below swapalg on transfer and cosine but BETTER than swapalg on E_correct (1.20 vs
  1.48) and patch drop (31 vs 34). Accurate statement: at these weights and this schedule the
  joint objective is dominated by the simpler grank. Not a statement that the two objectives are
  incompatible; the alternative explanation (nearest-action pairs supervise "look like a_j", the
  reader scores "look like a_i") is testable from C_pair / C_real_pair and is queued.
- **Standing order (transfer cosine, the scale-free number).** swap 0.49 ~ swapalg 0.49 > grank
  0.45 ~ rec 0.45 > swapalg_grank 0.44 > auxlabel 0.41 > grounded 0.26 > F_GT 0.23. On R2 grank
  sits between rec and swapalg with none of swapalg's costs.
- **Caveat on cosine itself (review sec. 3).** Cosine is computed on the reader's OUTPUT; the
  reader is affine (standardise, W, intercept), so scaling its input can rotate its output.
  Cosine on actions therefore does not fully exclude the exaggerated-feature explanation of the
  swap arms' lead. Reader-input scale sensitivity (gamma in {0.5, 1, 2}) is part of the same
  offline dissection.
- **Consequence.** Two distinct positive signals, not one winner: grank lowers transfer squared
  error at no clean/robustness cost; swap/swapalg raise direction similarity at a clean-probe
  cost. Neither's mechanism (scale vs direction; encoder vs decoder) is separated yet. The
  direct sum is paused, not rejected. F_GT stays as a true-action-conditioned baseline, not a
  ceiling.
- **Cosine floor is missing.** rec cosine exists for one run only (s1/s2 checkpoints predate the
  cosine readout). A +0.04 cosine claim needs the rec and swap seed spread on cosine before it is
  reported as a result. Registered: re-read rec_src_s1/s2 (eval-only) and train swap seeds 1/2.
- **rec cosine floor (job `20487187`, eval-only re-read of s1/s2).** transfer cos 0.449 / 0.443 /
  0.441 (mean 0.444, spread 0.008); replay cos 0.09-0.10; self cos 0.40 all three. grank 0.451
  sits inside this floor: confirmed no direction gain. swap 0.492 and swapalg 0.485 sit 5x the
  spread above it; the swap-side seeds (`20487188/89`) are still needed before the +0.04 is
  reported as a result.

### 2026-09-07 — offline dissection of the transfer number (review secs. 2, 3, 5; `diag_transfer.py`)

Same reader, same episode-disjoint split as the readouts; transfer decodes only; clip-weighted
over 28 sources. Squared error split as ||p-a||^2 = (|p|-|a|)^2 + 2|p||a|(1-cos) (both terms / sum
|a|^2, so they add to NMSE). alpha = a single positive scalar fitted on half the test episodes,
R2 read on the other half. gamma scales the reader's INPUT (the decoded feature difference).

| arm | R2 | cos | wcos | NMSE = radial + angular | |p|/|a| | alpha | R2 calibrated | cos at gamma 0.5 / 2 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| rec s0/s1/s2 | 0.253/0.251/0.248 | 0.449/0.443/0.441 | 0.62 | 0.78 = 0.63 + 0.15 | 0.27 | 2.1 | 0.298/0.292/0.288 | 0.37 / 0.47 |
| swap | 0.308 | 0.492 | 0.667 | 0.72 = 0.56 + 0.17 | 0.35 | 1.7 | **0.337** | 0.42 / 0.50 |
| swapalg | 0.335 | 0.485 | 0.672 | 0.69 = 0.47 + 0.22 | 0.46 | 1.4 | **0.333** | 0.45 / 0.48 |
| grank | 0.319 | 0.450 | 0.629 | 0.71 = 0.43 + 0.28 | 0.52 | 1.3 | 0.313 | 0.43 / 0.44 |
| swapalg_grank | 0.299 | 0.438 | 0.605 | 0.73 = 0.46 + 0.27 | 0.49 | 1.2 | 0.308 | 0.42 / 0.44 |

- **Finding 1 -- the transfer error is mostly shrinkage.** For rec 81% of the NMSE is radial: the
  reader reads a donor's action off the recipient decode at 0.27 of its true magnitude. Every
  arm shrinks; grank shrinks least (0.52).
- **Finding 2 -- grank's R2 gain is a global scalar, confirmed.** One scalar fitted on held-out
  episodes lifts rec to 0.288-0.298 and leaves grank at 0.313: the +0.066 becomes +0.015, about
  1.5x the rec calibrated spread. Magnitude-weighted cosine grank 0.629 vs rec 0.620. Stated
  as: grank = rec + a scale the decoder learned to apply; no direction gain on either the
  unweighted or the weighted cosine. (The angular NMSE term rises for grank because |p| rises;
  it is not evidence of worse direction.)
- **Finding 3 -- the swap arms' gain survives calibration.** After the same scalar swap 0.337 and
  swapalg 0.333 vs rec 0.29-0.30: +0.035-0.04 remains, and wcos +0.05. The direction gain is
  not a scale effect at the action level.
- **Finding 4 -- reader-input scale (sec. 3) partly matters, and does not explain the lead.** The
  reader's cosine IS input-scale sensitive (rec 0.37 -> 0.45 -> 0.47 at gamma 0.5/1/2, the
  intercept effect the review derived). But swap is above rec at every gamma; at rec's most
  favourable gamma=2 (0.467) swap sits at 0.500, swapalg at 0.483. Feature-scale exaggeration
  does not account for the swap lead; its size at matched scale is ~0.02-0.03, not 0.04.
- **Finding 5 -- pairing does not set up a conflicting target (sec. 5).** Matched nearest-action
  pairs agree in direction: C_pair (source-mean-removed) 0.93, median 0.97. The reader applied
  to the recipient's REAL future scores the donor's action (C_real_pair 0.38) nearly as high as
  the recipient's own (C_real_self 0.41). Supervising "look like the real Y_j" and scoring
  "reads as a_i" ask for nearly the same thing; the swapalg_grank failure is not a pairing
  conflict. (The R2 form of these two rows on the 30-50-clip pair subsets is dominated by a few
  large negative sources and is not reported.)
- **Standing after dissection.** swap ~ swapalg: real but small direction gain (+0.02-0.04 cos
  depending on the scale convention), clean probe -0.03 to -0.05, single seed each. grank:
  calibration only. rec: 0.44 cos floor spread 0.008. Seeds for swap/swapalg (`20487188/89`)
  and grank (`20492640`) queued; frozen-encoder rec-init test for grank and rec (`20492637`,
  `--init-from rec_src --freeze-encoder`) queued: does the calibration live in the decoder alone?

### 2026-09-08 — decisions after the dissection, and the registered order

- **Method line re-cut.** Plain `swap` is the core experimental method: the only candidate whose
  gain survives held-out scalar calibration (+0.04 R2, +0.05 weighted cos), reader-input scaling
  (above rec at every gamma), and -- so far -- one replicate seed (cos 0.492 / 0.485 vs rec
  0.441-0.449). `swapalg` is demoted to an ablation: after calibration it does not beat plain swap
  (0.333 vs 0.337) and costs more (E_correct 1.48, patch drop 34%). `grank` is a decoder
  magnitude-calibration baseline, not a latent-learning method; its rank margin was met by
  separating negatives (sec. 4), which a future grounding loss must not allow.
- **Seeds so far.** swap s0/s1: transfer 0.308/0.321, cos 0.492/0.485, clean 0.29/0.31.
  swapalg s0/s1: 0.335/0.341, cos 0.485/0.473. Seed 2 (`20487189`) running; grank seeds
  (`20492640`) queued.
- **Decisive experiment.** rec_src encoder frozen, decoder trained: `rec_fz` (control), `grank_fz`
  (`20492637`, running) and `swap_fz` (`20494896`, queued). Reading rule, on transfer cosine:
  swap_fz ~ full swap (0.49) => the gain is decoder adaptation and the method becomes a
  state-conditioned transport decoder; swap_fz ~ rec (0.44) => the swap objective changes the
  latent geometry and the representation claim stands; in between => report the split.
- **Prepared, not submitted.** `dat` = swap + positive-anchored directional transport:
  (1 - cos(D(c_j, z_i+), e_j)) + 0.5 [0.1 - s+ + s-]_+ on the u-free branch, pair-weighted,
  magnitude deliberately absent; `dat_shuf` = same with the positive partner shuffled (null).
  `run_swaplam_dat.sh` carries a GPU smoke gate. Submitted only if full swap beats swap_fz.
- **Registered last.** Few-label cross-context action recovery: fit a = Wz on p in {1, 5, 10,
  100}% of labelled episodes in context A, freeze W, read R2 in context B (episode- or
  task-disjoint within source); rec vs UniVLA vs swap vs grank vs dat. Offline on saved Z.

### 2026-09-08 — few-label cross-context action recovery (`recovery_curve.py`): swap does NOT win

Linear a = Wz fitted on p of the labelled context-A episodes (ridge, inner episode split), W
frozen, read in context B. Two within-source contexts: held-out episodes (28 sources) and
held-out INSTRUCTIONS (18 sources have >= 2 instructions; 10 fall back to the episode split).
Clip-weighted over sources; 5 draws of the labelled subset. R2 (cos):

| arm | p=1% | 5% | 10% | 25% | 100% | task-held-out, 100% |
|---|---:|---:|---:|---:|---:|---:|
| rec | -0.02 (0.22) | 0.00 (0.21) | 0.07 (0.25) | 0.17 (0.33) | 0.34 (0.47) | 0.11 (0.40) |
| swap s0 / s1 | -0.07 (0.18) | -0.03 (0.18) | 0.03 (0.22) | 0.12 / 0.13 (0.27) | 0.29 / 0.31 (0.41 / 0.43) | 0.14 / 0.12 (0.35 / 0.36) |
| swapalg | -0.05 (0.20) | -0.05 (0.20) | 0.02 (0.21) | 0.12 (0.29) | 0.32 (0.42) | 0.10 (0.35) |
| grank | -0.01 (0.23) | 0.02 (0.22) | 0.08 (0.26) | 0.19 (0.34) | 0.36 (0.47) | 0.11 (0.40) |

- **Reading.** At every label fraction and in both contexts the swap arms are below rec on
  cosine (-0.04 to -0.06) and, except one cell, on R2. The swap objective does not make the
  latent more linearly action-recoverable with few labels or under instruction shift; it makes
  it less so, by the same margin as the clean-probe cost already recorded. grank ~ rec (slightly
  above). Below p=25% (3-10 labelled episodes for a 256-d z) every arm is at or below zero and
  the curve carries no information; the informative cells are p >= 25%.
- **Consequence for the claim.** The review's own test for "swap learns a shared action
  coordinate" comes out negative. What swap improves is the DECODER-side transfer (donor action
  realised on another state, read by an independent effect reader), not the linear action
  content of z. This is consistent with the decoder-adaptation branch of the frozen-encoder
  test; `swap_fz` (`20494896`) decides it. If swap_fz reproduces full swap, the representation
  claim is dropped and the contribution is the transport decoder.
- **Limits.** Contexts are within source (units differ across sources); target is net xyz;
  UniVLA is missing from this table (its Z was never dumped; eval_mix needs a Z dump re-read).

### 2026-09-08 — swap / swapalg three seeds: swap's direction gain replicates; swapalg collapsed once

| arm | seeds | transfer R2 | transfer cos | calibrated R2 | clean R2 | KL(z) nats |
|---|---|---|---|---|---|---|
| rec | 0,1,2 | 0.253 / 0.251 / 0.248 | 0.449 / 0.443 / 0.441 | 0.298 / 0.292 / 0.288 | 0.34 / 0.35 / 0.36 | 33 |
| swap | 0,1,2 | 0.308 / 0.321 / 0.315 | **0.492 / 0.485 / 0.489** | **0.337 / 0.345 / 0.351** | 0.29 / 0.31 / 0.31 | 80 |
| swapalg | 0,1 | 0.335 / 0.341 | 0.485 / 0.472 | 0.333 / 0.330 | 0.32 / 0.31 | 63 / 60 |
| swapalg | 2 | **0.081** | **0.170** | 0.084 | 0.20 | **12** |

- **swap vs rec, seed-paired (s0-s0, s1-s1, s2-s2), independent trainings.** cos +0.043 / +0.042
  / +0.048: mean +0.044, sd 0.003, 95% t-interval [+0.036, +0.052]. Calibrated R2 +0.039 /
  +0.053 / +0.063: mean +0.052, interval [+0.022, +0.082]. Clean probe -0.04 (three of three).
  Reader-input gamma sweep identical across seeds (swap above rec at every gamma). The
  direction gain of the plain swap objective is **confirmed** on this instrument, with its
  clean-probe cost. Limit of the instrument stands: a linear reader in frozen-feature space
  applied to decodes; no simulator counterfactual, no policy.
- **swapalg seed 2 collapsed:** KL(z) stayed at 12 nats (others 60-80), transfer 0.08, clean
  0.20. The algebraic terms plus KL admit a low-information solution and one of three seeds
  found it. swapalg is seed-unstable and stays an ablation, reported with all three seeds.
- **Standing on the swap objective.** Real, replicated, small direction gain on transfer; a
  replicated cost on the linear action recoverability of z (clean probe and the few-label
  curve). Whether the gain lives in z or in the decoder: `swap_fz` (`20494896`, running) decides.

### 2026-09-08 — DECISIVE: frozen-encoder test. Both gains live in the decoder; the encoder update hurts

rec_src encoder (E, U, cell embeddings) frozen; decoder re-trained 10k steps from the rec_src
weights under three objectives. Same z for all three, so every readout difference is the decoder.
Jobs `20492637` (rec_fz, grank_fz), `20494896` (swap_fz). Single seed each.

| decoder objective | encoder | transfer R2 | transfer cos | wcos | calibrated R2 | clean R2 | patch drop | E_correct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rec (parent) | trained | 0.253 | 0.449 | 0.620 | 0.298 | 0.342 | 24.0 | 1.04 |
| rec, continued (control) | frozen | 0.258 | 0.449 | 0.618 | 0.299 | 0.342 | 24.0 | 1.03 |
| grank, full | trained | 0.319 | 0.450 | 0.629 | 0.313 | 0.360 | 24.6 | 1.20 |
| grank, decoder only | frozen | 0.318 | 0.463 | 0.633 | 0.320 | 0.342 | 24.0 | 1.26 |
| swap, full (3 seeds) | trained | 0.308-0.321 | 0.485-0.492 | 0.667-0.671 | 0.337-0.351 | 0.29-0.31 | 22-25 | 1.17-1.18 |
| **swap, decoder only** | **frozen** | **0.354** | **0.512** | **0.696** | **0.358** | **0.342** | **24.0** | 1.22 |

- **Control.** Continuing the rec decoder for 10k more steps changes nothing (0.258 / 0.449).
  Whatever the other two rows gain is the objective, not the extra steps.
- **grank: case A, exactly.** Decoder-only grank reproduces full grank's transfer (0.318 vs
  0.319). Nothing of it was in the encoder.
- **swap: stronger than case A.** Decoder-only swap BEATS full swap on every transfer column
  (cos 0.512 vs 0.485-0.492, 3 seeds; calibrated R2 0.358 vs 0.337-0.351) and pays none of its
  costs: clean R2 stays at rec's 0.342 (full swap 0.29-0.31), patch drop stays at 24.0. The
  encoder update in full swap was the source of the clean-probe cost and also cost transfer.
  Reader-input gamma sweep: swap_fz above every other row at every gamma (0.453 / 0.512 / 0.516).
- **Verdict on the representation claim.** Dropped. Neither the swap objective nor the rank
  objective improves the latent; both improve how a state-conditioned decoder realises the
  rec latent on another state. The few-label recovery curve (swap below rec on z) said the same
  from the other side. Consistent picture: the rec encoder's z already carries the transferable
  direction; the reconstruction decoder under-realises it (0.27 of true magnitude, 81% radial
  error); the swap objective teaches the decoder to realise it.
- **What the contribution is now.** A cross-context action transport DECODER trained by
  swap realism on a frozen reconstruction latent -- the J(s)z branch the review named. The
  latent is UniVLA-style rec; the method is the decoder objective. This is a smaller and more
  defensible claim than "better latent action geometry", and it has the best transfer numbers
  in the project with zero clean-probe or robustness cost.
- **Not submitted, per the registered rule.** `dat` (swap + positive anchor on the encoder+
  decoder) was conditional on full swap beating swap_fz; it did not. If a grounding term is
  tested at all it is on the frozen-encoder decoder.
- **Open.** swap_fz is one seed (rec_fz control says the decoder is deterministic enough that
  the parent's 3-seed spread of 0.003 cos is the relevant floor, but a swap_fz seed is cheap:
  20 min). No downstream policy number anywhere yet.

### 2026-09-08 — paper line fixed: Preserve (AC/DC) -> Disentangle (TC/TI) -> Transport (LATP)

- **Decision (user).** The frozen-encoder result relocates failure 3 (cross-context) to the
  decoder; failures 1 (temporal information loss) and 2 (TC/TI ambiguity) stay with the
  tokenizer. Main method = three stages, one per failure: Stage I AC/DC effect cells (preserve
  temporal modes before the bottleneck); Stage II TC/TI disentanglement over AC/DC cells
  (semantic axis orthogonal to the temporal axis: 2x2 table z_{TC/TI}^{DC/AC}); Stage III
  Latent Action Transport Pretraining (LATP): tokenizer frozen, state-conditioned transport
  operator T(s, z_TC) trained action-free by cross-context swap realism (grad wrt E = 0).
  "swap objective" is the training signal; LATP is the stage. Ablations only: relation
  diff/sym, RALA, swapalg (unstable), grank (calibration), InfoNCE, canonical head.
- **Evidence status per stage.** LATP: confirmed on the controlled tokenizer (rec z, cos 0.449
  -> 0.512 decoder-only; 3 full-swap seeds below it; seeds of swap_fz `20495434` queued).
  AC/DC: candidate (earlier z->action 0.086 -> 0.221, state leak 0.317 -> 0.076 on Bridge; not
  yet shown against the matched-data UniVLA baseline with a direct information metric). TC/TI:
  no evidence yet; no TC/TI model exists on top of AC/DC cells.
- **Registered metrics (user), not yet run.** (1) Temporal reconstruction probe: recover
  F_{1:T} from mean-pooled / endpoint / DC / DC+AC1 / DC+AC1+AC2, error vs bitrate, plus
  action probes (net AND profile targets) from each subset -- offline on saved DF is possible
  for the dF spectrum; profile targets need the val action sequences (MixDS._action_seq on the
  stratified_pair_indices(seed=123) rows). (2) Time-reversal probe: DC invariant, AC1
  antisymmetric under clip reversal. (3) TC/TI separation gap G_sep = (S_TC - S_TI) + (N_TI -
  N_TC) with action and nuisance (moving-patch) probes, ours vs UniVLA -- needs a Stage-II
  model and a nuisance target; design first. (4) End: downstream data efficiency / success.

### 2026-09-08 — method repo `latent_free/acdc_la/` created; `claude_survey` archived to the live set

- **Decision (user).** The main line is now real repo code, separate from the experiment
  ground: `/anvil/scratch/x-tding2/latent_free/acdc_la/` (README = the paper line, one table per
  stage: fix / objective / metric / evidence status). `claude_survey/` keeps only the 16 modules
  in the import closure of the live scripts (swaplam, oxe_mix, mixprobes, eval_mix, eval_rala2,
  rala2, diag_transfer, recovery_curve, lang_cache, univla_audit*, cycle, controls, transport,
  rala, summarize_transport_operational) plus their run scripts and STATUS.md; 40 closed
  scripts, 53 run files and 6 superseded docs moved to `claude_survey/archive/{py,run,docs}`
  (commit 06fa70d; running jobs 20492640 / 20495434 use swaplam.py, untouched).
- **Repo design.** One `LAM` (tokenizer E/U + state-conditioned transformer decoder) with the
  legacy attribute names, so every `swaplam_*.pt` loads (verified: rec_src, swap_fz). One
  `Featurizer` with the Stage-I operator as a switch (`--pool acdc --modes {1,2,3}` vs `mean` /
  `endpoint`), one training loop (`common/train_loop.py`; a stage = its `step()` + flags), one
  validation pass -> `results/<tag>_val.npz` (codes clean / patch / time-reversed, frozen dq,
  decodes, swap decodes, actions, patch trajectory); every metric is offline from that file.
  Stage trainers: `stage1_preserve.train` (rec+KL), `stage2_disentangle.train` (clean+patch
  renderings, z-invariance + symmetric cross-reconstruction so the nuisance lives in u),
  `stage3_transport.train` (init from a tokenizer, encoder frozen by default, MMD realism +
  cycle; `--train-encoder` = full-swap ablation, `--lam-real 0 --lam-cyc 0` = decoder-only
  control). Probes: `stage1_preserve.probes` (temporal-variance kept and action probes
  net/seg1/prof/seq per subset {DC}={mean}={endpoint}, {DC,AC1}, {DC,AC1,AC2}, full;
  time-reversal parity of z; patch drop), `stage2_disentangle.probes` (G_sep with the nuisance
  target = patch net displacement + mean position, exact from the render law),
  `stage3_transport.probes` (independent reader + the whole dissection), `.recovery` (few-label
  curve). `baselines/univla.py` dumps UniVLA stage-2 z_q (TC) and z_q_uncontrol (TI) on the
  same val clips in the same format. Slurm: `scripts/run_stage{1,2,3}.sh`, `run_univla_dump.sh`
  (ai partition, smoke first, probes at the end).
- **Reproduction check (confirmed).** `stage3_transport.probes rec_src swap_fz` on the legacy
  dumps: rec_src transfer R2 0.253 / cos 0.449 / calibrated 0.298 / clean 0.342; swap_fz 0.354
  / 0.512 / 0.359 / 0.342 -- identical to the ledger entry of the decisive test.
- **Correction to the registered time-reversal probe.** With DC = SUM of deltas (net
  displacement) and dF_k -> -dF_{K-1-k} under reversal, DC FLIPS SIGN and AC1 is INVARIANT
  (AC2 flips) -- not "DC invariant, AC antisymmetric". `common.features.reversal_signs` derives
  the sign of every mode from the basis parity (checked exactly on the frozen cells); the probe
  reports cos(z_rev, s_mode * z) per mode with s printed, so the reading cannot be typed wrong.
- **Not done.** No Stage-I / II / III job of the new repo has run yet; Stage-I model-free
  probe on the legacy rec_src dump running (CPU). `latent_free/.git` had 82 missing blobs /
  26 missing trees (pre-existing, not from this reorg); 49 blobs restored from identical
  worktree files, HEAD tree intact, commit 813afc9 succeeded; older history remains partly
  unreadable.

### 2026-09-08 — Stage-I model-free temporal-subset probe, first read (provisional; dimension-confounded)

- **What ran.** `acdc_la.stage1_preserve.probes rec_src` on the legacy val dump (9,753 clips, 25
  sources, K=4 quadrant-pooled frozen dq).  Per temporal subset of the DCT basis, the fraction of
  the dq sequence's variance kept, and per-source ridge (episode-disjoint, clip-weighted over
  sources with >= 80 test clips) from the subset's cells to net xyz and to the first segment's xyz
  (seg1).  Legacy dumps carry no full action sequence, so prof / seq targets are not available
  until the new dumps exist.

  | subset (cells) | dims | var kept | R2 net | R2 seg1 |
  |---|---|---|---|---|
  | DC (= mean-pooled = endpoint) | 3072 | 0.128 | **0.444** | 0.291 |
  | DC + AC1 (the method's cells) | 6144 | 0.356 | 0.410 | **0.366** |
  | DC + AC1 + AC2 | 9216 | 0.649 | 0.386 | 0.356 |
  | full sequence | 12288 | 1.000 | 0.365 | 0.342 |
  | tokenizer z (rec_src) | 256 | -- | 0.342 | 0.296 |

- **Reading (provisional).** (i) For the NET target the DC cells alone are best and every added
  mode lowers R2 -- while strictly adding information.  At ~270 training clips per source a
  ridge over 6k-12k dims pays for its dimension; this row therefore measures estimator variance
  as much as information and must not be read as "AC modes hurt".  (ii) For seg1, a target that
  is not a function of the clip sum, DC+AC1 beats DC by +0.075 (0.291 -> 0.366) despite the
  same dimension penalty; AC2 adds nothing further.  That is the direction the Preserve claim
  needs, on the field's own kind of target (a per-step action), model-free.  (iii) The rec_src
  tokenizer's z (256-d) reads seg1 at 0.296 -- the DC level, below its own input cells' 0.366:
  the 8x32 Gaussian bottleneck did not keep the AC1 information it was given.  This is the first
  direct measurement of Stage I's own loss and is the number a better tokenizer must move.
- **Fix registered and running.** The dimension confound is removed by probing every subset
  after unsupervised PCA to 256 dims (= z's size), added to `stage1_preserve/probes.py`
  (`pca_reduce`, randomized PCA fitted on all val clips, labels never touched); rerun on rec_src
  is in progress on the login node (load ~360, slow).  prof / seq targets and the
  time-reversal probe need the new dumps (smoke job 20496311 exercises them; the real Stage-I
  runs `s1_acdc / s1_dc / s1_mean / s1_acdc3` have not been submitted).
- **Dimension-matched read (done, same run).** Every subset PCA'd (unsupervised) to 256 dims
  before the same ridge:

  | subset @ 256 dims | R2 net | R2 seg1 |
  |---|---|---|
  | DC | 0.391 | 0.253 |
  | DC + AC1 | 0.388 | **0.323** |
  | DC + AC1 + AC2 | 0.363 | 0.287 |
  | full sequence | 0.331 | 0.258 |
  | tokenizer z (rec_src, 256) | 0.342 | 0.296 |

  At equal bitrate DC+AC1 costs nothing on net (-0.003) and gains +0.070 on seg1 over DC; a
  third mode or the full sequence spends the 256 dims on temporal detail the probe cannot use
  and loses on both targets.  So at this budget DC+AC1 is the operating point, not "more modes
  is better" -- the Preserve claim is a claim about ONE extra mode.  Against the linear ceiling
  of its own input (DC+AC1 @ 256) the learned z is -0.046 on net and -0.027 on seg1: the
  tokenizer's nonlinearity and KL bottleneck currently cost, not gain, and a mean-pooled
  tokenizer trained the same way (`s1_mean`) is the fair comparison still to run.
- **Not claimed.** Nothing here compares to UniVLA yet (its dump needs `baselines/univla.py`,
  in the smoke job), and nothing here is a trained-tokenizer comparison; the pooled tokenizers
  `s1_mean` / `s1_dc` are the registered Stage-I ablations.

### 2026-09-08 — swap_fz and grank three seeds (jobs 20495434, 20492640): LATP replicates; grank has no direction gain

- **What ran.** `swap_fz_s1/s2`: rec_src tokenizer frozen, decoder retrained with the swap objective,
  seeds 1, 2 (same z as seed 0, so the spread is decoder-training noise only).  `grank_g_s1/s2`:
  full grank (encoder trained), seeds 1, 2.  10k steps, batch 32, same readout set.  Independent
  reader numbers (`swaplam_*.json`); calibrated / dissection numbers from `acdc_la.stage3_transport.probes`
  are being computed.

  | arm | seed | transfer R2 | transfer cos | clean R2 | replay R2 | E_correct / E_wrong | discrim | swap_sens |
  |---|---|---|---|---|---|---|---|---|
  | rec_src (tokenizer) | 0 | 0.253 | 0.449 | 0.342 | -0.01 | 1.04 / 1.07 | 0.61 | 0.35 |
  | rec_fz (decoder-only control) | 0 | 0.259 | 0.448 | 0.342 | -0.02 | 1.03 / 1.07 | 0.62 | 0.37 |
  | **swap_fz (LATP)** | 0 / 1 / 2 | **0.354 / 0.352 / 0.348** | **0.512 / 0.493 / 0.496** | 0.342 (all) | -0.19 / -0.20 / -0.18 | 1.22 / 1.37 (all ~) | 0.60 | 0.71 / 0.72 / 0.70 |
  | swap (full, encoder trained) | 0 / 1 / 2 | 0.308 / 0.321 / 0.315 | 0.492 / 0.485 / 0.489 | 0.294 / 0.311 / 0.311 | -0.09 | 1.18 / 1.24 | 0.60 | 0.61 |
  | grank (full) | 0 / 1 / 2 | 0.319 / 0.298 / 0.305 | 0.451 / 0.454 / 0.445 | 0.360 / 0.344 / 0.347 | -0.29 / -0.44 / -0.43 | 1.20 / 1.25, 1.32 / 1.39, 1.26 / 1.32 | 0.67 / 0.66 / 0.67 | 0.49 / 0.61 / 0.56 |

- **Reading (confirmed).** LATP replicates: transfer R2 0.351 +- 0.003 (3 seeds) against the
  tokenizer's 0.253 and the decoder-only control's 0.259, with clean R2 identical to the
  tokenizer by construction (the encoder is frozen).  Against full swap, paired by seed, the R2
  gain is +0.037 [+0.016, +0.057] (every seed positive: +0.046, +0.031, +0.033) at zero clean
  cost where full swap pays -0.03 to -0.05.  **The cosine advantage over full swap is NOT
  established**: +0.012 [-0.006, +0.030]; seed 0's 0.512 was the high draw, seeds 1-2 give
  0.493 / 0.496 against full swap's 0.485 / 0.489.  Both beat the tokenizer's 0.449 on every
  seed (LATP +0.051 +- 0.010).  So the confirmed statement is: LATP >= full swap in
  direction, > full swap in R2, and = tokenizer in clean; the "0.512" in earlier entries should
  be quoted as 0.500 +- 0.010.
- **grank, three seeds (closes the question from the 2026-09-07 review).** Direction gain
  is zero: cos 0.450 +- 0.005 vs rec 0.449.  R2 0.307 +- 0.011 (seed 0's 0.319 was the high
  draw); clean 0.350 +- 0.008 vs rec 0.342 (within noise).  E_correct and E_wrong both rise on
  every seed (1.04 -> 1.26-1.32 and 1.07 -> 1.32-1.39): the margin is met by pushing the wrong
  donor away, exactly as read on seed 0.  grank stays an ablation (a calibration-type gain with
  no directional content), and the ledger's earlier "+0.066 R2" for grank is now 0.307 - 0.253
  = +0.054 +- 0.011.
- **Dissection of the seeds (`acdc_la.stage3_transport.probes`, same reader).** Scalar-calibrated
  transfer R2: swap_fz 0.359 / 0.347 / 0.357 (0.354), full swap 0.338 / 0.345 / 0.351 (0.345),
  grank 0.313 / 0.325 / 0.314 (0.317), rec 0.298.  Paired by seed, LATP - full swap calibrated =
  +0.010 [-0.015, +0.035] -- **not established**; LATP - rec = +0.056 [+0.040, +0.072] and full - rec = +0.047 [+0.031, +0.063] are.  What
  separates LATP from full swap after calibration is therefore magnitude, not direction:
  |p|/|a| 0.43 vs 0.35 (alpha 1.42 vs 1.73), radial NMSE 0.49 vs 0.55, angular NMSE equal within
  0.02 (0.18-0.20 vs 0.16-0.17).  grank's calibrated gain over rec is +0.019 [+0.003, +0.036] with angular NMSE
  0.28-0.33 (worse than rec's 0.15): its higher raw R2 comes from |p|/|a| 0.52-0.62 (least
  radial error of all arms) while its direction is the worst.  Reader-input scale sensitivity
  is unchanged in ordering at gamma 0.5 and 2.  E_* / matched-donor rows are absent for these
  dumps (legacy arrays do not carry them; the json values are in the table above).
- **Consequence for the repo.** `stage3_transport.train` defaults (frozen encoder, MMD + cycle)
  are the replicated configuration; `--train-encoder` is the ablation that loses.

### 2026-09-09 — formal ACDC-LA pretraining submitted (repo `latent_free/acdc_la`)

- **Smoke (job 20496311) passed** for every GPU path: `stage1_preserve.train` (acdc and mean pool),
  `stage2_disentangle.train`, `stage3_transport.train` (from a new Stage-I ckpt and from legacy
  rec_src), `baselines.univla`.  Probe code paths ran; their numbers on the smoke dumps are NaN
  because 20 clips per source cannot pass the 80-clip / 10-test-clip floors (not a bug; three
  edge cases fixed: env `ACDC_MIN_CLIPS`, missing ASW key in the UniVLA dump, empty-aggregate print).
- **Submitted (ai partition, all 10k steps, batch 32, seed 0, T=5 stride 4, all 25 sources):**
  - `20517212` Stage I: `s1_acdc` (DC+AC1, the method), `s1_dc` (DC only), `s1_mean`
    (mean-pooled), `s1_acdc3` (DC+AC1+AC2); ends with `stage1_preserve.probes` on all four.
  - `20517213` UniVLA stage-2 released weights dumped on the same val clips (`univla_official`),
    then Stage-I and Stage-II probes on it.
  - `20517214` Stage III from `s1_acdc` (afterok 20517212): `s3_latp` (frozen tokenizer; the
    method), `s3_ctrl` (decoder-only, no swap terms), `s3_full` (encoder trained); ends with
    `stage3_transport.probes` and `.recovery` on s1_acdc + the three.
  - `20517215` Stage II from `s1_acdc` (afterok 20517212, 6 h): `s2_tcti` (method), `s2_aug`
    (both renderings, no separation terms), `s2_inv` (invariance only), `s2_xrec`
    (cross-reconstruction only); ends with `stage2_disentangle.probes` -> G_sep.
- **Budget correction (user's question).** The 10k x 32 configuration every ledger comparison
  used is 320k clips = **0.039 epoch** of the 8,191,045-clip training mixture (29 source
  entries, 25 complete).  UniVLA's LAM config is batch 64 x 8 GPUs x 20 epochs.  So the jobs
  above are the CONTROLLED PILOT (all arms, equal budget, same as the legacy numbers), not a
  formal pretraining.  Formal budget added: `scripts/run_long.sh`, 128k steps x batch 64 = 1.0
  epoch, warm-up 4000, checkpoint every 5000 steps with `--resume` (train loop now saves
  optimiser + step and slices the deterministic batch list).  Submitted: `20517276`
  s1_acdc_1ep and `20517277` s1_mean_1ep (the pooled control at the same budget, parallel);
  `20517278` s3_latp_1ep (64k x 64 = 0.5 epoch, decoder-only, afterok s1_acdc_1ep) and
  `20517279` s2_tcti_1ep (64k x 64, afterok s1_acdc_1ep).  Measured pilot speed 0.08 s/step at
  batch 32 (frozen DINO, 34M-param model), so 1 epoch is ~6-10 h if the frame reads scale;
  walltime 30 h.  Still ~1/20 of UniVLA's total compute, on one GPU; that ratio goes in the
  paper, not hidden.
- **Pre-registered readings.** Stage I: `s1_acdc` z must beat `s1_mean` / `s1_dc` z on seg1 /
  prof / seq at equal or lower net cost (all 256-d vs 128-d: report both raw and PCA-matched);
  the dimension-matched frozen table (DC+AC1 +0.070 seg1 over DC at 256-d) is the ceiling
  direction.  Stage II: G_sep(s2_tcti) > G_sep(s2_aug) and > G_sep(univla_official) on the
  same clips; N_TC must fall relative to s1_acdc's z at no S_TC cost.  Stage III: s3_latp must
  reproduce the legacy pattern on the new tokenizer -- transfer R2 and cosine above s3_ctrl,
  clean identical to s1_acdc, s3_full below s3_latp on clean.

### 2026-09-09 — Stage I pilot (job 20517212, 4 tokenizers, 10k x 32) + UniVLA official on the same clips (20517213)

- **What ran.** `s1_acdc` (DC+AC1, 8 cells, z 256-d), `s1_dc` (DC only, 4 cells, 128-d), `s1_mean`
  (mean-pooled, 128-d), `s1_acdc3` (DC+AC1+AC2, 384-d); identical data, seed, steps, decoder.
  Val dump = 11,488 clips (400 per source, paired), 29 sources with a probe.  UniVLA released
  stage-2 encoded on the SAME clips as K=4 consecutive stride-4 pairs; its 4 z_q codes are
  concatenated (2048-d, 16 VQ codes of 16 entries = 64 bits).  Probes: per-source ridge from
  z to net xyz, seg1 (first segment), prof (sequence minus its mean, K*3), seq (K*3).

  | code | dims | R2 net | R2 seg1 | R2 prof | R2 seq | patch drop |
  |---|---|---|---|---|---|---|
  | s1_mean z | 128 | **0.375** | 0.263 | 0.071 | 0.238 | 25.3% |
  | s1_dc z | 128 | 0.371 | 0.254 | 0.064 | 0.240 | 25.2% |
  | **s1_acdc z** | 256 | 0.344 | **0.300** | **0.129** | 0.259 | 26.6% |
  | s1_acdc3 z | 384 | 0.329 | 0.271 | 0.136 | 0.250 | 22.9% |
  | UniVLA official, 4 pair codes | 2048 | 0.332 | **0.326** | **0.200** | **0.304** | **17.6%** |
  | frozen DC cells @ PCA-256 (ceiling) | 256 | 0.391 | 0.253 | 0.077 | 0.252 | -- |
  | frozen DC+AC1 cells @ PCA-256 (ceiling) | 256 | 0.388 | 0.323 | 0.119 | 0.279 | -- |

- **Reading (provisional, pilot budget = 0.039 epoch).**
  (i) The Preserve effect exists and is a TRADE-OFF, not a free gain: against the pooled
  tokenizers at equal budget, DC+AC1 gives +0.04 seg1, +0.06 prof (1.9x), +0.02 seq and costs
  -0.03 net.  mean and DC are identical within 0.01, as the basis predicts (same information).
  A third mode buys +0.007 prof and loses on everything else -- one extra mode is the operating
  point, consistent with the frozen-feature table.
  (ii) Against its own linear ceiling (DC+AC1 cells at 256-d) the learned z is -0.044 net,
  -0.023 seg1, +0.010 prof, -0.020 seq: at this budget the tokenizer has not yet reached what a
  PCA of its input gives.  The 1-epoch runs (20517276/7) are the test of whether that is budget.
  (iii) **UniVLA's released codes beat every pilot tokenizer on every shape target** (seg1
  0.326, prof 0.200, seq 0.304) and on patch drop (17.6% vs 25%), at lower net (0.332).  This
  is not a pooled clip code: UniVLA is applied PER PAIR, so its 4 codes are a per-segment
  sequence and never discarded the temporal axis.  Consequence for the framing: "pooling
  destroys temporal information" is a statement about clip-level LAMs (mean-pooled / endpoint
  codes), not about UniVLA's pair-wise use; the Stage-I comparison the paper can make against
  UniVLA is bitrate-matched (64 bits vs a 256-d continuous code) or budget-matched (their full
  training vs our 0.039 epoch), and neither is in hand yet.  The pooled tokenizers `s1_mean` /
  `s1_dc` are the controlled Stage-I baseline; UniVLA is the field reference, not the control.
  (iv) Time reversal: no tokenizer is reversal-equivariant (rev_cos on the sign-flipping DC
  cells 0.64-0.70, UniVLA -0.26; the net reader applied to reversed codes is off-scale, R2 ~ -10),
  so the AC/DC algebra is a property of the cells, not (yet) of z.  Recorded, not claimed.
- **Bug found by the UniVLA dump.** Quantised codes with unused entries give constant
  columns; the ridge's per-column standardisation (sd + 1e-6) then blew up any test row where
  the column moved (G_sep -7e11).  Fixed in `common/probes.py`: columns constant on the
  training rows are dropped (sd -> inf).  Continuous inputs are unaffected; UniVLA's G_sep is
  being recomputed.
- **Status of the other jobs.** Stage III pilot (20517214) running: s3_latp and s3_ctrl done,
  s3_full in progress; Stage II pilot (20517215) pending; 1-epoch runs `s1_acdc_1ep` /
  `s1_mean_1ep` at 98k / 128k steps (0.11 s/step at batch 64, 3 h in, ~1 h left), then
  s3_latp_1ep and s2_tcti_1ep start by dependency.

### 2026-09-09 — Stage III pilot on the new tokenizer (job 20517214): swap effect replicates; "frozen beats fine-tuned encoder" does NOT

  | arm (from s1_acdc, +10k steps) | transfer R2 | cos | calibrated | clean | replay | swap_sens |
  |---|---|---|---|---|---|---|
  | s1_acdc (tokenizer, no Stage III) | 0.261 | 0.465 | 0.319 | 0.344 | -0.02 | 0.36 |
  | s3_ctrl (decoder-only, no swap terms) | 0.268 | 0.460 | 0.309 | 0.344 | -0.02 | 0.38 |
  | **s3_latp (frozen encoder + swap)** | **0.354** | **0.511** | 0.356 | 0.344 | -0.19 | 0.74 |
  | s3_full (encoder fine-tuned + swap) | 0.360 | 0.517 | 0.360 | 0.348 | -0.21 | 0.84 |

- **Confirmed.** The swap objective is what moves transfer: +0.09 R2 / +0.05 cos over the
  decoder-only control on a second, independently trained tokenizer (same pattern and same
  magnitude as legacy rec_src -> swap_fz).
- **Retracted in part.** The legacy clean cost of training the encoder (0.342 -> 0.294) does
  not reproduce when the encoder is FINE-TUNED from a rec-trained tokenizer (0.344 -> 0.348) --
  the legacy full swap was trained jointly from scratch.  Fine-tuned encoder + swap is equal or
  slightly better than frozen on every transfer number (differences 0.004-0.006, single seed,
  within the 0.003-0.010 seed spread measured yesterday).  So LATP's "frozen tokenizer" is a
  modularity / compute statement (decoder-only, 0.06 s/step, tokenizer reusable), not a quality
  advantage over fine-tuning; the paper must say so.  The mechanism finding stands: the gain is
  in the operator, since freezing the encoder loses nothing.

### 2026-09-09 — Stage I at 1 epoch (jobs 20517276/7, 128k x 64): the shape gain persists, the net cost shrinks

  | code (z probes, held-out episodes) | dims | R2 net | R2 seg1 | R2 prof | R2 seq | patch drop |
  |---|---|---|---|---|---|---|
  | s1_mean pilot (0.039 ep) | 128 | 0.375 | 0.263 | 0.071 | 0.238 | 25.3% |
  | s1_acdc pilot (0.039 ep) | 256 | 0.344 | 0.300 | 0.129 | 0.259 | 26.6% |
  | s1_mean_1ep | 128 | **0.379** | 0.261 | 0.073 | 0.248 | 28.8% |
  | **s1_acdc_1ep** | 256 | 0.367 | **0.309** | **0.135** | **0.271** | 24.2% |
  | UniVLA official (4 pair codes) | 2048 | 0.332 | 0.326 | 0.200 | 0.304 | 17.6% |

- **Reading (provisional, single seed).** 25x more data moved the AC/DC tokenizer +0.023 on
  net and +0.01 on every shape target; the pooled tokenizer moved +0.004 / ~0.  The net gap
  to pooled shrank from -0.031 to -0.012 while the shape gains held (+0.048 seg1, +0.062 prof,
  +0.023 seq).  The AC/DC z is now above its pilot linear ceiling on prof and within 0.02 of it
  on net and seq.  UniVLA's per-pair sequence still leads on seg1 / prof / seq; the
  bit-matched pair-wise tokenizer (job 20533212) is the comparison that settles whether that
  is the unit or the method.  s3_latp_1ep / s2_tcti_1ep start from s1_acdc_1ep by dependency.
- **Probe robustness (two more fixes, `common/probes.py`).** (a) R2 returns nan on a constant
  target: austin_buds' test split is ONE episode and the patch nuisance is per-episode, which
  produced G_sep -7e11; Stage II now requires >= 5 test episodes.  (b) standardised features
  are clipped to +-10 at fit and read time: a UniVLA TI code entry seen once in training gave
  S_TI -5e6 on dobbe.  Continuous-code numbers are unaffected (no standardised value near 10).

### 2026-09-09 — UniVLA rows are the RELEASED weights; that comparison is not budget-matched (user's objection, accepted)

- Every "UniVLA official" row above is `cache/univla/lam-stage-2.ckpt` (authors' training: their
  mixture, 8 GPUs x 20 epochs) applied to our val clips.  Our tokenizers are 0.039 epoch
  (pilot) or 1 epoch (128k x 64) on one GPU.  Those rows are a field reference and must not be
  read as a method comparison; the ledger entries that put them in the same table now carry
  this note.  The only controlled comparisons so far are our own arms against each other.
- **Budget-matched UniVLA submitted** (`claude_survey/run_univla_1ep.sh`, job 20537284):
  UniVLA stage-2 recipe verbatim (`rala2.py --anchor none --T 2`, stage-1 codebook inherited
  and frozen, 200M-param encoder/decoder from scratch) on our mixture at 128k steps x batch 64,
  the same optimiser budget as s1_acdc_1ep / s1_mean_1ep.  Caveat recorded: a UniVLA step
  sees 64 pairs (128 frames), ours 64 five-frame clips; steps x batch are matched, frames are
  not.  The existing `s2mix_base` (10k x 16) is the pilot-level version of the same.  When
  20537284 finishes: `baselines/univla.py --arm s2mix_1ep`, then Stage I / II probes and the
  LIBERO offline + closed-loop arms on it -- the paper's UniVLA numbers come from that run,
  with the released weights reported separately as the reference.

### 2026-09-09 — Stage II pilot (job 20517215): NULL as designed; the metric had to be rebuilt first

- **Metric repair.** The registered G_sep read nuisance R2 per source with a 4-d target
  (displacement + mean position).  Diagnosis on the s2_tcti dump: (a) position targets are not
  linearly readable from any code (R2 ~ 0 everywhere); (b) a per-source reader (~270 rows from
  ~130 episodes, per-episode target) reads nothing even from frozen features (ref -0.21); the
  patch law is identical in every source, so the nuisance reader is now GLOBAL (one
  episode-disjoint split, 8k train rows) on motion targets only (net displacement + per-step
  velocity, 10-d).  The action half stays per source.  Clean-rendering reads (`N_*_clean`) are
  the zero check and sit at -0.02.
- **Result (rebuilt metric, pilot 10k x 32, from s1_acdc):**

  | code | S_TC | S_TI | N_TC (patch motion in z) | N_TI (in u) | G_sep | patch drop |
  |---|---|---|---|---|---|---|
  | s1_acdc (no Stage II) | 0.319 | -0.01 | 0.138 | 0.00 | 0.193 | 33.6% |
  | s2_aug (both renderings, no terms) | 0.326 | -0.02 | **0.228** | -0.01 | 0.116 | 37.7% |
  | s2_inv (z-invariance only) | 0.340 | -0.04 | 0.194 | -0.01 | 0.174 | 31.5% |
  | s2_xrec (cross-recon only) | 0.332 | -0.03 | 0.206 | -0.01 | 0.143 | 31.0% |
  | s2_tcti (both) | 0.341 | -0.03 | 0.169 | -0.01 | 0.193 | 28.8% |
  | UniVLA released z_q / z_q_unc | 0.306 | -0.02 | **-0.018** | -0.01 | **0.332** | 22.3% |

- **Reading.** (i) The intervention did not move the nuisance into u: u (8-d) reads the patch
  motion at 0 in every arm, and training on the patch renderings put MORE patch motion into z
  (0.138 -> 0.228 with no terms; the invariance term brings it back to 0.17-0.19, not below
  the untreated tokenizer).  The design is a null at pilot scale, for a structural reason (an
  8-d u cannot absorb a 10-d nuisance the decoder must reproduce), not a budget one; the
  1-epoch s2_tcti_1ep (20517279) was cancelled rather than spend 5 GPU-hours on it.
  (ii) UniVLA's DEPLOYED code carries no patch motion (N_TC -0.02) while its TI branch is dead
  (N_TI -0.01): its nuisance invariance comes from the quantiser, not from the TI head
  (consistent with the 2026-08 audits: pre-VQ emb 0.575 / 0.335, post-VQ z_q 0.12 / 0.003).
  On this nuisance UniVLA's shipped code is cleaner than every continuous code of ours; the
  paper's Stage-II statement must be "the TI branch does not separate" (true: N_TI = 0 for
  them and for us), not "the deployed code carries nuisance" (false for them on this test).
- **Redesign registered and submitted (job 20539655, from scratch, 10k x 32):** `s2_tcti_u64`
  and `s2_aug_u64` (u 8 -> 64 dims, same two terms vs none).  If u64 still reads 0, the next
  arm is a quantised z (VQ per cell) with the same intervention, i.e. the mechanism UniVLA has
  by accident, applied on the AC/DC cells on purpose.

### 2026-09-09 — R2 CONVENTION CORRECTED (affects every absolute R2 in this ledger); pair-wise tokenizer ablation (job 20533212)

- **The bug.** `_r2` averaged 1 - SSE/(SS_tot + 1e-9) over target dims.  For a dim that is exactly
  constant in a source (z of the 2-D navigation sources berkeley_gnm_*, language_table; z of
  several others) the ridge predicts the constant exactly, SSE = 0, and that dim scores 1.0.
  Every per-source mean over (x, y, z) in this ledger therefore carried up to +0.33 from a
  dim with no information.  Fixed (`common/probes.py`): R2 is averaged over the dims that vary
  on the evaluated rows; a source with no varying dim is nan and excluded from the aggregate.
  All ARM-vs-ARM deltas were computed under one convention and survive (checked below); the
  absolute levels move down by 0.03-0.05.  Legacy-convention numbers above are marked by
  their date; from here on every R2 is the corrected one.  Stage-III numbers (transfer,
  clean) are being recomputed under the same convention.
- **Stage I, corrected convention (z probes, held-out episodes):**

  | code | dims | budget | R2 net | R2 seg1 | R2 prof | R2 seq | patch drop |
  |---|---|---|---|---|---|---|---|
  | s1_mean | 128 | 0.039 ep | **0.332** | 0.212 | 0.016 | 0.190 | 29.4% |
  | s1_dc | 128 | 0.039 ep | 0.328 | 0.202 | 0.007 | 0.192 | 29.7% |
  | s1_acdc (DC+AC1) | 256 | 0.039 ep | 0.301 | **0.254** | **0.076** | **0.213** | 32.3% |
  | s1_acdc3 | 384 | 0.039 ep | 0.286 | 0.226 | 0.084 | 0.203 | 28.9% |
  | **s1_pair16_x4** (T=2 tokenizer per segment, 4 codes concatenated) | 256 | 0.039 ep | 0.226 | 0.193 | 0.083 | 0.177 | 37.8% |
  | s1_pair_x4 (same, d=32) | 512 | 0.039 ep | 0.228 | 0.207 | 0.095 | 0.192 | 29.4% |
  | s1_mean_1ep | 128 | 1 ep | **0.336** | 0.210 | 0.017 | 0.200 | 33.6% |
  | s1_acdc_1ep | 256 | 1 ep | 0.324 | **0.264** | **0.085** | **0.224** | 28.9% |
  | UniVLA released (4 pair codes; NOT budget-matched) | 2048 | authors' | 0.288 | 0.281 | 0.152 | 0.259 | 21.2% |
  | legacy rec_src (= s1_acdc recipe, seed 0) | 256 | 0.039 ep | 0.298 | 0.250 | -- | -- | 28.7% |

- **Reading (provisional, single seed each).** (i) The user's question -- "just use pairs" --
  is answered in-house: the SAME architecture, data and budget used pair-wise (UniVLA's unit)
  loses to the DC+AC1 clip code on every target at equal dims (256-d: net -0.075, seg1 -0.061,
  seq -0.036, prof +0.007) and even to the pooled clip code on net (-0.106).  A 4-frame pair
  is a low-SNR unit; the clip is the right unit, and one AC mode is what keeps the shape.
  UniVLA's released codes still lead on shape targets, so the remaining candidates for that
  lead are budget (their 20 epochs x 8 GPUs) and backbone (256 patches, 200M) -- the
  budget-matched UniVLA (20537284) tests the first.  (ii) The Preserve trade-off stands under
  the corrected convention: at 1 epoch DC+AC1 vs mean = net -0.012, seg1 +0.054, prof +0.068,
  seq +0.024.  (iii) Patch drop is 29-38% for every continuous code of ours vs 21% for
  UniVLA's quantised code, the same fact Stage II found.

### 2026-09-09 — Stage III under the corrected R2 convention (deltas unchanged)

  | arm | transfer R2 | cos | calibrated | clean |
  |---|---|---|---|---|
  | rec_src / rec_fz | 0.204 / 0.210 | 0.449 / 0.448 | 0.248 / 0.249 | 0.298 |
  | swap_fz s0 / s1 / s2 (LATP on rec_src) | 0.310 / 0.306 / 0.304 | 0.512 / 0.493 / 0.496 | 0.313 / 0.297 / 0.312 | 0.298 |
  | swap_src (full, from scratch) | 0.262 | 0.492 | 0.291 | 0.248 |
  | grank_g | 0.274 | 0.451 | 0.264 | 0.315 |
  | s1_acdc / s3_ctrl / s3_latp / s3_full | 0.212 / 0.220 / 0.311 / 0.318 | 0.465 / 0.460 / 0.511 / 0.517 | 0.270 / 0.260 / 0.312 / 0.317 | 0.301 / 0.301 / 0.301 / 0.305 |

  LATP - tokenizer = +0.10 transfer R2 on both tokenizers (legacy convention gave +0.10 too);
  cosines are convention-free and unchanged.  These are the Stage-III numbers to quote.

### 2026-09-09 — Stage III at the formal budget (job 20517278, s3_latp_1ep: 64k x 64 on frozen s1_acdc_1ep)

  | | transfer R2 | cos | calibrated | clean | replay | swap_sens |
  |---|---|---|---|---|---|---|
  | s1_acdc_1ep (tokenizer) | 0.244 | 0.484 | 0.305 | 0.324 | -0.09 | 0.43 |
  | **s3_latp_1ep** | **0.319** | **0.524** | 0.329 | 0.324 | -0.57 | 1.45 |

  LATP +0.075 R2 / +0.040 cos at zero clean cost -- the third tokenizer on which the swap
  objective moves the independent reader (pilot rec_src +0.10 / +0.06; pilot s1_acdc +0.10 /
  +0.05).  Highest absolute transport number so far (0.319 / 0.524).  swap_sens 1.45 is
  dominated by clips with near-zero ||dF|| (ratio readout), not a pathology; MMD sits at
  -0.007 (matched).  Single seed.  Corrected R2 convention.

### 2026-09-09 — directory move and clean-up (user)

- `claude_survey/` and `codex_survey/` now live under `latent_free/` (user moved them 23:14).  Every
  hardcoded path fixed (`LF` = parent dir in the 8 live modules; absolute paths in all run scripts;
  `acdc_la.paths.CLAUDE_SURVEY`; pair-cache symlinks); UniVLA budget-matched job resubmitted as
  `20543308`; accuracy jobs unaffected (they import through `paths`).  Nested repos are ignored by
  `latent_free/.git`.
- Deleted from `claude_survey`: 132 checkpoints (102 GB) = all `smoke_*` and the closed stage-2
  lines (RALA-2 anchors, TI-SSL variants, transport v1, codebook sweeps, language arms); v1
  mixed-batch readout arrays (superseded); SAM3 weights of the killed background-gate line
  (3.3 GB).  Kept: `s2_base`, `s2mix_base`, `s2_vae`, `s2_vae_olaf`, `s2_ubal_t01`, all non-smoke
  `swaplam_*` checkpoints and arrays, every json/csv.  131 GB -> 52 GB.  `acdc_la`: smoke
  checkpoints/dumps removed.  `codex_survey` (167 MB, the DCT pooling audit that seeded AC/DC)
  kept whole.  `challenge/` (ORena project): user decided not to delete; instead the project was gathered under `challenge/`
  (`lapchole-focus-vqa` 91 GB and `data/orena_focus` moved in, symlinks left; new `challenge/claude_survey/`
  workspace; `challenge/README.md` maps the layout).

### 2026-09-10 — review accepted: (1) u collapsed by construction; (2) downstream must reuse the LATP decoder; first accuracy numbers

- **u never carried anything, anywhere.** Every Stage-I / II / III log shows kl_u = 0.0000: with beta_u = 0.1
  vs beta_z = 2e-4 (a 500x cost ratio) the u posterior sits on the prior in all 20 runs, including the running
  `s2_tcti_u64`.  So N_TI = 0 in the Stage-II table is not "routing failed", it is "the channel was priced
  out".  The 8 -> 64 experiment cannot answer the question; submitted (job 20545431, pilot, from scratch):
  `s2_tcti_bu` / `s2_aug_bu` (d_u 8, beta_u 2e-4) and `s2_tcti_u64_bu` / `s2_aug_u64_bu`.  Reading rule: first
  kl_u must leave zero and shuffling u must change the decoder output; only then is G_sep informative.
  Consequence for Stage III: u_j in the swap decode was the prior mean all along; LATP results are unaffected
  (the residual was never used) but the paper's description of u must say so.
- **Downstream evaluation had a design hole (review sec. 6).** The LIBERO heads read z only; a frozen-tokenizer
  LATP run has z identical to its Stage-I parent, so the pipeline could not show any LATP effect by
  construction.  Added `+T` arms: the action head reads the transport decoder's hidden state
  h = D_hidden(f0, c = 0, z, u = 0) (quadrant-pooled 16 state tokens, 3072-d); u is the prior mean, never the
  real future residual; same z through the Stage-I decoder vs the LATP decoder isolates LATP's downstream
  contribution.  Offline (`oracle+T`, `chain+T`) and closed-loop (`<tag>+T`).  Jobs 20545455 (ai) /
  20545456 (gpu): tags s1_acdc_1ep, s3_latp_1ep, s1_mean_1ep.
- **First accuracy numbers (job 20541459, offline, LIBERO-spatial, held-out episodes, heads on frozen DINO
  g4 + task one-hot; action token accuracy = 256 uniform bins over the train 1-99% range; chain = policy ->
  latent -> action, the deployable path):**

  | latent | latent_acc (majority) | tok oracle | tok chain | L1 chain (standardised) |
  |---|---|---|---|---|
  | no latent (g4 + task -> chunk) | -- | 0.078 | -- | 0.434 |
  | s1_acdc_1ep (256-d) | 0.517 (0.095) | 0.088 | 0.091 | 0.421 |
  | s1_mean_1ep (128-d) | 0.634 (0.099) | 0.086 | 0.088 | 0.423 |
  | s1_acdc pilot | 0.521 (0.094) | 0.087 | 0.093 | 0.424 |
  | UniVLA released (2048-d, 16 VQ tokens) | 0.492 (0.314) | 0.087 | 0.097 | 0.430 |

  Reading (provisional): every latent beats the no-latent floor by a small margin (L1 0.434 -> 0.421-0.430;
  token accuracy 0.078 -> 0.086-0.097) and no latent separates from another at this resolution -- the
  differences are 0.002-0.01, below what a single split can resolve.  latent_acc is not comparable across
  tokenizers (different quantisers, different class entropy: UniVLA's majority is 0.31, ours 0.09); it is
  reported as UniVLA's monitoring number, not as a ranking.  These are z-only heads; the +T job is the one
  that can show a Stage-III effect.  Closed-loop success (5 held-out inits x 10 tasks) is running in the
  same job.

### 2026-09-10 — Stage II u64 at beta_u = 0.1 (job 20539655): identical to u8, as predicted by the collapse

  | arm | S_TC | S_TI | N_TC | N_TI | G_sep | kl_u |
  |---|---|---|---|---|---|---|
  | s2_tcti_u64 | 0.325 | -0.04 | 0.169 | -0.007 | 0.187 | 0.0000 |
  | s2_aug_u64 | 0.308 | -0.04 | 0.228 | -0.007 | 0.110 | 0.0000 |

  **RETRACTED 2026-09-10 06:40: these were NOT 64-d.** `--init-from s1_acdc` rebuilds the model from the
  checkpoint (d_u = 8) and the `--d-u 64` flag was silently ignored (ckpt args say 64, tensor says 8);
  the numbers are identical to the 8-d arms because they ARE the 8-d arms.  `run_stage2.sh` also
  turned INIT="" back into s1_acdc.  Both fixed (assert in train_loop; `${INIT-...}`); genuine
  from-scratch u64 arms resubmitted (job 20550460).  The bug class is the ledger's own R8.

### 2026-09-10 — offline +T (decoder reuse) accuracy, job 20545455: LATP's decoder adds nothing downstream on this proxy

  | latent | tok chain (z head) | L1 chain (z head) | tok chain+T (decoder-hidden head) | L1 chain+T |
  |---|---|---|---|---|
  | s1_acdc_1ep (Stage-I decoder) | 0.091 | 0.421 | 0.094 | 0.426 |
  | s3_latp_1ep (LATP decoder, same z) | 0.091 | 0.421 | 0.090 | 0.425 |
  | s1_mean_1ep | 0.088 | 0.423 | 0.088 | 0.429 |
  | no latent | 0.078 | 0.434 | -- | -- |

- **Reading.** Routing the policy's latent through the transport decoder's hidden state (u = prior mean,
  c = 0) does not improve action prediction over the raw-z head, and the LATP decoder is
  indistinguishable from the Stage-I decoder (0.090 vs 0.094 token accuracy, 0.425 vs 0.426 L1).  So the
  +0.10 transfer R2 that LATP gives on the independent reader does NOT show up as better offline action
  prediction on LIBERO-spatial with these heads.  Null, single split; candidate reasons to test rather
  than assert: (a) the action head already gets f0, so the decoder's state-conditioning is redundant for a
  fixed-scene benchmark; (b) LATP's gain is in cross-context realisation, and LIBERO-spatial's held-out
  episodes are the same scene; (c) c = 0 / u = 0 at deployment.  Closed-loop success (jobs 20549334 /
  20549335, after adding rotary_embedding_torch to the simulator env) is the remaining accuracy number.

### 2026-09-10 — Stage II with beta_u = 2e-4 (job 20545431, from s1_acdc, d_u = 8): u is now USED, and still does not carry the nuisance

  | arm | kl_u (nats) | S_TC | S_TI | N_TC | N_TI | G_sep | patch drop |
  |---|---|---|---|---|---|---|---|
  | s2_tcti_bu (both terms) | **13.4** | 0.326 | -0.03 | 0.174 | -0.009 | 0.171 | 26.5% |
  | s2_aug_bu (no terms) | 0.06 | 0.309 | -0.03 | 0.227 | -0.007 | 0.105 | 36.6% |
  | (beta_u = 0.1 versions) | 0.00 | 0.325 / 0.308 | -0.04 | 0.169 / 0.228 | -0.007 | 0.187 / 0.110 | 27.5 / 38.4% |

- **Reading.** Lowering the price made the intervention terms put 13 nats into u (the augmentation-only
  control leaves u empty even at the low price: without the separation terms nothing needs u).  But a
  linear reader still recovers no patch motion from u (N_TI -0.009), and z still leaks it (0.174, above the
  untreated tokenizer's 0.138).  So the KL price explained WHY u was empty, and removing it was
  necessary, but what u now holds is not linearly the patch motion.  Open questions, in order: is u's
  13 nats used by the decoder at all (shuffle-u test, registered), is 8-d too small once the price is
  fair (true u64 from scratch, job 20550460), and is a linear reader the right instrument for a 10-d
  nonlinear nuisance in an 8-d code.  The "u64" rows reported at 01:00 were invalid (see retraction).

### 2026-09-10 — MULTI-EPOCH PRETRAINING LAUNCHED (user: deadline ~1 week; pretraining is the priority)

- Submitted the formal chain: `s1_acdc_10ep` = 1.28M steps x 64 = **10 epochs** of the 8.19M-clip mixture
  (job 20554021, 48 h walltime, ~39 h at 0.11 s/step; resume leg 20554022 continues from the 5000-step
  checkpoints if the walltime cuts it) -> `s3_latp_10ep` = 640k x 64 (5 epochs, decoder-only, frozen
  tokenizer; job 20554023, afterok).  Equal-budget pooled baseline `s1_mean_10ep` (20554024 + resume leg
  20554025) in parallel.  Stage II is NOT in the chain (no evidence yet; its true-u64 pilot 20550460 was
  cancelled to keep the queue for pretraining and accuracy).  Still running: budget-matched UniVLA
  `s2mix_1ep` (20543308).  Still queued: closed-loop success (20552918 / 20552919); CPU fallback running
  on the login node.
- What can and cannot be compared to UniVLA's paper (user's question): their LIBERO / CALVIN numbers are
  a 7B VLA fine-tuned with their latent tokens; our small frozen-DINO policy cannot produce numbers on
  that scale.  Two comparisons are legitimate: (a) same small policy, their released codes vs ours (the
  current accuracy table); (b) plugging our tokenizer into their VLA fine-tuning (`vla-scripts/
  finetune_libero.py`) -- the head-to-head the paper needs, ~1 GPU-week, needs univla-7b weights and
  LIBERO RLDS, and a discrete version of our code (k-means per cell or a VQ tokenizer).  Not started;
  awaiting the user's go.

### 2026-09-10 — CLOSED-LOOP SUCCESS RATE, LIBERO-spatial (job 20552918): all arms 8-18%, no separation

  Protocol: 10 tasks x 5 held-out initial states (the demo split's val demos; the policy never saw these starts),
  220-step horizon, policy = latent head + action head on frozen DINO g4 + task one-hot (2000-step MLPs),
  re-plan every 4 steps.  50 episodes per arm.

  | arm | success |
  |---|---|
  | s1_acdc_1ep (z head) | 0.18 (9/50) |
  | s1_acdc_1ep +T (decoder-hidden head) | 0.12 |
  | s3_latp_1ep (same z) | 0.16 |
  | s3_latp_1ep +T (LATP decoder) | 0.08 |
  | s1_mean_1ep | 0.18 |
  | s1_mean_1ep +T | 0.12 |
  | UniVLA released codes | 0.18 |
  | no latent (direct BC) | 0.14 |

- **Reading.** With 50 episodes the binomial 95% interval at 0.16 is +-0.10, so every arm is within noise of
  every other; the two arms with IDENTICAL policies (s1_acdc_1ep vs s3_latp_1ep: same z, same head config)
  differ by 0.02, which is the floor.  The +T arms are consistently lower (0.08-0.12 vs 0.16-0.18) --
  routing through the decoder hurts here.  Absolute level (~15%) is what a frozen-feature MLP policy
  achieves on LIBERO-spatial; UniVLA's paper number (~96%) is a fine-tuned 7B VLA and is not comparable.
  Conclusion for the report: on this downstream stack, none of the tokenizers (ours, pooled, UniVLA's)
  improves over direct behaviour cloning beyond noise.  What would make this test informative: 50 starts
  per task (10x episodes), a stronger policy (VLA fine-tune with our tokens), and the multi-epoch
  tokenizers.
- **Data decision (user).** The 5,000-episode cap per source was never approved; UniVLA trains on full
  datasets (+ Ego4D).  Uncapped conversion started for every capped source (jobs 20557364-8: fractal,
  bc_z, gnm_recon/cory_hall/fmb/furniture/dobbe, kuka, language_table -> cache/oxe_full) and the OXE
  Bridge re-encode (`bridge_oxe`, 416 GB, 1538 files) is downloading at ~600 MB/s (the RAIL host of our
  97-shard Bridge copy is gone).  The loader takes `ACDC_FULL_DATA=1` to use the uncapped caches; the
  10-epoch chain will be resubmitted on the full data once conversions finish.  Ego4D deferred (user).

### 2026-09-10/11 — evaluation funnel fixed (PROTOCOL.md); in-training proxy curve; full-data pipeline status

- **Protocol (user's funnel, adopted).** `acdc_la/PROTOCOL.md`: Tier 0 health -> Tier 1 representation probes ->
  Tier 2 structural/counterfactual -> Tier 3 cheap policy-facing -> Tier 4 closed-loop, with numeric
  promotion criteria fixed in advance (Stage I: dprof >= +0.03, dseg1 >= +0.02, dnet >= -0.02, N_TC <= baseline
  + 0.02; Stage II: N_TC <= untreated - 0.05 and N_TI >= 0.05; Stage III: transfer >= control + 0.05 R2 /
  +0.03 cos, clean unchanged; Tier 3->4: few-label 10% >= baseline and chain L1 <= baseline - 0.01).
  Status: Stage I passed Tier 1; Stage III passed Tier 2, failed Tier 3; Stage II failed Tier 1.
- **In-training proxy curve** (`common/curve.py`): every 10% of the budget, a 100-clip-per-source dump and
  Tier 0/1 numbers (rec, kl_z, effective rank, swap sensitivity, net/seg1/prof/seq, patch drop, N_TC) appended
  to `results/<tag>_curve.json`; non-fatal; verified end to end on a CPU smoke (`cpu_curve`).  The queued
  full-data runs pick it up automatically (default eval-every = steps/10 for runs >= 200k steps).  Stop
  rule per PROTOCOL: profile gap <= 0 at 25% and not growing at 50% -> kill.
- **Budget-matched UniVLA stage-2 done** (`s2mix_1ep`, 128k x 64, 13.8 h, codebook 16/16 used, TI branch 1/16 as
  in the released weights).  Dump + Stage I/II probes queued (job 20567607).
- **Full-data conversion running.** fractal: 86,600 episodes / 3.76M frames converted in 12 min (5.4k frames/s;
  the .npy wrap of 744 GB is the slow part); bc_z running; gnm/fmb/furniture/dobbe, kuka, language_table,
  bridge_oxe queued.  Full-data chain (`s1_acdc_full`, `s1_mean_full`, `s3_latp_full`) gated on all six.

### 2026-09-11 — THE CONTROLLED UniVLA COMPARISON (job 20567607): same data, same 128k x 64 budget

  | code | dims (bits) | R2 net | R2 seg1 | R2 prof | R2 seq | patch drop | N_TC (patch motion in z) |
  |---|---|---|---|---|---|---|---|
  | **UniVLA stage-2, their recipe, our data, our budget** (`s2mix_1ep`) | 2048 (16 VQ tokens x 4 bits = 64 bits) | 0.236 | 0.222 | **0.125** | 0.215 | **22.9%** | **-0.027** |
  | UniVLA released weights (their data + Ego4D, 8 GPU x 20 ep) | 2048 (64 bits) | 0.288 | 0.281 | 0.152 | 0.259 | 21.2% | -0.018 |
  | **ACDC tokenizer** (`s1_acdc_1ep`) | 256 continuous | **0.324** | **0.264** | 0.085 | **0.224** | 28.9% | 0.106 |
  | pooled tokenizer (`s1_mean_1ep`) | 128 continuous | 0.336 | 0.210 | 0.017 | 0.200 | 33.6% | (0.23 on the aug arm) |

- **Reading (single seed each; corrected R2 convention).** At equal data and optimiser budget the AC/DC tokenizer
  reads the action better than UniVLA's recipe on net (+0.088), first segment (+0.042) and the full sequence
  (+0.009); UniVLA's per-pair code keeps more trajectory SHAPE (profile 0.125 vs 0.085) and is markedly
  cleaner of the moving-patch nuisance (N_TC -0.03 vs +0.11, patch drop 23% vs 29%) -- the quantiser effect
  documented earlier.  Budget matters for UniVLA too: their released weights are +0.05 net / +0.06 seg1
  above the matched retrain, so the earlier "UniVLA official beats us on shape" was partly budget.
- **What this settles for the paper.** The two Stage-I claims that survive a controlled comparison are
  (i) net / segment action readability at 4x fewer dims, and (ii) one temporal mode recovers most of the
  shape a pooled code loses (0.017 -> 0.085) but not all of what a per-segment code keeps (0.125).  The two
  things UniVLA does better are exactly the Stage-II axis (nuisance) -- which strengthens the case that a
  quantised or otherwise nuisance-suppressed z is the missing piece, not a rearranged TI head.
- Not yet: the same comparison on Stage III (their decoder vs LATP; needs the legacy eval_mix transport
  path on s2mix_1ep) and on the LIBERO accuracy table (`libero_offline` hardcodes `official_stage2`; add
  `--univla-arm`).  Multi-seed for all rows.

### 2026-09-11 — decision (user): full-data pretraining will not make the deadline; compare on the SAME capped mixture

- Full-data chain and the two remaining conversions (language_table, bridge_oxe) cancelled.  Eight sources are
  fully converted under `cache/oxe_full/` (fractal, bc_z, gnm_recon, gnm_cory_hall, fmb, furniture_bench, dobbe,
  kuka; ~3 TB) and stay on disk for after the deadline; `ACDC_FULL_DATA=1` switches to them.
- Every comparison for the paper uses the capped mixture (29 sources, 5,000 episodes/source, 8.19M clips):
  ours, the pooled baseline, and UniVLA's recipe (`s2mix_1ep`) all trained on exactly it.  The released
  UniVLA weights are reported as a reference row, not a comparison.
- Multi-epoch chain resubmitted on the capped mixture (12 h legs, resume): `s1_acdc_10ep` (20582822-5),
  `s3_latp_10ep` (20582826/7, after it), `s1_mean_10ep` (20582828-31).  Proxy curve every 10%; PROTOCOL stop
  rules apply.

### 2026-09-12 — paper shape fixed (user): diagnostics + method + UniVLA-on-the-same-data comparison; not "beat their success"

- Comparison rule: every UniVLA number in the paper comes from THEIR recipe trained on OUR capped mixture at OUR
  budget (`s2mix_1ep` = 128k x 64; `s2mix_2ep` = 256k x 64 submitted, job 20617444, ~28 h single job since rala2
  has no resume).  Released weights = reference row only.
- Same-data comparison being completed on all three axes: Stage I probes (done 09-11), Stage III transport
  with THEIR decoder (legacy `eval_mix.py --arm s2mix_1ep`, job 20617443), and the LIBERO accuracy tables with
  their matched latents (`--univla-arm s2mix_1ep`; offline + closed-loop with 20 starts/task = 200 episodes
  per arm, job 20617445).
- Multi-epoch chains: s1_acdc_10ep at 203k steps, s1_mean_10ep at 171k (0.165 s/step with both running;
  ~59 h to 10 epochs).  Every 10% point now also saves a model snapshot, so a matched-budget table can be
  built at 5 epochs (640k) if the deadline requires it.
- Diagnostic content already in hand for the paper: (1) temporal pooling loses trajectory shape (frozen-feature
  subset table + tokenizer table), (2) pair-wise tokenization is the worse unit at equal dims, (3) UniVLA's TI
  branch is dead and its nuisance invariance comes from the quantiser (N_TC -0.03 with N_TI = 0), (4) our
  intervention-based TC/TI fails and why (KL price; then linear unreadability), (5) LATP gain lives in the
  decoder (frozen-encoder test, three tokenizers, three seeds), (6) transfer-error dissection (radial vs
  angular; grank = calibration), (7) R2 convention artefact on constant action dims.

### 2026-09-12 — review's two readouts (job 20617653): the LATP effect is not more action-readable, and predicted z is the bottleneck; u is decorative

  **Downstream through the decoder OUTPUT effect (+E), LIBERO-spatial offline, standardised L1 of the 16-step chunk**

  | model | oracle z -> H | oracle z -> T -> dF^ -> H (+E) | dF^ alone, no f0 | policy z^ -> H (chain) | policy z^ -> T -> dF^ -> H (chain+E) | latent R2 (z^ vs z) |
  |---|---|---|---|---|---|---|
  | s1_acdc_1ep (Stage-I decoder) | 0.344 | 0.345 | 0.346 | 0.421 | 0.425 | 0.50 |
  | s3_latp_1ep (LATP decoder, same z) | 0.344 | 0.346 | 0.347 | 0.421 | 0.426 | 0.50 |
  | s1_mean_1ep | 0.368 | 0.373 | 0.373 | 0.423 | 0.428 | 0.59 |

- **Reading.** (i) Reading the action from the realised effect dF^ gives exactly what reading it from z gives
  (0.345 vs 0.344): the effect carries z's action information and nothing more, and the LATP decoder's
  effect is not more action-readable than the Stage-I decoder's (0.346 vs 0.345).  So on this benchmark the
  transfer proxy (+0.08-0.10 R2 on the independent reader) does not align with action control; the
  review's second branch applies.  (ii) The dominant loss is oracle -> predicted latent: 0.344 -> 0.421
  (+0.08 L1) for every model; the decoder interface moves things by <= 0.005.  The latent policy (frozen
  DINO + task one-hot -> z, R2 0.50) is the bottleneck of the whole chain.  (iii) The Stage-I effect
  survives at the oracle level (ACDC 0.344 vs pooled 0.368) and vanishes after prediction (0.421 vs 0.423).
  Consequence: closed-loop success on LIBERO-spatial cannot separate tokenizers with this policy; the
  paper's LATP evidence stays at Tier 2 (cross-context realisation), with this table as the honest
  boundary.

  **u-shuffle (does the decoder use u?)**, patch rendering, 30 val batches, MSE of dF^ vs dF_patch

  | ckpt | kl_u | own u | shuffled u | u = 0 |
  |---|---|---|---|---|
  | s2_tcti_bu (beta_u 2e-4, both terms) | 13.5 | 1.710 | 1.730 | 1.721 |
  | s2_aug_bu / s2_tcti / s1_acdc | 0.02 / 0 / 0 | 1.711 / 1.714 / 1.763 | identical | identical |

  With the fair price u carries 13 nats that the decoder uses for a 1% reconstruction change; everywhere
  else u is exactly unused.  Decision (review): u is an OPTIONAL private residual, not a contribution; the
  Stage-II principle in the method is z_clean ~ z_patch only.

### 2026-09-13 — Stage IV submitted: portable core + reconstruction residual (review 2026-09-13); 2-epoch probe on the small CPU dump is not usable

- **Method (provisional, no result yet).** The frozen Stage-I code z0 (256) of `s1_acdc_1ep` is split by two learned
  Gaussian heads into a portable core z^P (192) and a residual r (64).  One decoder (initialised from the LATP
  operator `s3_latp_1ep`) serves both paths: self D(s_i, z^P_i, r_i) -> dF_i (rec + 2e-4 KL on both heads), cross
  D(s_j, z^P_i, 0) with the Stage-III swap realism (MMD) + cycle on the core only.  The split is defined by which
  path may consume the information (cross-usable vs reconstruction-only), not by a TC/TI label; the routing
  pressure is that donor-specific content in z^P costs realism if rendered and cycle if not, while r is free of
  both.  Code: `acdc_la/common/pcrr.py`, `acdc_la/stage4_portable/train.py`, `scripts/run_stage4.sh` (commit a3cb367).
- **Jobs.** `s4_pcrr` (10k steps) + control `s3_latp_1ep_cont` (LATP continued 10k steps on the full code, so the
  comparison is not "more decoder steps") + Stage I probes (Z = z^P), Stage II probes (U = r: S_TI / N_TI read the
  residual), Stage III transfer with r = 0: job 20627476 (ai) / twin 20627479 (gpu partition, 16.8 SU left; the
  monitor cancels whichever does not start first).  Policy recipe with `--tokenizer s4_pcrr --transport s4_pcrr`
  (arms ours / ours_noeff offline, ours closed-loop 20 inits): job 20627477, afterok the ai job.  Promotion rule
  added to PROTOCOL.md (leakage N_TC(z^P) <= N_TC(z) - 0.03 with N_TI(r) >= N_TC(z^P); transfer >= control - 0.01;
  action R2 of the core >= z - 0.02; downstream latent R2 > 0.50 and chain L1 down).
- **2-epoch checkpoints.** `s1_acdc_2ep` (step 265k) / `s1_mean_2ep` (260k) copied from the 10-epoch chains after
  leg 1 (both legs ended at ~2 epochs; leg 2 queue estimates 09-15 and 09-19).  The login-node CPU dump had to
  use 100 clips/source (2,880 val clips, ~33 test clips per source): on it the frozen-feature reference itself
  drops from 0.362 to 0.185 R2_net and z reads 0.135, so **nothing from that dump is reportable**; the GPU
  eval (400 clips/source, jobs 20623320 ai / 20623321 gpu, gpu estimate 09-14 08:40) is the one that counts.
  Val reconstruction 2 ep vs 1 ep: 1.466 vs 1.451 (ACDC, different val subsets); in-training curve at 10 % / 20 %
  of the 10-epoch budget: rec 1.460 -> 1.452 (ACDC), 1.589 -> 1.581 (pooled); eff. rank 160 vs 79, swap sens. 0.41
  vs 0.25.  The second epoch moved reconstruction by < 0.01 on both.

### 2026-09-13 — same-data UniVLA on LIBERO-spatial offline (job 20617445): oracle L1 0.377 vs ours 0.344; chain equal; closed-loop part aborted on h004 and was resubmitted

  **LIBERO-spatial offline, held-out episodes, same recipe for every latent (MLP head, 2000 steps); UniVLA = `s2mix_1ep`,
  their Stage-2 trained with their recipe on our capped mixture at the 1-epoch budget**

  | latent | dims | latent_acc (majority) | latent R2 | l1 oracle | l1 chain | tok chain |
  |---|---|---|---|---|---|---|
  | no latent | - | - | - | - | 0.434 | 0.078 |
  | s1_acdc_1ep (ours, Stage I) | 256 | 0.517 (0.095) | 0.500 | **0.344** | 0.421 | 0.091 |
  | s1_mean_1ep (pooled) | 128 | 0.634 (0.099) | 0.593 | 0.368 | 0.423 | 0.088 |
  | univla_s2mix_1ep (same data, same budget) | 2048 | 0.669 (0.468) | 0.519 | 0.377 | 0.426 | 0.087 |

- **Reading.** At the oracle level the ACDC code is the most action-informative of the three (0.344 vs 0.368 pooled
  vs 0.377 UniVLA); after the latent policy every code lands at 0.421-0.426 and the no-latent floor is 0.434, so the
  posterior-to-policy gap erases the ordering (same finding as 09-12).  UniVLA's latent_acc 0.669 is not comparable:
  its majority-code baseline is 0.468 (nearly half of all its tokens are one code on LIBERO), against 0.10 for the
  continuous codes' k-means bins.  Token accuracy (256 bins) is at the floor for everyone (0.08-0.09).
- **Same-data UniVLA transport (job 20617443, their evaluation pipeline, 11,353 val clips):** indep self 0.218,
  transfer 0.116, replay 0.014, frozen reference 0.285; ours (s3_latp_1ep, stage3 probes) transfer 0.319, clean
  reference 0.36.  Provisional until both are recomputed with one reader implementation.
- **Closed-loop part** (6 arms x 10 tasks x 20 inits) died with SIGABRT (exit 134, no Python traceback) on node h004
  right after the orientation check; the same code ran on h006 / h000 / g001 before.  Resubmitted as
  `SKIP_OFFLINE=1 NINIT=20 UARM=s2mix_1ep OUT=libero_matched` on the gpu partition, 3 h, `--exclude=h004`
  (job 20628004); h004 excluded for the queued policy jobs 20623225 / 20627743 / 20627477 as well.
- Files: `acdc_la/results/libero_matched.json`, `acdc_la/logs/acc_20617445.out`.

### 2026-09-13 — policy recipe (job 20623225): offline gains of 0.01 L1 over bc; closed-loop bc 15 % > ours 5 % > same-data UniVLA 1 %

  **LIBERO-spatial, one pipeline for every arm: 4-frame history transformer policy, oracle->predicted curriculum, residual BC head,
  8,000 iters x 256; UniVLA = `s2mix_1ep` codes (same data, same budget); closed-loop 10 tasks x 10 held-out starts per task
  (the split has 10 held-out demos per task, so `--rollout 20` ran 10)**

  | arm | chain L1 | oracle L1 | latent R2 | closed-loop success (100 eps) | per task |
  |---|---|---|---|---|---|
  | bc (no latent) | 0.436 | - | - | **0.15** | 0 0 0 .7 .1 .3 0 0 0 .4 |
  | ours (ACDC z, LATP effect loss) | 0.427 | 0.373 | 0.46 | 0.05 | .1 0 0 .3 0 0 0 0 0 .1 |
  | ours_noeff | 0.429 | 0.367 | 0.44 | not rolled out | |
  | univla same-data | 0.425 | 0.389 | 0.50 | 0.01 | 0 0 0 .1 0 0 0 0 0 0 |

- **Reading.** (i) Offline, the three latent arms end within 0.004 of each other and 0.01 below bc, as with the MLP
  recipe; the history policy + curriculum did not close the posterior-to-policy gap (latent R2 0.44-0.50, same
  as the MLP's 0.50) and the effect-consistency loss changes L1 by 0.002.  (ii) Closed-loop, adding a predicted
  latent HURTS: bc 15 %, ours 5 %, UniVLA 1 %; the whole signal sits in three tasks (3, 5, 9) and bc wins all
  three.  Our latent arm still beats the same-data UniVLA arm (5 vs 1 %) but the statement the paper can make
  is only that; against bc the latent path is a liability on this benchmark with this policy.  (iii) Oracle
  L1 (0.373 / 0.367) is worse than the MLP recipe's 0.344 because the residual latent head is trained mostly on
  predicted z (curriculum ends at p = 0 at 60 % of training).
- **Consequence.** This is the "posterior-to-policy gap" finding again, now closed-loop and with the review's
  recipe; it is the boundary of the LATP / Stage-I evidence, not a Tier-4 pass.  Stage IV (portable core) is the
  one remaining route to a downstream number: it is the only arm that changes what the policy has to predict.
  Files: `acdc_la/results/libero_policy.json`, `acdc_la/logs/policy_20623225.out`.

### 2026-09-13 — 2-epoch checkpoints (job 20623320, 400 clips/source, corrected R2): nothing moves between epoch 1 and 2; the ACDC profile gap holds

  | tokenizer | budget | R2 net | seg1 | prof | seq | patch drop | own-decoder transfer R2 / cos |
  |---|---|---|---|---|---|---|---|
  | s1_acdc_1ep | 1 ep (128k x 64) | 0.324 | 0.264 | 0.085 | 0.224 | 28.9 % | 0.244 / 0.484 |
  | s1_acdc_2ep | 2 ep (265k x 64) | 0.321 | 0.259 | 0.084 | 0.228 | 28.7 % | 0.252 / 0.504 |
  | s1_mean_1ep | 1 ep | 0.336 | 0.210 | 0.017 | 0.200 | 33.6 % | - |
  | s1_mean_2ep | 2 ep (260k x 64) | 0.336 | 0.207 | 0.014 | 0.207 | 36.1 % | 0.188 / 0.427 |
  | univla_matched (s2mix_1ep) | 1 ep | 0.236 | 0.222 | 0.125 | 0.215 | 22.9 % | - |

- **Reading.** Every Stage-I number is within 0.005 of its 1-epoch value on both chains (same 11,456 val clips,
  same reader).  The claim the paper makes on this axis, ACDC keeps the temporal profile and the first segment
  that pooling discards (prof +0.07, seg1 +0.05, net -0.015), is the same at 1 and 2 epochs, so it is not a
  short-training artefact.  Transport without LATP: the 2-epoch ACDC decoder transfers at 0.252 / 0.504 vs the
  pooled 0.188 / 0.427 (the LATP-trained s3_latp_1ep is 0.319 / 0.524).  The 10-epoch chains keep running for the
  4-epoch point (ACDC leg 2 on h000 since 03:03), but there is no sign a longer budget changes the ordering.
  Snapshot of the 2-epoch checkpoints: `acdc_la/ckpt/s1_{acdc,mean}_2ep.pt`; dumps `results/s1_*_2ep_val.npz`.

### 2026-09-13 — Stage IV v1 (job 20627479): the portable core lost action, not nuisance; transfer down; two of three promotion criteria failed

  **Same clips (11,488 val), same readers; z^P = 192-d core, r = 64-d residual on the frozen s1_acdc_1ep code, decoder from
  s3_latp_1ep, 10k steps; control = s3_latp_1ep_cont (same 10k extra decoder steps, no split)**

  | | s4_pcrr: core z^P / residual r | s3_latp_1ep_cont: z / u | s3_latp_1ep: z / u |
  |---|---|---|---|
  | action R2 net from core (S_TC) / from residual (S_TI) | 0.261 / **0.114** | 0.328 / -0.22 | 0.324 / -0.27 |
  | seg1 / prof / seq from core | 0.216 / 0.081 / 0.186 | 0.269 / 0.086 / 0.229 | 0.264 / 0.085 / 0.224 |
  | patch drop of the action probe | 48 % | 29 % | 31 % |
  | N_TC (patch from core) / N_TI (patch from residual) | 0.035 / -0.001 | 0.138 / -0.008 | 0.106 / -0.007 |
  | independent transfer R2 / cos | 0.288 / 0.477 | 0.321 / 0.525 | 0.319 / 0.524 |
  | E_self (self decode with residual = 0) / E_correct / E_wrong | 1.73 / 2.44 / 2.77 | 0.88 / 1.98 / 2.58 | 0.87 / 2.26 / 3.09 |
  | val rec (with residual) | 1.499 | 1.485 | 1.484 (1 ep) |

- **Verdict against PROTOCOL.md.** (a) leakage: N_TC 0.106 -> 0.035 passes the -0.03 bound, but N_TI(r) = -0.001: the
  patch did not move to the residual, it is linearly gone from both (and the action probe's patch drop rose to 48 %,
  so the core is LESS invariant in the sense that matters).  (b) transfer 0.288 vs control 0.321: **fail** (-0.033,
  bound -0.01).  (c) action from the core 0.261 vs 0.324: **fail** (-0.063, bound -0.02).  The residual reads the
  action at 0.114 where u never did (-0.27): the free channel took action content.
- **Why (from the training log).** self_nor (self decode with r = 0) ends at 3.0 vs rec 1.62 with r: the decoder became
  dependent on the residual for the self path, so r absorbed whatever rec wanted, action included.  Nothing pulls
  action INTO the core: rec is satisfied through r, MMD is ~0 from step 2k (the cross distribution is easy to match
  without the donor's action), cycle (0.03) only asks for a self-consistent core.  KL on the core is 338 nats at
  2e-4, i.e. no price; the split is decided by the decoder's convenience, and the convenient split is "everything
  clip-specific, action included, in the free residual".  The review's operational definition is right; these two
  cross-path losses are too weak to enforce it against a free residual.
- **Pending.** The downstream test (policy predicts z^P only, job 20642928 on the gpu partition) is still the
  criterion the arm was built for; a less informative core may be easier to predict (latent R2 > 0.50?) but its
  oracle L1 will be worse.  Recorded as **failed at Tier 2** unless that job shows a chain-L1 drop.
- **Candidate v2 (not run; ~4 h gpu, 7 SU left on the gpu account).** Give the cross path the RECIPIENT's residual,
  D(s_j, z^P_i, r_j), and keep the cycle on z^P_i: then any action content in r_j fights the cycle (the output must
  carry i's action, not j's), so action is pushed into the core while scene-specific content may stay in r.  This is
  the u-semantics of the original LAM with a fair KL price and 64 dims; it is a one-line change in stage4 `step`.
  Files: `acdc_la/results/{stage1,stage2,stage3}_probes.json` (rows s4_pcrr, s3_latp_1ep_cont), `logs/s4_20627479.out`.

### 2026-09-13 — closed-loop with same-data UniVLA, MLP recipe, 100 episodes per arm (job 20628004): 5-22 %, every arm inside the noise band

  **LIBERO-spatial, 10 tasks x 10 held-out starts, MLP heads (2000 steps) on frozen DINO g4 + predicted latent; 220 steps max;
  same protocol as 09-10 but 100 episodes per arm instead of 50**

  | arm | success | per task (0-9) |
  |---|---|---|
  | univla_s2mix_1ep (same data, same budget) | 0.22 | .2 0 0 .3 .3 .1 .3 .4 .1 .5 |
  | s1_acdc_1ep (ours, Stage I) | 0.19 | .3 0 0 .5 .4 0 0 .1 0 .6 |
  | nolatent | 0.18 | .2 0 0 .7 .5 0 0 0 .1 .3 |
  | s1_mean_1ep (pooled) | 0.17 | .3 0 0 .2 .1 0 0 .2 .1 .8 |
  | s1_mean_1ep+T / s1_acdc+T (legacy 10k) / s1_acdc (legacy 10k) | 0.14 / 0.14 / 0.11 | |
  | s1_acdc_1ep+T | 0.05 | .1 0 0 .1 0 0 0 0 .1 .2 |

- **Reading.** Binomial SE at p ~ 0.18 with n = 100 is 3.8 pp, and the episodes within a task are not independent,
  so anything within ~8 pp is one number: UniVLA 22, ours 19, no latent 18, pooled 17 are indistinguishable, and
  none beats the no-latent head.  The same pair of arms swings by 4-5 successes in opposite directions from task
  to task (ACDC vs ACDC+T: 0.5 vs 0.1 on task 3, pooled+T vs pooled: 0.5 vs 0.1 on task 4).  The only arm outside
  the band is ACDC+T at 5 %, which is also the arm that the history recipe (09-13, job 20623225: bc 15, ours 5,
  UniVLA 1) put lowest, so "decoder-hidden-state interface on top of a predicted ACDC latent" is the one
  configuration that reliably hurts.
- **Together with job 20623225:** two recipes, two orderings (MLP: UniVLA >= ours >= bc; history: bc > ours > UniVLA),
  both inside noise.  The paper's closed-loop statement is that LIBERO-spatial success at this policy scale does
  not separate any latent from any other or from no latent, and the offline oracle/chain table is where the
  tokenizer differences live.  No further closed-loop runs are worth SU unless a policy first moves the offline
  chain L1 by more than 0.02.
  Files: `acdc_la/results/libero_matched_closedloop.json`, `acdc_la/logs/acc_20628004.out`.

### 2026-09-13 — Stage IV v1 downstream (job 20642928): the portable core is harder to predict, not easier; chain unchanged; closed-loop 7 %

  **Policy recipe of job 20623225 with `--tokenizer s4_pcrr --transport s4_pcrr` (the policy predicts z^P only, T(s, z^P, 0) feeds the head)**

  | arm | chain L1 | oracle L1 | latent R2 | closed-loop (100 eps) |
  |---|---|---|---|---|
  | ours, full code z (09-13, 20623225) | 0.427 | 0.373 | 0.46 | 0.05 |
  | ours, portable core z^P | 0.427 | 0.374 | **0.38** | 0.07 (.2 0 0 .2 0 0 0 0 0 .3) |
  | ours_noeff, portable core | 0.427 | 0.373 | 0.39 | - |

- **Reading.** The split's downstream premise was that removing clip-specific content would make the policy's target
  easier to predict; the measured latent R2 went the other way (0.46 -> 0.38) while chain L1 did not move at all.
  Consistent with the Tier-2 probes: the core lost action content, not unpredictable junk.  Stage IV v1 is
  **closed: failed at Tier 2 and Tier 4**.  The v2 candidate (recipient residual in the cross path, entry above)
  remains the only untested variant; it is not queued.
  Files: `acdc_la/results/libero_policy_pcrr.json`, `acdc_la/logs/policy_20642928.out`.

### 2026-09-13 — direction (user review): deployment-constrained latent actions; VQ gate submitted; full-VLA downstream in preparation

- **Decision.** Stage IV v2 is not pursued.  The diagnosis is restated as "a good posterior latent is not a good
  deployment target": the policy must predict z without the future, and reconstruction lets the posterior keep
  what the deployment-time information set cannot determine.  Method direction: a deployment prior
  p(z | o_{t-H:t}, l) trained WITH the tokenizer (KL(q || p) on a factorised discrete per-cell code), keeping the
  Net-Dynamics cells and LATP ("preserve, predict what can be deployed, transport").  The toy closed-loop heads are
  declared unable to separate latents; the paper's success-rate comparison has to be UniVLA's own VLA recipe with the
  latent labels swapped.
- **Gate (jobs 20659658 gpu / 20659659 ai, `downstream/libero_vqgate.py`).** Per-cell k-means (16 / 64 / 256 codes) of
  the existing `s1_acdc_1ep` code; oracle on the dequantised code; the same history policy as a per-cell classifier;
  chain L1 with the mode and with the expectation.  Go if a vq arm beats the continuous arm of the same run by
  >= 0.015 L1 (it was 0.427); otherwise the joint prior pretraining is not started.
- **Full-VLA preparation (plan `acdc_la/VLA_DOWNSTREAM_PLAN.md`).** Both arms from `openvla/openvla-7b`, UniVLA's LIBERO
  fine-tuning script and action decoder unchanged, labels = their same-data LAM (`s2mix_1ep`) vs our per-cell VQ
  tokens (8 tokens).  Downloads (OpenVLA-7B, LIBERO-spatial RLDS) and the env build started 09-13 evening.
  Cost ~50-60 H100-hours; the whole-node queue wait (1-3 days) is the risk against the 09-17 deadline.

### 2026-09-13 — post-hoc quantisation of the ACDC code is lossy in variance: per-cell k-means keeps 16-38 %, product quantisation up to 58 %

  Codebooks fitted on the OXE validation dump of `s1_acdc_1ep` (11,456 clips, 8 cells x 32 dims); relative quantisation
  error = ||z - z_q||^2 / ||z - mean||^2.  Files `acdc_la/ckpt/vq_s1_acdc_1ep_{K16,K64,K256,pq4x16,pq2x64,pq4x64}.npz`.

  | codebook | tokens per clip | words per token | rel. quant. error |
  |---|---|---|---|
  | per cell K16 / K64 / K256 | 8 | 16 / 64 / 256 | 0.837 / 0.733 / 0.622 |
  | pq2x64 (2 sub-vectors per cell) | 16 | 64 | 0.619 |
  | pq4x16 / pq4x64 (4 sub-vectors) | 32 | 16 / 64 | 0.591 / 0.424 |
  | pq8x16 (8 sub-vectors of 4 dims) | 64 | 16 | 0.354 |

- **Reading.** The continuous code is high-entropy per dimension (KL 51 nats over 256 dims, no per-dim collapse), so
  a small post-hoc codebook cannot represent it; whether the LOST variance carries action information is exactly
  what the gate's `oracle_dequant` column measures (the continuous oracle is 0.344 MLP / 0.373 recipe).  If the
  oracle survives at pq4x64 but not at K64, the deployment-prior method needs a jointly learned discrete
  bottleneck (FSQ / VQ inside the tokenizer), not a post-hoc one; the gate's pq arms were added for this reason.
  The VLA labeller (`downstream/vla/acdc_tokens.py`) accepts any of these codebooks (n_sub field).

### 2026-09-13 (night) — full-VLA controlled comparison: pipeline verified on CPU, production jobs queued

- Env `envs/univla_vla` (UniVLA pins + protobuf 3.20.3 / tensorflow_metadata 1.14 / wandb 0.16.6 / braceexpand /
  webdataset / bddl / easydict), data (`data/libero_rlds/libero_spatial_no_noops`, 432 episodes; `cache/vla/openvla-7b`),
  code `acdc_la/downstream/vla/` (commits ca5227d .. latest).  CPU check: one RLDS 14-frame window -> ACDC labeller
  (frames 0,3,6,9,12 -> DINO g4 -> effect cells -> E -> per-cell codebook) -> 8 `<ACT_i>` tokens (ids 32001+) ->
  prompt + labels masked to the 8 tokens + EOS, 12 x 7 action chunk.  s2mix_1ep loads into their LAM class (frozen DINO
  from the hub cache, rala2's lang_proj ignored).  Their evaluator imports with n_tokens / Kc from ctrl_meta.json.
- Jobs: smoke 20659988 (ai) / 20659989 (gpu); production `LATENT=univla` 20661177 and `LATENT=acdc` (K64 codebook,
  to be replaced by the gate winner if the gate says otherwise) 20661178, 4 x H100, 15k steps x 64; eval scripts ready
  (`scripts/run_vla_eval.sh`, 50 trials x 10 tasks).  Whole-node queue estimate: none given.
- Gate: CPU run (per-cell vq arms) still extracting LIBERO DINO features on the login node; GPU twins 20660076 /
  20660077 carry the pq arms as well.

### 2026-09-14 — VLA smoke passed on Anvil and on the user's second server; one bug fixed; code on GitHub

- Bug (found on the second server, confirmed here): `finetune_libero_ctrl.py` assigned `lam.device`, a read-only property of
  UniVLA's LAM class -> the univla arm crashed at start.  Fixed (commit cf5eb3d on `main`, plus `CTRL_STAGGER_SEC` to stagger
  7B loads on small-RAM hosts).  Anvil smoke (job 20659989, gpu partition): acdc arm 12 steps + merge in 645 s, univla arm
  in 685 s, both exit 0; token loss 28 -> 8.7 (acdc, 64-word vocab) and 26 -> 3.7 (univla, 16 words, token acc 0.44 after
  12 steps -- its peaked codebook again).  Evaluator smoke (1 trial/task) submitted: 20670798 gpu / 20670799 ai.
- Second server (user's, 2 x Blackwell, 62 GB RAM): repo cloned, envs rebuilt on torch 2.7.1+cu128, mujoco pinned to 2.3.7
  (the univla_vla env's mujoco 3.13 was incompatible with robosuite 1.4.1 -- latent on Anvil since no eval ran there),
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for LIBERO init states; both training arms smoke-pass; closed-loop smoke running.
  Production there: 2 GPUs, batch 8 x accum 4, ~8-10 h per arm; univla arm first, acdc arm with K64 unless the gate
  says otherwise.  Anvil's queued production jobs (20661177/8) stay as backup.
- Code: https://github.com/DingTianxingjian/Latent_Free (`main`, fresh history; local `master` has corrupt old blobs).
  `REPRODUCE.md` is the guide; `transfer/acdc_la_artifacts.tar` (1.23 GB) carries the checkpoints.

### 2026-09-14 — VQ gate (job 20660076): NO-GO for post-hoc discretisation; every discrete code loses 0.03-0.04 at the oracle and the chain stays at the floor

  **LIBERO-spatial, history-policy recipe (8,000 iters), codebooks fitted on LIBERO train steps; `ours` = continuous code, same run**

  | arm | tokens | words | quant. err | L1 oracle (dequantised) | L1 chain mode | L1 chain expect | code acc (majority) |
  |---|---|---|---|---|---|---|---|
  | ours (continuous) | - | - | - | **0.367** | 0.430 | - | latent R2 0.44 |
  | vq64 | 8 | 64 | 0.524 | 0.408 | 0.434 | 0.430 | 0.403 (0.027) |
  | vq256 | 8 | 256 | 0.429 | 0.402 | 0.433 | 0.429 | 0.299 (0.007) |
  | pq4x16 | 32 | 16 | 0.479 | 0.401 | 0.432 | 0.428 | 0.391 (0.080) |
  | pq2x64 | 16 | 64 | 0.459 | 0.401 | 0.430 | 0.425 | 0.335 (0.025) |
  | pq4x64 | 32 | 64 | 0.337 | 0.394 | 0.433 | 0.428 | 0.249 (0.023) |

- **Verdict.** Gate rule was chain L1 <= continuous - 0.015 (i.e. <= 0.415): best discrete arm 0.425 (pq2x64, expectation
  decoding), i.e. -0.005.  **Not passed.**  (i) Quantising the posterior costs 0.03-0.04 at the oracle even at 33 %
  residual variance (pq4x64): the action-relevant part of the code is spread over dims a small codebook does not keep.
  (ii) The policy predicts the codes far above chance (25-40 % vs 1-8 % majority) and it does not help: what is
  predictable in the code is not what carries the action.  (iii) Hence a deployment prior on a POST-HOC code cannot
  work; if the direction is pursued, the discrete bottleneck must be learned jointly with the tokenizer and the prior
  (VQ/FSQ inside the model), which is a new pretraining run, not a downstream fix.  No joint pretraining is started
  before the deadline; the paper reports this gate as the third negative on the posterior-to-policy gap.
- **Consequence for the VLA comparison.** The acdc arm must feed discrete tokens, and every post-hoc codebook is a
  handicap relative to UniVLA's end-to-end VQ (their tokens are what their LAM was trained to emit).  Chosen codebook:
  pq2x64 (16 tokens, best chain, oracle 0.401, code acc 0.335); Anvil job 20661178 (K64) cancelled and resubmitted with
  pq2x64; the second server should use `--acdc_codebook vq_s1_acdc_1ep_pq2x64.npz` as well.  This caveat goes in the
  paper next to the table.  Files: `acdc_la/results/libero_vqgate.json`, `logs/vqgate_20660076.out`.

### 2026-09-14 — same-data UniVLA at 2 epochs finished (job 20617444): `claude_survey/ckpt/s2mix_2ep.pt`, 256k x 64, 26.2 h; evaluations submitted

- Their Stage-2 recipe on our capped mixture, 2 x the 1-epoch budget, final loss 0.118 (mse 0.111), 15/16 controllable
  codes in use, 1/16 uncontrollable (the collapsed branch again).  Submitted: our probes on its codes (`univla_matched_2ep`
  dump + Stage I/II probes vs s1_acdc_2ep / s1_mean_2ep / s1_acdc_1ep / univla_matched; jobs 20685768 gpu / 20685769 ai)
  and their eval_mix transport (20685771 gpu / 20685772 ai).  This completes the 1-ep -> 2-ep pair for all three
  tokenizers on the same data.

### 2026-09-14 — controlled VLA, univla arm trained on Anvil (job 20661177): 15k steps x 64 in 5.4 h; acdc arm (pq2x64) training; evaluation submitted

- `ckpt/vla_runs/univla+libero_spatial_no_noops+b16x4+lr0.00035+r32+s15000/`: merged OpenVLA-7B + LoRA, action decoder,
  dataset statistics.  Training curve (their recipe, our same-data LAM labels): token accuracy 0.72 (step 300) -> 0.82
  (15k), action L1 (normalised, 12-step chunk) 0.173 -> 0.063; lr x0.1 at 12k.  Evaluation (50 trials x 10 tasks, their
  evaluator with SDPA attention) submitted: 20711528 gpu / 20711529 ai.
- acdc arm (`--acdc_codebook vq_s1_acdc_1ep_pq2x64.npz`, 16 tokens of 64 words), job 20675236 on h005: at step 5k token
  accuracy 0.36 (vs 0.80 for the 4 x 16-word univla tokens), action L1 0.121 vs 0.111 at the same step -- the post-hoc
  token handicap predicted by the gate shows up as a ~0.01 training-loss gap.  ETA ~4 h + merge.

### 2026-09-14 — same-data UniVLA at 2 epochs on our probes (job 20685769): its second epoch helps (+0.025 net); it reads the temporal PROFILE better than ACDC, and net worse

  **Same 11,456 OXE val clips, corrected R2, per-source episode-disjoint ridge; Stage II N_TC = patch readable from the patch-rendering code**

  | tokenizer | budget | R2 net | seg1 | prof | seq | patch drop | N_TC |
  |---|---|---|---|---|---|---|---|
  | s1_acdc_2ep (ours) | 2 ep | **0.321** | **0.259** | 0.084 | 0.228 | 28.7 % | 0.102 |
  | s1_mean_2ep (pooled) | 2 ep | 0.336 | 0.207 | 0.014 | 0.207 | 36.1 % | 0.106 |
  | univla_matched_2ep (their recipe, our data) | 2 ep | 0.261 | 0.249 | **0.139** | **0.237** | **18.2 %** | -0.023 |
  | univla_matched (1 ep) | 1 ep | 0.236 | 0.222 | 0.125 | 0.215 | 22.9 % | -0.027 |
  | univla_official (released, their full data) | - | 0.288 | 0.281 | 0.152 | 0.259 | 21.2 % | -0.018 |

- **Reading.** (i) Unlike both of ours (flat from 1 to 2 epochs), the same-data UniVLA still improves with budget
  (+0.025 net, +0.027 seg1, +0.014 prof, +0.022 seq) and closes part of the gap to the released model.  (ii) The
  honest ordering on the same data: ACDC > UniVLA on net displacement (0.321 vs 0.261) and marginally on seg1
  (0.259 vs 0.249); UniVLA > ACDC on the temporal profile (0.139 vs 0.084), on seq (0.237 vs 0.228), and on patch
  robustness (18 % vs 29 % drop; N_TC ~0 from its quantiser).  The paper's Stage-I profile claim therefore holds
  **against pooling** (0.084 vs 0.014), not against UniVLA; the claim against UniVLA is net displacement and the
  transport result.  (iii) UniVLA's 2048-d code is 16 quantised 128-d embeddings x 4 patches x (K = 4 pairs); its
  profile readability comes from per-pair codes, which is the pair-wise structure the 09-12 review already noted.
  Files: `acdc_la/results/stage1_probes.json`, `stage2_probes.json` (rows univla_matched_2ep), `results/univla_matched_2ep_val.npz`.
  Their transport eval (job 20685772, their pipeline, 11,353 clips): indep self 0.235, transfer **0.121**, replay 0.012
  (1 ep: 0.218 / 0.116 / 0.014; ours s3_latp_1ep 0.319 transfer).  The second epoch does not move their transport either.

### 2026-09-14 — controlled VLA, both arms trained on Anvil; evaluations queued

  | arm | tokens | job | wall | final train token acc | final train action L1 | checkpoint |
  |---|---|---|---|---|---|---|
  | univla (same-data LAM s2mix_1ep) | 4 x 16 words | 20661177 | 5.4 h | 0.82 | 0.063 | `ckpt/vla_runs/univla+libero_spatial_no_noops+b16x4+lr0.00035+r32+s15000` |
  | acdc (s1_acdc_1ep, pq2x64 post-hoc codebook) | 16 x 64 words | 20675236 | 4.8 h | 0.41 | 0.079 | `ckpt/vla_runs/acdc+libero_spatial_no_noops+b16x4+lr0.00035+r32+s15000` |

- Both from openvla-7b, LoRA r32 + trained token embeddings, 15k steps x 64, lr 3.5e-4 (x0.1 at 12k), image aug, their
  action decoder (MAP over the last n latent-token states) and 12-step L1.  The training-time action loss ends 0.016
  higher for the acdc arm; the gate predicted a handicap for any post-hoc code (oracle 0.401 vs 0.367 continuous).
- Evaluations (their `run_libero_eval.py` with SDPA attention, 50 trials x 10 tasks): univla 20711541 gpu / 20711542 ai,
  acdc 20719072 gpu / 20719073 ai; evaluator smoke 20711539 / 20711540 still queued (two evaluator bugs fixed today:
  flash-attn forced by their loader, then a dropped `.register` in my wrapper -- commits ae1ac6b, e5adc21).

### 2026-09-14 — the same-data UniVLA baseline is NOT warm-started: 5,120 of 201.8M params inherited at init, none of them the deployed codebook, and all of them moved by training

  Concern (user): `s2mix_1ep` initialises from the RELEASED stage-1 checkpoint (trained on their full data), so the
  baseline may be pre-trained in disguise.  Measured on the checkpoints (`downstream/vla/check_univla_tokens.py`):

  | stage-2 module | params | inherited from the released stage 1 |
  |---|---|---|
  | encoder | 113,959,680 | no |
  | decoder | 86,211,072 | no |
  | to_codebook / to_codebook_uncontrol | 98,432 each | no |
  | `action_latent` (query embedding) | 3,072 of 6,144 | yes, at init only |
  | `vq` (task-IRRELEVANT codebook, not deployed) | 2,048 | yes, at init only |
  | **`vq_action` (the DEPLOYED task-centric codebook)** | 2,048 | **does not exist in stage 1** (rala2.py:617 asserts it) |

- **Findings.** (i) `init_from_stage1` copies only keys matching `vq` / `action_latent` (their `genie/model.py:66-71`):
  5,120 params = 0.0025 % of the 201.8M stage-2 model.  (ii) Those are the uncontrollable branch's codebook and the
  query embedding, NOT the codebook whose indices become the `<ACT_i>` tokens downstream.  (iii) After our training,
  even `vq.codebook.weight` differs from the released one by max 2.42 (the VQ library restarts dead codes: the training
  log prints "Restarting 10 codes"), so nothing in the final checkpoint is inherited (0 of 201,757,184 tensors
  byte-identical).  (iv) Everything that produces the deployed code -- the 114M encoder, the projections, `vq_action` --
  is trained from scratch on our capped mixture at our budget.
- **Conclusion: the controlled comparison is fair on this axis**, and the earlier caveat in this ledger ("the codebook
  saw their full data, an advantage we gave the baseline") is **RETRACTED** as substantively wrong: the released
  stage-1 gives the baseline nothing that survives to the deployed tokens.  Both sides do use the same frozen Meta
  DINOv2 ViT-B/14 (ours plain, theirs with registers), which is symmetric.
- **The one asymmetry that remains favours US and must be stated:** budget is matched by optimiser steps x batch
  (128k x 64) but not by frames -- a UniVLA step sees 64 frame PAIRS (128 frames), a step of ours sees 64 five-frame
  clips (320 frames).  **Verified (job 20720187):** the labeller class (UniVLA's own `ControllableDINOLatentActionModel` loaded with our
  weights, `dino_encoder` rebuilt from the hub cache, rala2's `lang_proj` unused) reproduces rala2's training-time
  deployed indices **bit for bit** on the same input, so the univla VLA arm is trained on UniVLA's own task-centric
  tokens.  Inherited at init: 5,120 params (`action_latent` 3,072 + `vq.codebook.weight` 2,048); after training both
  differ from the released values (max |diff| 0.20 and 2.42).

### 2026-09-14 — what the VLA token accuracies mean (job 20725946): different floors, 9x different label entropy, and our post-hoc tokens are 87 % augmentation-flippable

  **1,000 LIBERO-spatial windows through each arm's labeller, the same RLDS pipeline and augmentation the VLA arms train on
  (`downstream/vla/label_stats.py`).  "stability" = agreement between the label of a window and of the SAME window under one
  further draw of the training augmentation (crop 0.9, brightness 0.2, contrast/saturation 0.8-1.2, hue 0.05).**

  | arm | tokens | words | chance | majority floor | label entropy | codes used | stability |
  |---|---|---|---|---|---|---|---|
  | univla (`s2mix_1ep`) | 4 | 16 | 0.062 | **0.469** (per slot .42/.17/.51/.78) | 9.45 of 16 bits | 11/16/10/9 | **0.610** |
  | acdc (`s1_acdc_1ep` + pq2x64) | 16 | 64 | 0.016 | **0.088** | 85.4 of 96 bits | 57-62 of 64 | **0.126** |

- **The headline 0.82 vs 0.41 is not a comparison.**  Removing the majority floor, (acc - maj)/(1 - maj) = **0.66 (univla)
  vs 0.35 (ours)**; and their label carries 9.5 bits per sample where ours carries 85.4.  In recovered information the
  ordering reverses: their VLA leaves ~2 bits of residual (0.35 nats x 4 tokens), ours ~45 bits (1.93 x 16), i.e. ours
  recovers ~41 bits of label against their ~7.  Their tokens are easy because the codebook is peaked on this domain
  (one slot is 78 % a single code; 9-11 of 16 codes ever used).
- **New and mechanistic: our post-hoc tokens are mostly nuisance.**  Re-drawing the training augmentation on the same clip
  changes 87 % of our tokens but only 39 % of theirs.  The ACDC code is continuous and fine-grained, so k-means cell
  boundaries are dense (relative quantisation error 0.46 at pq2x64) and a photometric/crop jitter flips the nearest
  centroid; a 16-bit native codebook cannot represent that much detail and is forced to keep only reconstruction-relevant
  structure.  This is an independent reason -- separate from the gate's oracle loss -- that post-hoc quantisation of a
  continuous tokenizer is the wrong VLA interface, and it is direct support for training the discrete bottleneck jointly
  (`common/tcvq.py`, built and smoke-tested, not yet trained).
- **Caveat.** The jitter is an ADDITIONAL draw on top of the pipeline's own augmentation, so the perturbation magnitude is
  comparable to, not identical with, the training one.  Files: `acdc_la/results/vla_label_stats.json`, `logs/labstat_20725946.out`.

### 2026-09-14 — moved to the workstation; full-VLA pipeline verified end to end on GPU; the Anvil univla-arm smoke had crashed

- **Host.** `~/Projects/latent_free`, 2 x RTX PRO 6000 Blackwell (96 GB), 62 GB RAM shared with other jobs, no Slurm.
  This ledger moved into the repository as `LEDGER.md` (Anvil copy of 2026-09-13 21:20 is the starting point); GitHub
  `main` is the sync point.
- **Envs (deviations from `REPRODUCE.md`).** Blackwell (sm_120) needs cu128: all three venvs use torch 2.7.1+cu128 /
  torchvision 0.22.1, every other pin from `acdc_la/envs/*.requirements.txt` (installed `--no-deps`; the univla freeze
  conflicts by design, protobuf 3.20.3).  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the torch >= 2.6 default breaks
  LIBERO's init-state load).  `envs/univla_vla`: mujoco 3.13.0 -> 2.3.7 (3.13 asserts inside robosuite 1.4.1 when any
  LIBERO env is built) and flash-attn 2.8.3.post1 added (UniVLA's `openvla_utils.get_vla` hard-codes
  flash_attention_2; the Anvil env has no flash_attn).  Both eval blockers also hold on Anvil, where no eval had run.
- **Training smoke (1 GPU, 12 steps, batch 2, shuffle 200).** acdc: token_loss 26.6 -> 8.2, token_acc 0.00, 53 s incl.
  merge, peak RSS 35 GB.  univla: first run crashed in `build_labeller` (`can't set attribute 'device'`), the same
  error as Anvil smoke job 20659988 (`SMOKE univla exit 1`) -> that smoke is **invalid** for the univla arm; fixed in
  cf5eb3d; rerun token_loss 26.1 -> 3.56, token_acc 0.50, 48 s, peak RSS 35 GB.  Smoke numbers validate the code path
  only.
- **Eval smoke.** `run_libero_eval_ctrl.py`, libero_spatial, 1 trial x 10 tasks per arm on the 12-step checkpoints:
  both exit 0, 0/10 (expected), ~50 s per arm.
- **Conclusion — confirmed (infrastructure).** train -> merge -> closed-loop eval runs for both arms on this host.
  Anvil production jobs 20661177 / 20661178 pick up cf5eb3d at start, but their eval would fail on mujoco 3.13 /
  missing flash_attn.
- **Artifacts.** `acdc_la/ckpt/vla_runs/*--smoke`, `acdc_la/results/vla_eval_smoke/`; host-local logs and env recipe in
  `logs_migrate/` (`build_envs.sh`, `env_local.sh`, `vla_smoke.log`, `eval_smoke_*.log`).
- **Next.** Production (15k steps x 64) on this host: 2 GPUs, batch 8 x accum 4 per GPU; RAM is the constraint
  (35 GB peak per process on a shared host) -> `CTRL_STAGGER_SEC` and a measured steady-state RSS before both ranks run.
  Pretraining caches: capped `cache/oxe` (1.8 TB) + `cache/bridge_big` (95 GB) on `/data/ti874297/latent_free_cache`
  (symlinked), transfer in progress; `cache/oxe_full` (5.3 TB) not copied.


### 2026-09-14 — 2-GPU DDP on the workstation: NCCL 2.26.2 crashes, 2.31.2 works; production probe 3.1 s/step, 16 GB RSS

- **Failure.** Any NCCL collective between the two Blackwell cards (PCIe, different host bridges, IOMMU on) died with
  `CUDA error: an illegal memory access` in the NCCL watchdog, also with `NCCL_P2P_DISABLE=1` or `NCCL_CUMEM_ENABLE=0`;
  only P2P + SHM both disabled (socket transport) passed — too slow for ~1.5 GB of gradients per micro-batch.
  Fix: `nvidia-nccl-cu12` 2.26.2 -> 2.31.2 in `envs/univla_vla` (torch loads it at runtime); default transport and
  P2P-disabled both pass a 2-rank all_reduce up to 256 MB (`logs_migrate/nccl_test.py`).  **Confirmed (infrastructure).**
- **Probe (plan B = arms in sequence, 2 GPUs each; univla arm, batch 8 x accum 4 per GPU = 64, shuffle 16000,
  `CTRL_STAGGER_SEC=90`, 105 steps, killed before any save; GPUs shared with another job).** Stable NCCL; token_acc
  0.40 -> 0.71, action_l1 0.45 -> 0.195 over steps 10-105 (train batch, code-path check only); 3.1 s/step after the
  shuffle buffer fills -> ~13 h per arm at 15k steps; process-tree RSS 15-16 GB steady, 19 GB peak at the staggered
  loads, host MemAvailable min 13 GB.  Merge at the end (rank 0 reloads the 7B base) peaked at 35 GB RSS in the smoke.
- **Next.** Production via `logs_migrate/vla_production.sh` (tmux): univla then acdc (K64 codebook, the 09-13 default;
  the VQ gate has not run), each 15k steps x 64 on 2 GPUs, then 50 trials x 10 tasks LIBERO-spatial eval per arm.

### 2026-09-14 — 10-epoch chains stopped at 7.4 / 6.0 epochs (user decision): the profile gap is flat from 30 % of budget, and the queue was needed for the decisive downstream run

  **In-training proxy curve (200 clips/source, same reader at every point); `rec` = validation reconstruction, `prof` = R2 of the
  temporal profile from the code.  Checkpoints kept and resumable: `ckpt/s1_acdc_10ep.pt` (step 955,000), `ckpt/s1_mean_10ep.pt` (775,000).**

  | budget | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % |
  |---|---|---|---|---|---|---|---|
  | ACDC rec | 1.460 | 1.452 | 1.427 | 1.421 | 1.417 | 1.414 | 1.412 |
  | ACDC prof | -0.019 | -0.031 | 0.030 | 0.029 | 0.032 | 0.032 | 0.031 |
  | pooled rec | 1.589 | 1.581 | 1.557 | 1.552 | 1.550 | 1.549 | - |
  | pooled prof | -0.040 | -0.044 | -0.003 | -0.008 | -0.003 | -0.003 | - |

- **Reading.** From 30 % of the 10-epoch budget (3 epochs) onward the profile gap is +0.034 and does not move out to 7.4 epochs,
  while the pooled tokenizer never reads the profile at all at any budget.  Reconstruction still creeps down (1.427 -> 1.412)
  without changing what the code carries.  This is the long-budget verification the paper needs on this axis; the full 10
  epochs would add nothing.  (Curve probes use 200 clips/source and are not comparable to the 400-clip dump numbers.)
- **Decision (user, 2026-09-14).** Cancel both running legs and all five dependent legs; keep the checkpoints; give the
  fairshare to the continuous-latent decisive run and to the two closed-loop evaluations, which are the deadline path.

### 2026-09-15 — scheduling reset (user): everything is now ranked by "does it produce a main model that beats matched UniVLA"

- **Cancelled as strictly superseded:** the old-tokenizer continuous arm (`acdc_cont`, 2- and 4-GPU race), the old
  ACDC-PQ 500-rollout evaluation, the no-latent smoke, all further 10-epoch pretraining, Portable Core v2, further
  nuisance arms, AC2/AC3 sweeps and post-hoc codebook sweeps.  None of them can change the final model: whatever the
  old quadrant+MLP tokenizer shows, it will not be the main model now that a stronger structured one is training.
- **Kept (P0):** `tc_cont` and `tc_vq256` tokenizers (128k x 64 each), and the matched-UniVLA full closed-loop
  evaluation (job 20730279) as the external baseline the paper must beat.
- **Gate before any VLA run (P1, `scripts/run_tc_gate.sh`):** OXE probes from the training dump + LIBERO oracle action
  L1 for both tokenizers.  `tc_cont` proceeds if oracle L1 < 0.344 (the old ACDC) and the temporal probes do not
  regress; `tc_vq256` proceeds only if its NATIVE quantised oracle is close to the continuous one (post-hoc pq2x64
  was 0.401).  Only the winner gets the 8-11 h VLA run; both only if both are strong.
- **Deferred until the main model wins (P2):** the endpoint vs endpoint+dynamics ablation on the winning architecture
  (the causal test for A_1, currently confounded with the spatial and encoder upgrades), LATP retrained on the winning
  tokenizer (the paper cannot present NetDyn and LATP from different checkpoints as one method), and functional
  equivariance only if LATP again lifts transfer without lifting control.
- **The five comparisons the final paper needs, and no more:** (A) UniVLA vs full ours, closed-loop; (B) endpoint vs
  endpoint+dynamics; (C) continuous vs native VQ interface; (D) reconstruction decoder vs LATP on the winning
  tokenizer; (E) LATP vs LATP+equivariance, only if D lifts transfer but not control.

### 2026-09-15 — direct measurement: how much of each latent the deployment state can determine (P)

- **Test.** Every diagnostic so far inferred the posterior-to-policy gap *through a trained policy* (oracle L1 0.344
  vs chain L1 0.42).  This measures the underlying quantity directly and without any VLA:
  `P = R2(z | F_0)`, ridge from the first-frame quadrant features to the code on a per-source episode-disjoint split,
  clip-weighted over sources; reported beside the code's action content `S = R2(net displacement | z)` and the
  reference `R2(net displacement | F_0)` under the identical reader.  `usable = max(0,P) * max(0,S)` is an ordering
  heuristic, not an identity.
- **Setup.** `acdc_la/common/predictability.py` on the existing OXE validation dumps (11,456 clips, 29 sources).
  `F_0` is the deployment information set minus language, so P is a *lower bound* on what an observation-conditioned
  policy could recover; the same bound applies to every arm, so the comparison is fair even though the level is not.

  | metric | ACDC 1ep | pooled 1ep | UniVLA matched 1ep | UniVLA matched 2ep | ACDC 2ep | pooled 2ep | PCRR core |
  |---|---|---|---|---|---|---|---|
  | P = R2(z \| F_0) | 0.045 | **0.087** | -0.164 | 0.016 | 0.045 | 0.088 | 0.034 |
  | S = R2(action \| z) | 0.324 | 0.336 | 0.236 | 0.261 | 0.321 | 0.336 | 0.261 |
  | usable = P x S | 0.020 | **0.041** | 0.010 | 0.012 | 0.020 | 0.039 | 0.015 |
  | R2(action \| F_0) | 0.142 | 0.142 | 0.139 | 0.139 | 0.139 | 0.139 | 0.137 |
  | code dims | 256 | 128 | 2048 | 2048 | 256 | 128 | 192 |

- **Result.** (1) P is near zero for every latent measured *by this reader* -- but see the retraction below: for
  UniVLA this reader is the wrong instrument and the conclusion does not hold.  (2) The predictable
  action content of any code (usable, 0.010-0.041) is **below what the same ridge already gets from the image alone**
  (0.139-0.142).  (3) Ours dominates matched UniVLA on all three columns (P 0.045 vs 0.016, S 0.324 vs 0.261,
  usable 0.020 vs 0.012), so the information argument for the tokenizer holds.  (4) Within our own family the
  *most compressed* variant is the most usable: mean-pooled 128-d has 2x our P and 2x our usable while matching S.
  (5) A second epoch changes nothing (0.045/0.020 both epochs).
- **Control.** One reader implementation for all three regressions and all arms, on a per-source episode-disjoint split
  (CORRECTED 2026-09-16: this line originally said leave-one-source-out, which the code never did).  UniVLA's negative
  P comes from a 2048-d target fitted on ~270 clips per source, not from source shift; not a claim that their code is
  anti-predictable.  `F0Q` is quadrant-pooled DINOv2, not the
  7B backbone's state, and carries no instruction: absolute levels are lower bounds, only the ordering is claimed.
- **Conclusion — PARTLY RETRACTED 2026-09-15, see the entry below.**  This relocates
  the binding constraint.  The lever is **not** S — we already beat UniVLA on action content — it is **P**, and
  nothing in the pipeline optimises P.  It also predicts the closed-loop result we could not explain: if the
  predictable action content of the latent is smaller than what the image already gives the policy, a latent-conditioned
  VLA cannot separate from the no-latent control, which is exactly what we measured (0.19 vs 0.18, inside noise).
- **Risk this creates for the queued work.** `tc_cont` raises encoder capacity (patch-level transformer, no quadrant
  pooling, 256-d continuous).  Row (4) says capacity moves P the *wrong* way.  A tokenizer that wins the oracle gate
  and loses P would repeat the closed-loop null result, so the gate in `run_tc_gate.sh` is extended: report P, S and
  usable for each candidate alongside the oracle L1, and prefer the arm that raises usable even at equal oracle L1.
- **Artifacts.** [`probe`](latent_free/acdc_la/common/predictability.py),
  [`results`](latent_free/acdc_la/results/predictability.json), log `acdc_la/logs/predictability.out`.
- **Decision.** Add a third P0 tokenizer, `tc_cont_prior`: identical to `tc_cont` plus the deployment-prior term
  `lam_prior * || z - p(z | F_0, language) ||^2` (weight 0.1), which is the only term in the codebase that prices P
  directly.  Keep `tc_cont` unchanged as its control, so the prior's effect is a clean one-factor comparison rather
  than a confound with the architecture change.  Do not touch the two already-queued jobs.

### 2026-09-15 — RETRACTION: the predictability probe does not show that UniVLA's latent is unrecoverable at deployment

- **What was claimed and is now withdrawn.**  From `P = R2(z | F_0) = -0.164` for matched UniVLA I concluded that
  "all latent action models, ours and theirs, produce codes that are near-unrecoverable at deployment".  That
  conclusion is wrong, and a measurement already in this ledger contradicts it.  The within-family comparison among
  our own identically-trained arms (ACDC 0.045, pooled 0.087, PCRR 0.034) is unaffected and stands.
- **The contradicting measurement (already on disk, not a new run).**  Each trained VLA's latent-token cross-entropy
  is exactly "how well the deployment information set determines the label", in the units the policy optimises:

  | arm | tokens | nats/token (step 15,000) | residual | label entropy (LIBERO) | recovered | aug-stable |
  |---|---|---|---|---|---|---|
  | univla | 4 | 0.349 | 2.0 bits | 9.45 bits | **7.4 bits = 79 %** | 0.610 |
  | acdc (pq2x64) | 16 | 1.933 | 44.6 bits | 85.4 bits | **40.8 bits = 48 %** | 0.126 |

  Their label is not unrecoverable at deployment; it is *mostly* recoverable.  (Training-set loss at the end of
  fine-tuning, not held out -- both numbers are optimistic, and by an unknown and possibly different amount.)
- **Why the probe said otherwise -- four ways it is the wrong instrument for the cross-method column.**
  (1) *Wrong target space.*  UniVLA deploys 4 indices into a 16-entry codebook, 9.45 bits of actual entropy on this
  domain.  I ridge-regressed the 2048-d concatenated codebook embedding; R2 in that space is not the quantity a
  classifier optimises, and a near-miss in embedding space and a wrong index are not the same event.
  (2) ~~*Wrong domain: leave-one-source-out, i.e. cross-embodiment extrapolation.*~~  **WITHDRAWN 2026-09-16: the split
  is per-source and episode-disjoint, never leave-one-source-out, so this reason was false.**  What remains true is only
  that the probe is on OXE while their deployment is LIBERO after LoRA fine-tuning.  The retraction's conclusion still
  stands on reasons (1), (3) and (4).
  (3) *Wrong reader.*  Ridge on 16 quadrant-pooled DINOv2 vectors **with no language**.  Language is decisive for
  their code specifically: its majority floor is 0.469 and one slot is a single code 78 % of the time, so much of
  their label is "which task/phase is this", which the instruction states outright.  Their actual reader is a 7B VLM.
  (4) *The reference row inherits the same weakness.*  `R2(action | F_0) = 0.14` is that same weak reader, so
  "usable < 0.14" is a statement about the probe, not about what a VLA can extract.
- **What the two measurements say together, which is the real result.**  The two methods sit at opposite corners,
  and neither is dominant: UniVLA's code carries the least action information of anything measured (S = 0.236, vs
  our 0.324) but ~79 % of it is recoverable at deployment; ours carries the most and ~48 % is recoverable.  The
  ordering in raw recovered bits favours us (41 vs 7), but 87 % of our tokens flip under one further draw of the
  training augmentation against 39 % of theirs, so most of our recovered bits are nuisance we injected ourselves.
  The quantity that decides the downstream result is therefore neither P nor S nor recovered bits alone but
  **recovered AND augmentation-stable task-relevant bits**, which nothing in this repo has yet measured.
- **A separate reason not to reason from "UniVLA succeeds".**  Their published LIBERO numbers come from a pipeline
  whose VLA-pretraining stage we deliberately skipped (both arms start from `openvla-7b`).  In our own controlled
  setup no arm beats the no-latent control (univla 0.22, ours 0.19, no-latent 0.18, pooled 0.17, all inside noise),
  so this ledger has no internal evidence that *any* latent helps, theirs included.
- **Decision.**  (a) Keep `common/predictability.py` for within-family comparisons only and say so in its docstring.
  (b) Build the right instrument: held-out latent-token cross-entropy on LIBERO for each trained VLA, in bits, next
  to the label entropy and restricted to augmentation-stable windows.  Both VLA checkpoints exist, so this is a
  forward pass, not a training run.  (c) Nothing about the queued jobs changes: `tc_cont_prior` is still worth its
  slot, because the within-family result it targets (capacity buys action content and loses predictability) is the
  part that survived, and because the stability column is an independent argument for a native discrete bottleneck.

### 2026-09-15 — RETRACTION: "the VLA load test passed" — it never passed, and the first closed-loop evaluation scored 181 crashes as failures

- **What was claimed and is withdrawn.**  On 2026-09-15 I recorded that the load test passed ("10 episodes, 0/10
  success at 1 trial/task") and released the full closed-loop evaluations on that basis.  Both load-test raw logs
  (`loadtest_raw_20729085_univla.txt`, `..._acdc+l.txt`) contain **10 of 10** episodes ending in
  `Caught exception: 'OpenVLAForActionPrediction' object has no attribute 'predict_latent_action'`.
  The matched-UniVLA evaluation (job 20730279) then ran 181 episodes, every one of them the same exception scored as
  a failed rollout, before it was cancelled.  **There is no valid full-VLA closed-loop number in this ledger.**
  (The toy closed-loop figures -- univla 0.22 / ours 0.19 / no-latent 0.18 / pooled 0.17 -- come from the MLP-recipe
  harness, not this evaluator, and are unaffected.)
- **Cause.**  The merged checkpoints carry their own `configuration_prismatic.py` / `modeling_prismatic.py`, written
  by `save_pretrained`.  With `trust_remote_code=True` transformers builds the model from that dynamic module, which
  is a *different class object* from the `OpenVLAForActionPrediction` imported from the `prismatic` package at the
  top of `run_libero_eval_ctrl.py` and patched there with `predict_latent_action`.  The patch never reached the
  instance.  The checkpoint's copy descends from `openvla-7b`: it has `predict_action` and `get_action_stats`, but
  `predict_latent_action` is UniVLA's addition and is absent.  This is the third instance of the same class of bug
  in this evaluator (after `_AutoModelSDPA` subclassing and the `get_vla` function replacement).
- **Why the gate did not catch it, which is the part that matters.**  A total crash and a genuine zero produce the
  *same* headline: "0/10 success".  The gate reported the headline and I read the headline.  The exception text was
  in the summary I was shown and I did not act on it.
- **Fix.**  (a) `type(vla).predict_latent_action = ...` after loading -- correct whichever class object won -- plus
  an assertion that it took.  (b) `run_vla_loadtest.sh` now counts `Caught exception` in the raw output and exits
  non-zero if it is not 0, so a crashing evaluator can no longer be reported as a low success rate.
- **Cost.**  ~1 GPU-hour of the load test plus ~1.5 GPU-hours of job 20730279, and a day of believing the pipeline
  was verified end to end.
- **Decision.**  Re-ran the load test (job 20739385) with the new failure rule and hung BOTH full evaluations behind
  it with `afterok` (20739386 univla, 20739387 acdc), so neither can start unless zero episodes raise.

### 2026-09-15 — LATP v2 built; and a same-day self-correction about what its recovery loss measures

- **Built.** `stage3_transport/train_v2.py`: EMA target encoder, a same-source cross-episode buffer of target
  latents, `L_recover` on a self-chosen intervention, MMD realism, and Relational Transport Consistency (RTC), which
  asks that the DIFFERENCE between two interventions realise in a context-stable direction in the fixed [net, AC1]
  descriptor.  Built on TCVQ, and the checkpoint stays a plain TCVQ so every downstream tool reads it unchanged.
  Smoke asserts the EMA lag (4.7e-4), that `L_recover` reaches the online encoder (grad 4.06e3), and that RTC's
  gradient into the encoder is exactly 0 (its donors are stop-grad, so a positive RTC result localises to T).
- **First reading, which was wrong, and the correction.**  Running the full objective on PURE RANDOM tensors --
  where by construction no action semantics exist -- `rec_cos` went 0.55 -> 0.93 in six steps.  I read this as
  "E and T built a channel together" and changed the default so that T receives no recovery gradient.  The
  follow-up smoke ran both routings: **detached 0.977 vs attached 0.980, and the frozen-operator probe read 0.976
  in both.**  Co-adaptation is therefore NOT the mechanism and the default change did not address anything; it has
  been reverted to the specified routing.
- **What is actually going on.**  `z -> T(f0, z)` is injective and E simply learns to invert it.  Inverting a
  decoder is generic, carries no semantics, and in a CONTINUOUS code it is free.  This is the substantive difference
  from CycleMimic's discrete version, where recovering an index means landing in the right cell of a small shared
  codebook that cannot absorb arbitrary information.
- **Scope of the claim.**  On random data nothing pins T: `L_self` has a noise target and MMD compares to noise, so
  the run degenerates to a z -> Y -> z autoencoder.  This does NOT show that a real run degenerates, where `L_self`
  and MMD pin T to a genuine world model.  It shows that **`rec_cos` near 1 is not evidence of anything**.
- **Consequences for how a real run is read.**  (a) `rec_frozen` is kept and IS meaningful in a real run, where T_0
  is the trained Phase-1 operator -- it then asks whether E can read z out of a genuine world model's output; the
  smoke cannot exercise it because there T_0 is the random initialisation and T has not drifted.  (b) `rec_noise` is
  blind in both routings and is kept only as a cheap line.  (c) The real guard is OUTSIDE this loss: run with
  `--eval-every` so `curve_eval` reports the ONLINE encoder's action / net / profile R2 and transfer cosine during
  training.  v2 works if transfer rises while the action probes hold; E is collapsing into T's inverse if they fall.
- **Also passed today.**  The deployment-prior term end to end (`smoke_prior`, job 20738375, exit 0): the loss is
  active (`prior` 4.44, `prior_r2` -3.44 at step 6 as expected from an untrained prior net), gradient reaches both
  sides, and halving the code RAISES the normalised loss 6.17 -> 16.26 where a plain MSE would have quartered it,
  so the collapse direction is priced rather than rewarded.
- **Artifacts.** `acdc_la/stage3_transport/train_v2.py`, `scripts/run_latp2_smoke.sh`, logs `latp2smoke_20739115.out`,
  `latp2smoke_20739838.out`, `tcsmoke_20738375.out`.

### 2026-09-15 — WITHDRAWN as not comparable: "our transport transfer 0.319 vs same-data UniVLA 0.121"

- **Status.** The cross-method half of this claim is withdrawn.  It was already marked provisional ("different
  reader implementations"); checking whether it could be repaired from disk shows it cannot, and the reason is
  structural rather than a matter of re-running a script.
- **What survives, unchanged.**  The INTERNAL comparison, which is measured with one reader on one dump format:
  LATP lifts independent transfer R2 0.244 -> 0.319 and transfer cosine 0.484 -> 0.524 over the decoder-only
  reconstruction control.  Every number in it comes from `stage3_transport/probes.py` with the same per-source ridge
  fitted only on real frozen dF -> net xyz.  This is the LATP result the paper should carry.
- **Why it cannot be repaired from the existing dumps.**  `univla_matched_val.npz` and `univla_matched_2ep_val.npz`
  carry only `DF`, `Z`, `NET`, `F0Q` -- there is no `DSW` / `ASW_NET`, i.e. no swap path was ever dumped for their
  model.  Verified by listing the archives.
- **Why re-dumping is not a small job either.**  Our transfer probe reads `G(DSW)` with G a ridge from the real
  segment tensor dF (K x 16 x 768) to net xyz.  Their LAM decoder does not produce that object: it predicts a single
  future frame at patch level from (frame 0, z), not K segment deltas.  Feeding its output to G is not a matter of
  reshaping; the two decoders predict different things.
- **The one honest repair, if it is wanted.**  Define transport transfer on the ENDPOINT only, which both decoders
  do produce: fit G_end from the real endpoint displacement F_T - F_0 to net xyz, and evaluate
  `G_end(decoder output under a deranged donor code - F_0)` for both.  That is one metric, one reader, both sides,
  and it costs a shim exposing their LAM through our dump interface plus one GPU dump pass.  Deliberately NOT queued
  now: it is a Phase-2 comparison and the deadline path is the main model.
- **Decision.**  Until that exists, the paper states the LATP gain against our own controls only and makes no
  cross-method transport claim.  This is the third instrument-mismatch correction today (predictability probe vs
  UniVLA, the load-test read, and this), so the standing rule is now explicit: **a cross-method number is not
  reportable unless both sides pass through the same reader on the same object.**

### 2026-09-15 — the 1-trial load test scored 10/10 on BOTH arms; what that does and does not mean

- **Numbers.**  With the evaluator fixed, `libero_spatial`, 1 trial/task, 10 tasks: univla 10/10, acdc 10/10.
- **It refutes one thing.**  Skipping UniVLA's VLA-pretraining stage and going straight from `openvla-7b` + LoRA is
  not broken: both arms drive the robot.  The earlier impression that "every arm sits in the noise" came from the
  MLP toy harness, not from this pipeline.
- **It does not compare the arms.**  Ten episodes have essentially no resolution, and my stated prediction (that the
  acdc arm would fall clearly behind, from its 12.6 % augmentation stability and 0.41 token accuracy over a 0.088
  floor) is neither confirmed nor refuted by it.
- **The risk it exposes.**  If the 500-episode runs also land near ceiling for both arms, `libero_spatial` cannot
  separate the methods and the headline comparison is dead on arrival.
- **Prepared, not scheduled.**  Downloaded the other three suites (`libero_10` 3.5 GB, `libero_object` 2.7 GB,
  `libero_goal` 1.8 GB) on the login node -- no GPU, no queue position, no change to any scheduled job.
- **Honest cost of actually using them.**  Both VLA arms were LoRA-fine-tuned on `libero_spatial` only, so running
  them on another suite is zero-shot transfer, not a like-for-like comparison: a real suite swap means re-training
  both arms (~5.5 h each) plus two evaluations (~3 h each), about 17 GPU-hours.  Cheaper alternatives to consider
  first if spatial saturates: report steps-to-success and partial progress rather than only the success rate, or
  fine-tune on fewer demonstrations so the task is not saturated.  Decide from the 500-episode numbers, not before.

### 2026-09-15 — bits recovered at deployment: the cross-method number that replaces the retracted probe

- **Instrument.** `downstream/vla/token_bits.py`: the reader IS each trained VLA, so the units are the ones it
  optimises.  residual = latent-position cross-entropy in bits/sample, H = empirical label entropy on the same
  windows, recovered = H - residual.  600 LIBERO-spatial windows through the same RLDS pipeline and augmentation
  the arms train on; `--split val` is the RLDS validation split, which the LoRA fine-tune (train=True) never saw.

  | arm | code | split | H | residual | recovered | frac |
  |---|---|---|---|---|---|---|
  | univla | 4 x 16 | **val** | 9.52 | 3.83 | **5.70** | 0.598 |
  | univla | 4 x 16 | train | 9.33 | 2.42 | 6.91 | 0.740 |
  | acdc (pq2x64) | 16 x 64 | **val** | 83.13 | 53.22 | **29.91** | 0.360 |
  | acdc (pq2x64) | 16 x 64 | train | 84.56 | 47.48 | 37.08 | 0.438 |

- **Result.** Held out, our arm recovers **5.2x more bits** of its own label (29.9 vs 5.7) while recovering a smaller
  FRACTION of it (0.36 vs 0.60).  The two arms sit where the earlier diagnosis put them: high-rate and partly
  recoverable versus low-rate and mostly recoverable.
- **The in-sample optimism is now measured, not assumed.** The retraction quoted 79 % / 48 % from end-of-training
  loss; under one instrument the train-split figures are 0.740 / 0.438 and the held-out ones 0.598 / 0.360.  The
  train->val drop is LARGER for UniVLA (-0.142) than for us (-0.078).
- **Instrument defect found and fixed.** The "all tokens agree under one more augmentation draw" stability
  definition returned **0 of 600** for the 16-token arm, so its stable row vanished and no comparison was possible.
  That is the definition degenerating with token count, not a property of the code: at the measured per-token rates
  (0.610 vs 0.126) all-token agreement is 0.61^4 = 0.14 and 0.126^16 = 1e-14.  Replaced by PER-POSITION stability,
  which sums per-position entropies and cross-entropies over the windows where that position is unchanged.  Both
  rates are now printed together, since 0.163 and 0.610 for the same checkpoint are one measurement under two
  definitions and had been quoted interchangeably.  Rerun at n = 1500 for the consistent table.
- **What it does not show.** Recovered bits are not usefulness: they may encode nuisance, and our per-token
  stability is 0.126 against their 0.610.  Whether the information converts is what the 500-episode closed-loop
  evaluations (20740894, 20740897) decide.
- **Artifacts.** `acdc_la/results/token_bits.json`, per-window arrays `token_bits_<arm>_<suite>_<split>.npz`,
  log `acdc_la/logs/bits_20739661.out`.

### 2026-09-15 — FAILED and killed at 18k steps: the variance-normalised deployment prior is gamed by inflating the code

- **Test.** `tc_cont_prior` = `tc_cont` plus `lam_prior * ||z - p(z|F_0, language)||^2 / var(z)`, the only term in the
  codebase that prices the measured bottleneck P = R2(z | deployment state) directly.
- **Result, at 18,200 of 128,000 steps (job 20738376).**

  | | tc_cont (no prior) | tc_cont_prior |
  |---|---|---|
  | z_rms trajectory | 1.64 -> 3.61, flattening | **1.37 -> 7.81 -> 10.98 -> 13.79, still climbing** |
  | prior_r2 | - | **0.974** |
  | rec | **1.73** | 1.89 |

- **Reading.** A prior R2 of 0.97 is not predictability appearing: the same quantity measured honestly on a trained
  tokenizer is 0.045, and no 18k-step run makes the deployment state determine 97 % of a code.  The denominator was
  being inflated.  Dividing by `var(z)` closes the shrink exploit (which a plain MSE has, and which I fixed earlier
  today) and OPENS an inflate exploit: add a large component the prior can copy, the denominator grows, the numerator
  does not.  Reconstruction getting worse than the no-prior control is the corroborating signature -- code capacity
  spent on the decoy.  This is a design error in my formulation, not an implementation bug.
- **Fix.** `--code-norm` pins the per-token code RMS to sqrt(d), closing BOTH directions, and the prior term reverts
  to a plain MSE.  A decoy component must then be paid for out of a fixed norm budget that reconstruction needs.
  `--lam-prior` now asserts `--code-norm`.  Resubmitted as job 20742783.
- **Confound to declare.** The new prior arm differs from `tc_cont` by TWO factors (code normalisation and the prior
  term).  The clean control would be `tc_cont_norm` (normalisation only).  NOT scheduled: three concurrent jobs
  already saturate the shared OXE cache.  Run it only if the prior arm wins.
- **Throughput, measured.** 0.46 s/step with three jobs streaming the same cache (0.16 s/step with one), so 128k
  steps needs 16.3 h against an 8 h walltime and every leg is cut near 49 %.  The budget cannot be shortened without
  breaking the match to UniVLA's 128k x 64, so two `RESUME=1` legs are chained per tokenizer behind `afterany`
  (tc_cont 20742784 -> 20742785, tc_vq256 20742786 -> 20742787).
- **Watch on tc_vq256.** At step 20,000 only 99 of 256 codes are in use (perplexity 63), i.e. ~6.6 of the requested
  8 bits per token.  Dead-code restart runs every 5,000 steps; if usage does not recover, the native 16x256 arm is
  not delivering the bitrate it was given.

### 2026-09-15 — FINAL bits table (n = 1500, per-position stability): ours recovers 7.5x more nuisance-invariant bits

Supersedes the n = 600 table above, which used the degenerate all-token stability definition.

| arm | split | subset | n | residual | H | recovered | frac |
|---|---|---|---|---|---|---|---|
| univla | **val** | all | 1500 | 4.04 | 9.44 | 5.40 | 0.572 |
| univla | **val** | **stable positions** | 920 | 2.29 | 6.77 | **4.47** | 0.661 |
| univla | train | all | 1500 | 2.36 | 9.38 | 7.02 | 0.749 |
| univla | train | stable positions | 935 | 1.28 | 6.81 | 5.53 | 0.812 |
| acdc | **val** | all | 1500 | 52.64 | 83.20 | 30.57 | 0.367 |
| acdc | **val** | **stable positions** | 133 | 39.37 | 73.04 | **33.66** | 0.461 |
| acdc | train | all | 1500 | 47.54 | 85.06 | 37.53 | 0.441 |
| acdc | train | stable positions | 145 | 33.83 | 76.70 | 42.87 | 0.559 |

Per-token stability: univla 0.579 (val) / 0.596 (train); acdc 0.117 / 0.118.  All-token: 0.140 / 0.141; 0.000 / 0.000.

- **Result, held out.**  Over all positions ours recovers **5.7x** more bits of its own label (30.6 vs 5.4); over
  positions whose code is invariant to one further augmentation draw, **7.5x** (33.7 vs 4.5).  It does so while
  recovering a smaller FRACTION (0.37 vs 0.57), and while being 5x less nuisance-stable per token (0.117 vs 0.579).
- **The estimator bias runs AGAINST us, so the ratio is conservative.**  The stable subset is 133 of 1500 windows per
  position for our 16 x 64 code against 920 of 1500 for their 4 x 16 one.  The plug-in entropy estimate is biased low
  by about (K-1)/(2N ln2) per position: ~0.34 bits x 16 = 5.5 bits for us, ~0.012 x 4 = 0.05 bits for them.  H low
  means recovered and frac low, so our 33.66 and 0.461 are lower bounds and theirs are essentially unbiased.
- **Generalisation.**  train -> val recovered bits: univla 7.02 -> 5.40 (-23 %), acdc 37.53 -> 30.57 (-19 %); on
  stable positions -19 % and -21 %.  Comparable, with no advantage to either.
- **Scope, and it matters.**  This is the POST-HOC pq2x64 arm on the old quadrant tokenizer -- the arm the
  2026-09-15 scheduling reset deprioritised.  It says the information route carries more nuisance-invariant
  signal to a policy; it does not say the main model wins.  The main line is `tc_cont`, and the verdict is the
  500-episode closed-loop (20740894, 20740897), not this table.
- **Artifacts.** `acdc_la/results/token_bits.json`, arrays `token_bits_<arm>_<suite>_<split>.npz`, log
  `acdc_la/logs/bits_20741108.out`.

### 2026-09-15 — FIRST VALID full-VLA closed loop: matched UniVLA 94.8 % on LIBERO-spatial, and the suite is saturated

- **Result (job 20740894, 50 trials x 10 tasks = 500 episodes, their evaluator, per-step query + temporal ensemble).**
  **474 / 500 = 94.8 %.**  Per task: 0.90, 0.98, 0.96, **1.00**, 0.86, 0.96, **1.00**, 0.96, **1.00**, 0.86.
- **This is the first valid full-VLA number in the project.**  Everything before it was either the MLP toy harness or
  the evaluator bug that scored 181 crashes as failures (retracted earlier today).
- **The baseline is strong and the suite is saturated.**  Three of ten tasks are at 100 % and the floor is 0.86, so
  there is no headroom to measure a better tokenizer in.  The acdc arm (job 20740897, post-hoc pq2x64) is tracking
  at 95.0 % through four tasks with an identical first task (0.90 vs 0.90).
- **The distinction that must not be collapsed.**  If the two arms finish tied, that is "this measurement cannot
  separate them", NOT "the extra information is useless".  The bits table of today shows our route delivers 7.5x
  more nuisance-invariant recovered bits; a saturated success rate gives that information nowhere to show up.
- **Consequence, prepared but not launched.**  Fine-tune on fewer demonstrations instead of swapping suite: the
  split counts are verified (`train[:95%]` = 410 episodes, `train[:25%]` = 108, `train[:10%]` = 43, held-out
  `train[95%:]` = 22).  `LIBERO_TRAIN_FRAC` is patched in and smoke-tested end to end (job 20743597, exit 0).
  Cheaper than a suite swap in reviewer terms too: it is the axis UniVLA's own paper reports, and it targets the
  mechanism we claim rather than a different benchmark.  Note that action normalisation statistics are NOT
  recomputed per split -- every arm loads the same cached `dataset_statistics_eb1c44` -- which keeps arms
  comparable but must be stated in the method.
- **Also verified today.** VLA-arm plumbing for both structured tokenizers against partial checkpoints:
  `tc_cont` (job 20743222, exit 0) and `tc_vq` (job 20743223, exit 0, token acc 0.016 against a 1/256 chance).
- **Artifacts.** `acdc_la/logs/vlaeval_20740894.out`, `acdc_la/results/vla_eval/`.

### 2026-09-15 — `tc_cont_prior` cancelled by the user shortly after it started; what that decision was based on

- **Action.** Job 20742783 (the code-norm rebuild of the deployment-prior arm) cancelled at ~25 minutes in.
- **Why its value had fallen since it was queued.**  (a) It differs from `tc_cont` by TWO factors -- the pinned code
  scale and the prior term -- so a win could not be attributed without a third run (`tc_cont_norm`).  (b) Its whole
  purpose is to raise how much of the code the deployment state determines, but the benchmark that would show the
  benefit, LIBERO-spatial at full data, is saturated: matched UniVLA 474/500 = 94.8 %, three tasks at 100 %, and four
  of the first six tasks identical between the two arms.  (c) It was occupying one of three GPUs against the
  deadline path.
- **Measured effect of the cancellation.** The queue estimate for the next dependent job moved from 2026-09-16 18:40
  to 2026-09-15 19:12.
- **Kept deliberately.** The second `RESUME=1` legs (20742785, 20742787) hold no resources while pending and are
  cheap insurance: `tc_cont` is at ~60k of 128k steps with ~25 minutes of walltime left, so it needs both legs.
  A leg that starts after the tag has already reached 128k returns immediately without retraining.
- **Still open and more valuable than this arm was.** There is no full-VLA NO-LATENT control.  With both latent arms
  at 94-95 %, we cannot say whether a plain `openvla-7b` + LoRA + action head reaches the same number, and if it
  does the entire LIBERO table shows nothing about latent actions at all.  About 10 GPU-hours.

### 2026-09-15 — the full-data LIBERO comparison is a TIE, and the tie is the benchmarking result

- **Both arms, 500 episodes each, their evaluator (per-step query + temporal ensemble), identical VLA recipe.**

  | | univla (20740894) | acdc pq2x64 (20740897) |
  |---|---|---|
  | total | **474 / 500 = 94.8 %** | **472 / 500 = 94.4 %** |

  Per task (univla, acdc, diff): 0.90/0.90 **+0.00** | 0.98/0.98 **+0.00** | 0.96/0.98 +0.02 | 1.00/0.94 -0.06 |
  0.86/0.86 **+0.00** | 0.96/0.96 **+0.00** | 1.00/0.98 -0.02 | 0.96/0.92 -0.04 | 1.00/0.98 -0.02 | 0.86/0.94 +0.08
- **Two episodes out of five hundred separate them, and four of ten tasks are exactly equal.**  Three tasks are at
  1.00 for UniVLA.  The two tokenizers differ by everything a tokenizer can differ by -- 4x16 vs 16x64 codes, 9.4 vs
  83.2 bits of label entropy, 0.579 vs 0.117 per-token augmentation stability, 5.4 vs 30.6 bits recovered at
  deployment -- and the benchmark returns the same number.
- **This is a result about the benchmark, not about the methods.**  Full-data LIBERO-spatial success rate is not a
  measurement of the latent action model.  The paper states it as such, and it is what licenses changing the
  measurement rather than the change looking like a response to losing.
- **It also sets up the prediction the method section is tested on.**  If the extra recoverable, nuisance-invariant
  information is real, it has to pay where information is scarce.  Hence the data-efficiency arm at 43 of 410
  demonstrations, which is now the main chain: `tc_cont` -> gate -> VLA@10 % -> 500-episode evaluation, with the
  matched UniVLA@10 % baseline already queued (20744452).  The full-data `tc_cont` run is retained as a tail
  (20746342/3) to add its own row to the tie.
- **Artifacts.** `acdc_la/logs/vlaeval_20740894.out`, `vlaeval_20740897.out`.

### 2026-09-15 — FIRST POSITIVE METHOD RESULT: LATP v2 improves the encoder, with every cheating probe negative

- **Setup.** `latp2_anchor` = `stage3_transport/train_v2.py --init-from tc_cont --lam-effect 1 --lam-recover 0
  --lam-rel 0.5`, 15,000 steps, batch 64.  The anchored cycle: an ONLINE code realised on another row's state must
  carry that donor's REAL effect descriptor Psi = [N, A1], fixed and whitened from real data.
- **Encoder probes (OXE val, the same per-source episode-disjoint ridge as every other row in this ledger).**

  | | dims | r2_net | seg1 | prof | seq | patch_drop |
  |---|---|---|---|---|---|---|
  | `tc_cont` (the starting point, 65k of 128k) | 256 | 0.265 | 0.232 | 0.091 | 0.192 | 22.4 % |
  | **`latp2_anchor`** | 256 | **0.300** | **0.267** | 0.085 | **0.216** | **19.7 %** |
  | `s1_acdc_1ep` (old tokenizer, full budget) | 256 | 0.324 | 0.264 | 0.085 | 0.224 | 28.9 % |

  Action information rose on net (+0.035), seg1 (+0.035) and seq (+0.024), profile fell 0.006, and nuisance
  robustness improved to the best of the three.  seg1 now exceeds the full-budget old tokenizer.
- **Why it matters.** LATP v1 froze the tokenizer and trained only the transport operator, and the VLA labeller
  calls only the encoder -- so v1's gains could not reach a policy by construction.  This is the first evidence that
  a transport objective can improve the part of the model that is actually deployed, and it did so WITHOUT the
  feared cost: the concern was that the transport objective would drag the encoder off its action content.
- **Every instrumented failure mode came back negative (step 15,000).**

  | probe | value | reading |
  |---|---|---|
  | `effect_cos` / `effect_shuffle` | 0.564 / 0.016 | |
  | **`effect_margin`** | **0.548** | the anchored term really carries the donor's effect |
  | **`replay_cos`** | **0.051** | NOT replay: the realisation is near-orthogonal to the recipient's own motion |
  | `z_erank` | 54.0 of 64 | no collapse |
  | `ctx_sens` | 0.465 | the operator still uses the state; RTC is not satisfied by ignoring it |
  | `rtc_cos` | 0.945 | the relation between two interventions transfers in direction |
  | `mmd` | -0.006 | synthetic transitions are distributionally indistinguishable from real ones |

  `replay_cos` is the surprise: I had written off the replay solution as the unavoidable residual degeneracy, and it
  did not happen -- the transported effect correlates with the DONOR (0.56) and not with the recipient (0.05).
- **The confound, and the control now running.** `latp2_anchor` saw 15,000 optimiser steps that `tc_cont` did not,
  and `tc_cont` was stopped at 51 % of budget and still improving.  The gain may be extra training rather than the
  objective.  `tc_cont_plus15k` (job 20752402) is the control: same checkpoint, same 15,000 steps, same fresh
  schedule, plain reconstruction objective.  **No claim until that lands.**
- **A negative signal to report alongside.**  `rec_frozen` falls 0.90 -> 0.27 over training: the encoder can no
  longer read codes out of the FROZEN Phase-1 operator's output, i.e. it has co-adapted with the current operator
  rather than learning an operator-independent reading.  That is the inversion signature the probe was built to
  catch, and it did appear.  With `lam_recover = 0` it is not optimised and the encoder probes say it did no harm,
  but it belongs in the paper.
- **Bug found while building the control.** `run()` prefers `build()` over its own `--init-from` branch, so
  `stage2_tc --init-from` was **silently ignored** and would have trained from scratch.  Fixed in `build()`.  Same
  class as the `tc_tag` and knob-accepted-and-ignored failures earlier in this project.
- **Artifacts.** `acdc_la/logs/long_20746301.out`, `acdc_la/logs/probe_latp2.out`.

### 2026-09-16 — LATP v2 CONFIRMED: the encoder gain is the objective, not the extra training

- **The control.** `tc_cont_plus15k` (job 20754166): the same `tc_cont` checkpoint, the same 15,000 steps, the same
  fresh cosine schedule, plain reconstruction objective.  Probed identically.

  | | objective | r2_net | seg1 | prof | seq | patch_drop |
  |---|---|---|---|---|---|---|
  | `tc_cont` | (starting point, 65k of 128k) | 0.265 | 0.232 | 0.091 | 0.192 | 22.4 % |
  | **`tc_cont_plus15k`** | **plain reconstruction, +15k** | **0.268** | 0.240 | 0.091 | 0.193 | **22.6 %** |
  | `latp2_anchor` | anchored transport | 0.300 | 0.267 | 0.085 | 0.216 | 19.7 % |
  | **`latp2_inv`** | anchored + nuisance invariance | **0.329** | **0.283** | **0.091** | **0.236** | **17.9 %** |

- **Attribution.** 15,000 more steps of reconstruction buys **+0.003** on r2_net and **nothing** on nuisance
  robustness (22.4 -> 22.6 %, slightly worse).  The transport objective buys **+0.035**, and with the invariance
  term **+0.064** -- twenty-one times the control -- while nuisance drop improves by 4.5 points.  The gain is the
  objective.
- **`latp2_inv` beats the FULL-budget old tokenizer on every metric, from a half-budget starting point.**

  | | `s1_acdc_1ep` (128k steps) | `latp2_inv` (from a 65k checkpoint) |
  |---|---|---|
  | r2_net / seg1 / prof / seq | 0.324 / 0.264 / 0.085 / 0.224 | **0.329 / 0.283 / 0.091 / 0.236** |
  | patch_drop | 28.9 % | **17.9 %** |
  | LIBERO latent_r2 (deployment predictability) | 0.50 | **0.730** |
  | LIBERO latent_acc above its majority floor | 0.469 | **0.621** |

- **Why this closes the v2 argument.**  Stage III froze the tokenizer and the VLA labeller calls only the encoder, so
  v1's gains could not reach a policy by construction.  v2 makes the encoder trainable; the anchored cycle blocks the
  vacuous solution that the code-space cycle admits (0.98 on semantics-free tensors); the encoder improves; and the
  control says it is the objective.  Every instrumented failure mode is negative on the anchor arm (`effect_margin`
  0.548, `replay_cos` 0.051, `z_erank` 54/64, `ctx_sens` 0.465).
- **On `latp2_inv`'s collapse alarm.**  `rec_margin` fell 0.686 -> 0.099 and `z_erank` 54 -> 48, which reads as
  partial collapse -- but every action probe ROSE.  The degeneracy is in the E-after-T inversion channel, not in the
  code's content.  My earlier reading of that alarm as a content collapse was wrong, and the probes that decide it
  are the action ones.
- **What is still missing.**  None of this reaches a policy yet.  Offline chain L1 moves 0.427 -> 0.423 while the
  whole spread from the best latent to NO LATENT AT ALL is 0.012 (0.421 vs 0.433), so that harness cannot resolve it
  either -- the second independent confirmation, after the full-data closed-loop tie, that the field's measurements
  are insensitive to what they are meant to measure.  The data-efficiency pair at 43 demonstrations remains the only
  in-budget test with headroom.
- **Artifacts.** `acdc_la/logs/long_20754166.out`, `acdc_la/logs/probe_control.out`, `acdc_la/logs/tcgate_20747389.out`.

### 2026-09-16 — spatial resolution curve: patch-level evidence is the WORST, and the most nuisance-sensitive by far

- **Question.** Every tokenizer here reads a 4x4 DINO grid made by a fixed 4x4 mean pool of the 16x16 patch grid,
  while UniVLA's LAM reads the patch grid.  How much task information does that shared pooling destroy?  The pooling is
  linear, so it commutes with the temporal decomposition (N~ = P(N), A1~ = P(A1)) and cannot touch the Net/A1
  argument, and it cannot confound tc_cont against the old ACDC (both read the same cache).  What it can remove is
  spatial frequency -- and possibly nuisance.
- **Setup (job 20758828, `common/spatial_curve.py`).** 11,472 OXE validation clips, 29 sources.  DINO patch tokens
  extracted once per resolution and pooled to g x g; the same [N, A1] evidence the tokenizer is given; every resolution
  PCA'd to 256 dims; per-source episode-disjoint ridge; repeated under the moving-patch distractor.  No tokenizer, no training.

  | | 16x16 = 256 cells | 8x8 = 64 | **4x4 = 16 (ours)** | 2x2 = 4 (old ACDC) |
  |---|---|---|---|---|
  | r2_net | **0.325** | 0.342 | **0.358** | 0.350 |
  | r2_seg1 | 0.269 | **0.283** | 0.258 | 0.276 |
  | r2_prof | 0.065 | 0.070 | **0.073** | 0.062 |
  | r2_seq | 0.233 | 0.245 | **0.254** | 0.233 |
  | r2_net under distractor | **0.023** | 0.140 | **0.201** | 0.190 |
  | **patch_drop** (ratio of means; CORRECTED, see below) | **93.0 %** | 59.0 % | **44.0 %** | 45.7 % |

- **Result.** Patch-level evidence reads the LEAST action of the four on net displacement and is catastrophically
  nuisance-sensitive: under the distractor its net readout collapses from 0.325 to 0.023.  Nuisance sensitivity falls
  monotonically from 256 to 16 cells (93.0 -> 59.0 -> 44.0 %) and is then FLAT at 4 cells (45.7 %, +1.7 points, and
  the clean net readout differs by 0.008).  The evidence is the large step down to 4x4, NOT a sweet spot: there is no
  meaningful degradation at 2x2 to make the curve U-shaped.  The shared pooling is not a demonstrated liability in
  this measurement; it behaves like a spatial low-pass that removes more nuisance than action.
- **The same trade the paper makes about latent capacity, one level down.**  More spatial detail raises what the
  evidence could encode and admits background texture and distractor motion with it.  256 > 16 is not a given, and
  for a linearly read action it is false here.
- **Caveats, and the first is serious.**  (1) PCA to 256 dims compresses the 16x16 evidence 1,536:1 and the 4x4
  evidence 96:1, so the clean-readout columns partly measure what survives PCA; a nonlinear encoder over all 256
  patches could extract more.  The patch_drop column is less exposed, because it compares clean and distractor
  readouts WITHIN one resolution through the same PCA.  (2) A linear ridge, not a transformer.  (3) One nuisance
  family (a moving natural-video patch).  So the claim is "patch-level evidence is much more nuisance-sensitive and
  no better for a linear action readout", not "patch level is worse for a learned tokenizer".
- **Consequence for the paper.**  The limitation stays in, stated as agreed ("all variants operate on a shared 4x4
  DINO grid ... our spatial ablations evaluate 2x2 vs 4x4 aggregation, not patch-level modelling"), and this curve
  goes next to it as evidence that the choice sits at the robust end of a measured trade-off rather than being an
  unexamined shortcut.
- **Artifacts.** `acdc_la/results/spatial_curve.json`, `acdc_la/logs/spatial_20758828.out`.  The first attempt
  (job 20750817) held all four resolutions in memory and was OOM-killed.

### 2026-09-16 — CORRECTION to the spatial curve entry: its patch_drop column was a different statistic

- **What was wrong.** `common/spatial_curve.py` computed the drop per source and averaged those ratios, over only the
  sources whose clean r2 exceeded 0.02 (24 of 29 at 16x16).  Every tokenizer table in this ledger computes it from the
  aggregated means, `1 - mean(r2_net_patch) / mean(r2_net)` (`stage1_preserve/probes.py::on_code`).  So the curve's
  drop column was not comparable with the tokenizer drop columns (17.9 / 19.7 / 22.4 / 28.9 %), against the rule set
  on 2026-09-15 that a comparison needs one instrument.  Noticed because 0.325 -> 0.023 is a 93 % drop, not 66.5 %.
- **Why the per-source form is also a bad statistic here.** A per-source ratio divides by a clean r2 that can sit just
  above the 0.02 floor.  Across sources it ranged -11.4 % .. +309.6 % at 16x16 and -4.6 % .. +406.8 % at 8x8.
- **Recomputed from the stored aggregated means (no rerun needed; both inputs were already in the json).**

  | | 16x16 | 8x8 | 4x4 | 2x2 |
  |---|---|---|---|---|
  | reported first (mean of per-source ratios) | 66.5 % | 53.9 % | 43.7 % | 47.6 % |
  | **corrected (ratio of means, as in the tokenizer tables)** | **93.0 %** | **59.0 %** | **44.0 %** | **45.7 %** |

- **What changes in the reading.** (1) The main result STRENGTHENS: patch-level evidence loses 93 % of its net readout
  to the distractor, against 44 % for the 4x4 grid.  (2) The "U-shaped sweet spot at 4x4" reading WEAKENS: the rise
  to 2x2 is 1.7 points, not 3.9, and the clean readout differs by 0.008.  The defensible statement is "robustness
  improves monotonically down to 4x4 and is flat below it", not "4x4 is an optimum".
- **Label.** I had also called the column "accuracy lost to the distractor" in conversation.  It is a relative R2 drop.
- **Fix.** The script now reports the ratio of means as `patch_drop_pct` and keeps the old statistic, renamed
  `patch_drop_pct_mean_of_ratios`; `results/spatial_curve.json` was rewritten the same way.

### 2026-09-16 — two corrections: the probe split was never leave-one-source-out; and 10 % data is NOT saturated

**Correction 1 — every OXE R2 in this ledger is a per-source, episode-disjoint probe.**
- `common/readers.py::source_splits` loops over sources and, WITHIN each source, puts 1/3 of its episodes in test
  (RandomState(1)) and 2/3 in train.  A separate ridge is fit per source; results are clip-weighted over sources with
  >= 80 test clips.  I described it as "leave-one-source-out" in the predictability, retraction, LATP and spatial
  entries and in conversation.  It never was.  Fixed in place in those entries and in two docstrings.
- **Scope of every probe claim, stated correctly:** generalisation to UNSEEN EPISODES of SEEN sources -- not to
  unseen embodiments.  An easier test than the one I kept describing.
- **One retraction reason was false and is withdrawn.**  The 2026-09-15 retraction of "UniVLA's latent is
  unrecoverable" gave four reasons; reason (2), "leave-one-source-out = cross-embodiment extrapolation", is false.  The
  retraction's conclusion still stands on reasons (1) embedding space vs indices, (3) no language, (4) weak reader.
- **Full protocol, for the paper.**  Data: OXE validation episodes (fixed per-source `is_val` flag, never seen by any
  tokenizer), stratified to 200 anchor/partner pairs = 400 clips per source, 29 sources, ~11.5k clips.  Target: xyz of
  the per-segment action (summed per-step deltas for delta sources, pose difference for absolute-pose sources; rotation
  and gripper not probed) -> net = sum over the 4 segments (3-d), seg1 (3-d), seq (12-d), prof = seq minus its clip mean
  (12-d).  Reader: features standardised and clipped at +-10 sd, ridge alpha in {1e-1..1e4} chosen on an inner 1/5
  episode split.  R2: 1 - SSE/SST per target dim on test rows, averaged over dims that vary.  ~267 train clips per
  source against 256-d features, so the ridge penalty is doing real work.

**Correction 2 — the UniVLA 10 %-data run is NOT saturated; my extrapolation from two tasks was wrong.**
- Job 20759214, `univla` fine-tuned on `train[:10%]` = 43 demonstrations, 500 episodes:
  **421 / 500 = 84.2 %**, against 474 / 500 = 94.8 % on 410 demonstrations.

  | task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 410 demos | 0.90 | 0.98 | 0.96 | 1.00 | 0.86 | 0.96 | 1.00 | 0.96 | 1.00 | 0.86 |
  | **43 demos** | 0.92 | 0.98 | 0.96 | **0.78** | **0.66** | 0.98 | 0.96 | 0.90 | 0.88 | **0.40** |

- I had called 10 % "also saturated" from the first two tasks (0.92, 0.98), which are two of the easiest, and
  recommended cancelling our arm on that basis.  The hard tasks fall a long way: task 10 0.86 -> 0.40, task 5
  0.86 -> 0.66, task 4 1.00 -> 0.78.  **This is the headroom the data-efficiency comparison needed.**
- **Consequence.** The queued `tc_cont@10%` VLA run (20754114) and its evaluation (20754115) are now the most valuable
  jobs in the queue, not a long shot.  They stay.  Its labeller now reads `z_stats_tc_cont.npz` (written today; the run
  would otherwise have trained on unstandardised targets).  Steps-to-success is logged for both arms.

### 2026-09-16 (night) — main table first (user); deadline 2026-09-24; seven main-table VLA runs queued; the continuous-arm evaluator was broken and is fixed before it produced anything

- **Decision (user).** Ablations are deprioritised; the paper needs its main table.  Deadline is ICLR 2026-09-24.
  The plan (sections, tables, figures, job map, open decision) is `PAPER_PLAN.md`.
- **Main table.** Controlled VLA post-training at **10 % of the demonstrations on all four LIBERO suites**, UniVLA
  same-data latent (`s2mix_1ep`) vs ours (`latp2_inv`), identical recipe (15k x 64, their evaluator, 500 episodes).
  The 10 % split covers every task in every suite (demos per task, sorted by instruction): spatial 43 total (2-9),
  object 45 (1-7), goal 43 (2-7), libero_10 38 (1-7).  Held-out `train[95%:]` is untouched.
- **Queue swap, and how.** `scontrol top` is denied to users on Anvil, and a pending job's environment is frozen.
  `finetune_libero_ctrl.py` now reads `downstream/vla/job_overrides/<SLURM_JOB_ID>.json` before anything reads cfg,
  prints it, and records it in `ctrl_meta.json` (which now also stores `tc_tag`, `dataset_name`, `train_frac`).
  20754114 (priority 2942, submitted as `tc_cont`@10 % spatial) now trains **`latp2_inv`@10 % spatial**; 20775018
  (submitted as `latp2_inv`@10 %) now trains **UniVLA@10 % libero_object**.  Tested on CPU against the real `Cfg`: both
  resolve to the run directories the new evaluation jobs expect; a job without a file is unchanged.
- **BUG, caught before it produced a number (nothing to retract).**  The continuous-arm evaluation had never run for
  a `tc_cont` checkpoint, and would have been wrong in three ways: (1) `run_vla_eval.sh` / `run_vla_loadtest.sh`
  routed only `acdc_cont` to `run_libero_eval_cont.py`, so `tc_cont` checkpoints would have gone to the TOKEN
  evaluator; (2) `run_libero_eval_cont.py` asserted `latent == acdc_cont`; (3) it **did not apply UniVLA's 0.9-area
  center crop**, which their evaluator asserts because training uses random-crop augmentation.  (3) is the dangerous
  one: it would have run without an error while showing our arm a crop no training batch ever looked like, a
  train/test mismatch only our arm would have suffered.  All three fixed; the crop is their `crop_and_resize` call
  and was checked on CPU.  A libero_10 smoke plus load test of both evaluators (20783066 -> 20783067) runs as an early
  warning.  It first gated every evaluation, but Slurm estimated the 1-GPU smoke for 9/18 22:00, so the evaluations
  now depend only on their training job; the evaluators are read at run time, so a failure can still be fixed first.
- **Stale jobs to cancel.** 20754115 and 20775019 are evaluations holding frozen copies of the OLD routing script
  (20775019 would evaluate the `latp2_inv` run through the token evaluator).  My `scancel` was blocked by the
  auto-mode permission check; the user was asked to cancel them.
- **Residual asymmetry, stated, not fixed.** The continuous head pools text hidden states that include right-padding
  in training batches and none at batch-1 evaluation.  Removing it needs retraining.
- **Jobs.**

  | cell | train | eval |
  |---|---|---|
  | spatial / Ours | 20754114 (override) | 20783070 |
  | object / UniVLA, Ours | 20775018 (override), 20783072 | 20783071, 20783073 |
  | goal / Ours, UniVLA | 20783074, 20783076 | 20783075, 20783077 |
  | libero_10 / Ours, UniVLA | 20783078, 20783080 | 20783079, 20783081 |
  | LATP on `tc_acdc` (`latp2_acdc_inv`, same flags as `latp2_inv`) -> probes vs `tc_acdc`, `latp2_inv`, `tc_cont` + z-stats | 20783064 -> 20783065 | |

  `train_v2.step` was checked on CPU against the `tc_acdc` architecture (netdyn, structured queries, 8 x 32) before
  submission.  Intermediate `_stepN` snapshots do not carry the architecture config and cannot initialise LATP; the
  final `tc_acdc.pt` does.
- **Open decision.** Which base is "Ours": `latp2_inv` (`tc_cont`: F0 in the encoder, 4 x 64) or `latp2_acdc_inv`
  (the clean ACDC definition).  Proposed rule in `PAPER_PLAN.md`; the not-yet-started Ours runs can be re-targeted
  by override.

### 2026-09-17 (00:10) — `tc_acdc` stopped by walltime at 127.2k; the 125k checkpoint is used; the queue puts Goal and Long past the deadline

- **Walltime.** 20747388 (gpu partition, `-t 08:00:00`) was cancelled at step 127,200 of 128,000.  The last full
  checkpoint, `ckpt/tc_acdc.pt`, is **step 125,000 (97.7 % of budget)**.  The in-training curve has been flat since
  50k (net 0.129 at 50k, 0.137 at 120k; drop 44 % -> 35 %).  Decision: use 125k rather than wait for a 1-GPU slot
  to run the last 3k steps (the 1-GPU smoke is estimated at 9/18 17:30).  The paper states 125k of 128k, or the
  run is resumed later.
- **Chain repaired.** 20781752 (probes: `tc_acdc`, `tc_cont`, `latp2_inv`) and 20783064 (`latp2_acdc_inv`) had
  `afterok:20747388`, which can no longer be satisfied; both dependencies were cleared, so they run from the 125k
  checkpoint.  20783065 still follows 20783064.
- **Main-table progress.** 20754114 (`latp2_inv` @ 10 % spatial, override confirmed in the log): step 7,250 after
  1 h 53 min, 0.93 s/step on h008, so training ends around 02:10.  Train-batch latent_r2 is 0.75-0.78.
- **Queue risk.** The ai partition has 303 pending jobs and 64 GPUs running.  Priority is age plus partition only
  (sprio shows no fairshare term), so order is effectively submission order.  Our pending jobs rank 54 (UniVLA
  object), 90 (Ours object), 96 (Ours goal) and 146-148 (UniVLA goal, both Long).  Slurm estimates: 9/18 06:10,
  9/21 21:00, 9/22 09:00, and no estimate for the rest.  Evaluations enter the queue at priority 75 when their
  training ends.  **At these estimates Goal and Long do not finish before 2026-09-24.**  Estimates have been
  pessimistic (20754114 was estimated for 9/17 06:00 and started 9/16 22:05), but not by three days.  Options put
  to the user: run Goal and Long on the workstation (2 x RTX PRO 6000, ~13 h per training arm), or shrink the main
  table to the suites that finish.

### 2026-09-17 (11:30) — clean ACDC (`tc_acdc`, 125k) reads LESS action than `tc_cont` and than the old tokenizer; Ours@10 % spatial trained; libero_10 smoke passed

- **Probes (job 20781752; OXE val, 400 clips/source, per-source episode-disjoint ridge, the same protocol as every row
  in this ledger).  The `tc_acdc` dump was written by `--eval-only` from `ckpt/tc_acdc.pt` = step 125,000; the
  architecture comes from the checkpoint's stored `tcvq` config.**

  | | steps | r2_net | seg1 | prof | seq | patch_drop | P(z given state) |
  |---|---|---|---|---|---|---|---|
  | **`tc_acdc`** (clean ACDC: netdyn, 8 x 32 structured slots, no F0 in the encoder) | 125k | **0.223** | **0.199** | 0.074 | **0.157** | **31.9 %** | 0.018 |
  | `tc_cont` (endpointdyn, 4 x 64, F0 in the encoder) | 65k | 0.265 | 0.232 | 0.091 | 0.192 | 22.4 % | 0.019 |
  | `latp2_inv` (LATP v2 on `tc_cont`) | 65k + 15k | 0.329 | 0.283 | 0.091 | 0.236 | 17.9 % | 0.224 |
  | `s1_acdc_1ep` (old quadrant + MLP tokenizer) | 128k | 0.324 | 0.264 | 0.085 | 0.224 | 28.9 % | 0.045 |
  | `s1_mean_1ep` (pooled) | 128k | 0.336 | 0.210 | 0.017 | 0.200 | 33.6 % | 0.087 |

- **Reading.** `tc_acdc` is below `tc_cont` on every column (net -0.042, seg1 -0.033, prof -0.017, seq -0.035,
  distractor drop +9.5 points) despite ~2x the optimiser steps, and below the old tokenizer on net, seg1 and seq.
  **Confounded, so no single cause is claimed:** it differs from `tc_cont` in three ways at once (no state/F0 evidence
  in the encoder; 8 x 32 structured slots vs 4 x 64; netdyn vs endpointdyn evidence).  Deployment predictability is the
  same (0.018 vs 0.019).  The in-training curve (net ~0.13 on 200 clips/source) had already pointed this way.
- **Consequence for the open "which base is Ours" decision.** The proposed rule required LATP on `tc_acdc` to land
  within 0.01 of `latp2_inv`.  `tc_acdc` starts 0.04-0.06 below `tc_cont`, and LATP added +0.064 on `tc_cont`.
  Recommendation to the user: keep `latp2_inv` as Ours and describe the `tc_cont` architecture as it is.  20783064
  (LATP on `tc_acdc`) is kept, as evidence that LATP helps a second base, not as the main model.
- **Main table.** 20754114 finished: 15k steps in 3 h 53 min, train-batch latent_r2 0.826, action_l1 0.0214;
  `ckpt/vla_runs/tc_cont+libero_spatial_no_noops+b16x4+lr0.00035+r32+s15000--latp2inv_frac10`.  Its evaluation
  20783070 is pending (1 GPU, no Slurm estimate yet).  UniVLA object (20775018) started 11:01 on h016.  Estimates:
  Ours object 9/18 00:20, Ours goal 9/18 00:55.
- **libero_10 smoke (20783066): both arms exit 0**, dataset statistics computed (12-step numbers validate the code path
  only).  The evaluator load test 20783067 is still pending.
- **Guard added.** `run_libero_eval_ctrl.py` now refuses a checkpoint with no latent vocabulary (continuous arm).  The
  stale 20775019 (old routing script, frozen at submission) becomes eligible when 20775018 finishes today and would
  have evaluated the `latp2_inv` spatial checkpoint with the token evaluator; it now fails at startup.  The user has
  been asked to cancel it and 20754115.

### 2026-09-17 (20:30) — the evaluations were queued for 2026-09-24; split into per-task jobs so they backfill

- **Training is fine, the queue is not.** UniVLA @ 10 % object (20775018) COMPLETED 16:34 (5 h 33 min); Ours object
  (20783072) started 19:34 on h010 and Ours goal (20783074) 19:59 on h004.  But the two 1-GPU evaluations of the
  finished checkpoints, 20783070 (Ours spatial) and 20783071 (UniVLA object), were estimated to start
  **2026-09-24 05:00 and 07:00** — the deadline itself.  Trained checkpoints with no numbers is the worst failure mode
  available at this point.
- **Fix: evaluate a task subset per job.**  `LIBERO_TASK_IDS` (comma or underscore separated, because `sbatch --export`
  splits on commas) in both evaluators, carried into the run id / output filename; `NOTE` in `run_vla_eval.sh` names the
  output files (the old default, the first 40 characters of the run dir, is NOT unique across arms and would have made
  two arms' logs merge); `downstream/vla/eval_merge.py` merges the parts and refuses a task counted twice.
- **What the split does and does not change.**  Each task keeps its own 50 initial states and every task is evaluated by
  exactly one job, so the per-task numbers are the protocol's.  The only difference is the token arm's sampling RNG
  stream (`do_sample`, temperature 0.75), which starts fresh per job instead of continuing through the suite.
- **Submitted** (5 jobs x 2 tasks = 100 episodes each, 1.5 h limit, h004/h006 excluded):
  Ours spatial 20810386-20810390 (`NOTE=ours_spatial_f10`), UniVLA object 20810391-20810395 (`NOTE=univla_object_f10`).
  The whole-suite jobs 20783070/20783071 stay queued as a backup; `eval_merge` defaults to parts-only so they cannot be
  double-counted.
  Merge: `python -m acdc_la.downstream.vla.eval_merge --suite libero_spatial --note ours_spatial_f10 --kind cont`.
- **Housekeeping.** The user cancelled the two stale evaluations (20754115, 20775019) at 14:38 and pushed `main` to
  GitHub; the Spatial evaluation's walltime was cut 6 h -> 4 h earlier for the same backfill reason.

### 2026-09-18 (09:45) — MAIN TABLE, first cells: **Ours LOSES to UniVLA at 10 % demos on Spatial (70.4 vs 84.2) and Object (78.2 vs 82.8)**

- **Results (500 episodes each, their evaluator, per-step query + temporal ensembling; single training seed per arm).**

  | suite | UniVLA same-data latent (token interface) | Ours `latp2_inv` (continuous interface) | diff |
  |---|---|---|---|
  | Spatial | **421/500 = 84.2 %** (20759214) | 352/500 = 70.4 % (20783070) | **-13.8** |
  | Object | **414/500 = 82.8 %** (20783071) | 391/500 = 78.2 % (20783073) | -4.6 |

  Per task, spatial (UniVLA / Ours): 0.92/0.80, 0.98/0.74, 0.96/0.76, 0.78/**0.96**, 0.66/0.54, 0.98/0.98, 0.96/0.98,
  **0.90/0.10** (bowl on the stove), 0.88/0.86, 0.40/0.32.
  Per task, object: **0.34/0.80** (alphabet soup), 0.84/0.54, 0.82/0.76, 0.72/0.68, 0.94/0.92, 0.80/0.64, 0.98/0.84,
  0.88/0.84, 1.00/0.90, 0.96/0.90.
- **Reading.**  Ours is worse on both suites.  The spatial gap is 14 points against a standard error of about 2.5 for a
  difference of two 500-episode rates, so it is not evaluation noise; 8 of the 14 points come from one task
  (stove, 0.90 -> 0.10), and without it ours is still ~6 points lower.  The object gap (4.6) is under 2 SE.  **Both arms
  have one collapsed task** (Ours: stove 0.10; UniVLA: alphabet soup 0.34 where Ours gets 0.80), which says training
  variance at 43 demonstrations is large and one seed per arm cannot resolve differences of a few points.
- **Training fit runs the other way.**  Final train-batch action L1: Ours 0.021 (spatial) / 0.019 (object) vs UniVLA
  0.027 / 0.028.  Our arm fits the few demonstrations more tightly and does worse in closed loop.  Train-batch numbers,
  not held-out; consistent with overfitting of the richer conditioning or of the regression head, not proof of it.
- **The comparison is confounded, and this must be said wherever the numbers go:** the two arms differ in the latent AND
  in the VLA interface.  UniVLA's arm predicts discrete latent tokens with the LM head (with the history prompt) and
  decodes from their hidden states; ours regresses a continuous code from pooled prompt states (`ContHeads`) — the
  plain-regression setting LAFP criticises.  Nothing here separates "our latent is a worse target" from "the continuous
  regression interface is worse".  The no-latent arm through the same `ContHeads` (`--latent none`) is the control that
  separates them; it has never been run.
- **Checked and ruled out as the explanation: prompt padding.**  Training pools over right-padded prompts, evaluation
  has none (asymmetry noted 2026-09-16).  The padding per task is 0-6 tokens and does not track the losses (the stove
  task and three tasks with 0.86-0.96 success all have 5 pad tokens; the 0-pad task is hard for both arms).
- **Evaluation infrastructure.**  The split evaluation reproduces the full run exactly (tasks 0-3: 40/37/38/48 in both),
  so `LIBERO_TASK_IDS` + `eval_merge.py` are validated; the continuous evaluator is deterministic.  The libero_10 load
  test (20783067) died with SIGABRT on **h003** for both arms: h003 joins h004/h006 as a node to exclude for MuJoCo/EGL.
- **LATP on the clean-ACDC base (20783064 -> 20783065).**  `latp2_acdc_inv`: net 0.238 / seg1 0.207 / prof 0.079 /
  seq 0.165 / drop 30.2 %, P(z|state) 0.048, against `tc_acdc` 0.223 / 0.199 / 0.074 / 0.157 / 31.9 %, 0.018.  +0.015 net
  where it gave +0.064 on `tc_cont`, and no +15k control exists for this base, so not attributable.  `latp2_inv` stays
  the main-table model (the open decision is moot at these numbers).
- **Still running.**  Ours goal evaluation 20783075, UniVLA goal training 20783076; both Long arms queued.

### 2026-09-18 (11:00) — interface change (user): flow-matching head for the continuous arm; Spatial + Object runs and a no-latent control queued

- **Decision (user).** Keep the latent continuous, change the VLA interface.  The field's stated fix for the policy side
  of continuous latent actions is a generative head: LAFP (plain behaviour cloning "tends to collapse inherently
  multimodal action distributions into unimodal ones") and RotVLA (a flow-matching head that "jointly denoises latent and
  robot actions").  Our `ContHeads` is exactly the regression setting LAFP criticises.
- **Also clarified to the user:** the UniVLA arm has no stage-2 (latent-pretraining) advantage and no released weights in
  the deployed path — both arms start from `openvla-7b`, both LAMs were trained on the same capped OXE mixture (theirs
  128k steps, ours 65k + 15k), and 0 of `s2mix_1ep`'s final tensors equal the release (ledger 2026-09-14).
- **Implementation (`finetune_libero_ctrl.py`, `--head flow`).** `FlowHeads`: rectified flow over [12 x 7 action chunk,
  z split into 8 tokens of 32], conditioned on the same MAP-pooled visual and prompt states as `ContHeads` (UniVLA's
  decoder pools the same way); a 4-layer, 512-wide transformer denoiser with a time embedding; loss = action velocity
  MSE + `lam_z` x latent velocity MSE; logged diagnostics are the one-step action L1 and latent R2.  `--latent none
  --head flow` is the identical head without z.  Evaluation (`run_libero_eval_cont.py`) samples with 10 Euler steps,
  clips to [-1, 1], and fixes the noise per (seed, task, episode), so a rerun is reproducible.  Run dirs get `+fm`.
- **Jobs.**  smoke (tc_cont + none, 12 steps) 20817929 -> spatial load test 20817930;
  Ours-FM spatial 20817931 -> evals 20817934-38 (`NOTE=ours_fm_spatial_f10`);
  none-FM spatial 20817932 -> evals 20817939-43 (`NOTE=none_fm_spatial_f10`);
  Ours-FM object 20817933 -> evals 20817944-48 (`NOTE=ours_fm_object_f10`).  Evaluations are split 5 x 2 tasks and
  exclude h003/h004/h006.  Goal/Long with the flow head only after the Spatial result.
- **What the three Spatial numbers will separate.**  Ours-reg 70.4 vs Ours-FM: the interface.  none-FM vs Ours-FM:
  what the latent adds through the same head.  UniVLA 84.2 stays the external reference (their native interface).

## Registered next method: intervention first, real pairs second


## Registered next method: intervention first, real pairs second

### 2026-09-01 — temporally consistent nuisance intervention v0

- **Test.** Determine whether same-transition pseudo-pairs can teach nuisance invariance
  without first detecting equal actions across real transitions.
- **Setup — planned, not yet run.** Compare matched-budget baseline, photometric-only, and
  photometric-plus-static-style arms. For each transition, sample augmentation parameters
  once and apply them identically to both endpoints: brightness/exposure, contrast,
  saturation/hue/white balance, gamma, a smooth low-frequency illumination or shadow field,
  blur/noise/compression, and a static texture/style overlay. Exclude independent per-frame
  jitter, moving distractors, geometric warps, heavy occlusion, and literal donor-background
  replacement from v0.
- **Result.** No measured result yet; this entry registers the method and evaluation before
  training.
- **Control.** Keep seeds, data, model, steps, and losses matched. Measure pretrained-feature
  sensitivity before training so an augmentation that is already invisible to the encoder is
  not mistaken for useful supervision.
- **Conclusion — planned.** Color and illumination are safe first nuisances because they
  preserve pixel correspondence and motion by construction. A static texture/style overlay
  approximates background variation, but is not evidence that true background replacement
  has been solved.
- **Artifacts.** Design registration in this ledger; no run artifact yet.
- **Decision.** Use cross-view reconstruction so appearance remains available from the
  destination initial frame:
  `D(F0^b, z(x^a), c_task) -> Fk^b` and
  `D(F0^a, z(x^b), c_task) -> Fk^a`, while matching `z(x^a)` and `z(x^b)`.
  Advance only if action/net/profile probes remain stable, nuisance/state leakage decreases,
  clean reconstruction is not harmed, and held-out nuisance robustness improves.

### 2026-09-01 — Bridge photometric SSL screening

- **Test.** Test whether exact same-transition photometric views improve effect-latent
  invariance and action information, and whether the gain comes from cross reconstruction,
  InfoNCE, or a negative-free VICReg objective.
- **Setup.** Planned Bridge 622-episode episode-disjoint split, continuous 8x32 latent,
  one seed, 4,000 steps and batch 96. Matched arms are clean reconstruction (`baseline`),
  two-view cross reconstruction (`photo_xrec`), `photo_xrec+InfoNCE` (`photo_nce`), and
  `photo_xrec+VICReg` (`photo_vic`). Each episode receives two fixed colour/exposure/
  white-balance/gamma and low-frequency illumination fields, so no artificial temporal
  motion is introduced.
- **Result.** Feature extraction job `20284956` completed successfully. Frozen V-JEPA
  frame cosine is `0.9635/0.9636` for clean versus A/B and `0.9588` for A/B, but temporal
  delta cosine falls to `0.5392/0.5425` for clean versus A/B and `0.4853` for A/B. To obey
  the gpu-debug two-job submission cap without dropping arms, the current jobs are bundled:
  `20307539` runs `baseline` then `photo_xrec`, and `20307540` runs `photo_nce` then
  `photo_vic`, all at the registered 4,000-step budget. Job `20307539` completed; its
  control result is recorded separately below. Job `20307540` never started because the
  requested gpu-debug nodes were unavailable and was cancelled on 2026-09-02; the identical
  SSL bundle was resubmitted to the regular GPU partition as job `20314703`. That job also
  never started and was cancelled before execution so VICReg would not wait behind a now-
  redundant NCE rerun. At the time of this entry standalone `photo_vic` job `20314851`
  was pending on the regular GPU partition. **Update, 2026-09-06:** its September 2
  completion and negative screening result are recovered below; the pending status is
  historical. No completed photo-NCE result was found in this audit.
- **Control.** `photo_xrec` separates augmentation/cross-reconstruction effects from the
  SSL loss. Clean action probes, both trained photometric views, and the existing unseen
  moving-patch condition are evaluated from the same clean-trained probe.
- **Conclusion — planned.** This is a one-seed screening experiment, not a transport or
  broad embodied-data conclusion.
- **Artifacts.** [`latent_free/phase1/bridge_photo_extract.py`](latent_free/phase1/bridge_photo_extract.py),
  [`claude_survey/intervention_ssl.py`](claude_survey/intervention_ssl.py),
  [`claude_survey/run_intervention_ssl_multi.sh`](claude_survey/run_intervention_ssl_multi.sh),
  [`claude_survey/results/photo_feature_gate.json`](claude_survey/results/photo_feature_gate.json),
  [`latent_free/logs/lf_photo_20284956.out`](latent_free/logs/lf_photo_20284956.out), with
  training logs under `claude_survey/logs/cs_intssl_<job>.out` after completion.
- **Decision.** Promote an arm only if it beats `photo_xrec` rather than merely baseline,
  preserves clean action information, and improves held-out nuisance robustness. A winner
  must then be repeated across seeds and evaluated on transport. Prioritize standalone
  VICReg because moving-patch NCE already has a provisional positive result, whereas VICReg
  is the untested negative-free alternative.

### 2026-09-01 — photometric cross-reconstruction controls

- **Test.** Determine whether two-view photometric cross reconstruction alone explains any
  later SSL gain, using a matched clean-reconstruction baseline.
- **Setup.** Bridge-622 episode-disjoint split, continuous 8x32 latent, seed 0, 4,000 steps,
  batch 96; `baseline` and `photo_xrec` ran serially in job `20307539` and completed with
  exit code 0.
- **Result.** `baseline -> photo_xrec`: clean xyz R2 `0.0166 -> 0.0213`, clean cont6
  `0.0097 -> 0.0128`, and held-out moving-patch cont6 `0.0054 -> 0.0103`. Relative patch
  drop changes `44.5% -> 19.5%`, but both absolute action readouts remain near zero.
- **Control.** Same seed, model, split, steps, batch and clean-trained probe. The relative
  robustness percentage is interpreted beside absolute clean/patch R2 to avoid a small-
  denominator artifact.
- **Conclusion — negative.** Photometric cross reconstruction alone does not create a
  meaningfully action-informative latent in this screening model. Its smaller relative patch
  drop is not sufficient evidence of a useful representation.
- **Artifacts.** [`claude_survey/results/intervention_baseline_s0.csv`](claude_survey/results/intervention_baseline_s0.csv),
  [`claude_survey/results/intervention_photo_xrec_s0.csv`](claude_survey/results/intervention_photo_xrec_s0.csv),
  [`claude_survey/logs/cs_intmulti_20307539.out`](claude_survey/logs/cs_intmulti_20307539.out).
- **Decision.** Require NCE/VICReg to improve absolute clean action information and held-out
  patch R2 over `photo_xrec`; do not promote an arm based on relative drop alone.

### 2026-09-01 — photometric intervention pixel gate

- **Test.** Verify that the registered two photometric views change appearance materially
  without injecting independent frame noise or destroying the original temporal difference.
- **Setup.** 64 deterministically sampled Bridge stride-4 transitions. Each view uses one
  episode-static colour/exposure/white-balance/gamma transform and one smooth low-frequency
  illumination field.
- **Result.** Pixel MSE is `0.0188` (clean versus view A), `0.0260` (clean versus B), and
  `0.0422` (A versus B). Temporal pixel-difference cosine is `0.9796` (clean/A), `0.9839`
  (clean/B), and `0.9668` (A/B).
- **Control.** The implementation contains no per-frame parameter sampling; both endpoints
  use exactly the same episode/view parameters. Geometry and frame indices are unchanged.
- **Conclusion — confirmed.** At the pixel level, the intervention is nontrivial in
  appearance while preserving the transition difference direction. This does not establish
  V-JEPA feature sensitivity or learned-latent invariance.
- **Artifacts.** [`claude_survey/results/photo_pixel_gate.json`](claude_survey/results/photo_pixel_gate.json),
  [`latent_free/phase1/photo_pixel_gate.py`](latent_free/phase1/photo_pixel_gate.py).
- **Decision.** Continue to the frozen-feature gate and matched SSL training; do not change
  augmentation strength before seeing the V-JEPA response.

### 2026-09-01 — photometric feature gate: nulls and action-probe read (audit of the entry above)

- **Test.** The feature gate reports V-JEPA `delta cos = 0.539 / 0.543` (clean vs A / B) and
  infers that the latent will treat lighting as a transition effect. That number has no null
  and no action-level consequence attached, so this entry adds both, on the gate's own 3000
  transitions (same `RandomState(0)` draw order; the 0.5392 / 0.5425 reproduce exactly).
- **Setup.** Bridge-622, stride 4, frozen V-JEPA 4x4 features. Nulls: cross-episode random
  partner, and same-episode other-time partner. Calibrated references: the existing moving
  and static patch caches. Operational read: an episode-disjoint ridge (alpha chosen on an
  inner train-episode split) from clean `dF` to 4-step net xyz, 6000 train / 2000 held-out
  transitions, evaluated on clean / A / B / moving-patch / static-patch `dF` of the SAME
  held-out transitions.
- **Result.** Nulls: cross-episode `-0.001`, same-episode other-time `-0.019`, so chance
  is 0 and 0.54 preserves most of the motion direction. References: moving patch `0.592`,
  static patch `0.672` -- by cosine the photometric views perturb `dF` MORE than the moving
  patch does. Probe net-xyz R2: clean `0.360`, A `0.354` (**-1.6%**), B `0.352` (**-2.3%**),
  moving patch `0.306` (**-15.1%**), static patch `0.335` (-7.1%).
- **Control.** Same probe, same held-out rows, all conditions paired; alpha not tuned on the
  held-out episodes.
- **Conclusion — negative for the gate's inference.** The cosine drop is real but lies almost
  entirely in action-irrelevant directions of `dF`: episode-static photometric change costs a
  frozen-feature action readout ~2%, an order of magnitude less than the moving patch. The gate
  does not establish that SSL intervention is necessary for lighting; cosine is the wrong
  instrument for that claim. (Frozen `dF` is not the learned latent, so a learned 8x32 latent
  could still be lighting-sensitive -- but the necessity argument was made from frozen cosine,
  and frozen cosine does not support it.)
- **Artifacts.** [`claude_survey/results/photo_feature_gate_nulls.json`](claude_survey/results/photo_feature_gate_nulls.json),
  [`claude_survey/photo_gate_nulls.py`](claude_survey/photo_gate_nulls.py).
- **Decision.** Read the four-arm screen primarily on the **moving-patch** (untrained
  nuisance) condition, where ~15% is there to recover, not on the photometric views, where
  ~2% is. Report the photometric-view drop with this frozen-feature 2% as its reference.

### 2026-09-01 — held-out action decodability for the transport, equipartition and language arms (`eval_all`)

- **Test.** One eval job, one probe, one split (600 held-out episodes, 14,879 transitions,
  UniVLA stage-2 arms, T=2) so the three groups below are read on identical instruments.
  `z_q` = post-VQ deployed code, `emb` = pre-VQ; `clean_xyz` is the linear action R2 that
  LAPA / Genie / UniVLA-style probes report; `z_beyond_df` is the increment over the frozen
  frame difference.
- **Result — transport arms (the control the 08-30/09-01 entry lacked).** `z_q clean_xyz`:
  base `0.3215`, control `0.3195`, **oracle `0.4072` (+0.086)**, **oracle-positive `0.4012`
  (+0.080)**, pseudo `0.3134`, DINO pairs **`0.2807` (-0.041)**. `z_beyond_df`: base `0.079`,
  oracle `0.139`, oracle-positive `0.137`. Moving-patch drop: base 11.7%, oracle 7.7%.
  Decodability and transport co-move with pairing quality in both directions: correct pairs
  raise both, 56%-agreement pairs lower both, random pairs move neither.
- **Result — equipartition on the deployed codebook (P6's pre-registered test, z_q half).**
  `z_q clean_xyz`: no-balance 256 `0.2948` / `0.2914`; **lambda=1 x16 `0.1827`, lambda=1 x256
  `0.2438`, lambda=5 x256 `0.1962`**. Pre-VQ `emb` also falls (`0.51 -> 0.46--0.49`). Forcing
  uniform code usage costs action information at every lambda and width tried. Dataset-level
  usage counts are in the separate `latacc_usage` entry when it lands.
- **Result — language arms.** `z_q`: none `0.2659`, enc `0.2957` (+0.030), both `0.2895`
  (+0.024); `emb` flat (`0.511 / 0.509 / 0.501`). One seed each.
- **Control / caveat on the floor.** `s2_lang2_none` differs from `s2_base` only by an
  uncalled `lang_proj` shifting the init RNG, yet `z_q` reads `0.2659` vs `0.3215`: a gap of
  `0.056` between two configurations with no substantive difference. STATUS.md C11's R2
  floor of `0.019` rests on two seeds; until a third `s2_base` seed is run, treat **~0.05** as
  the working single-seed floor. Under that floor: the oracle transport gains (+0.08/+0.09)
  and the x16 / lambda=5 equipartition losses (-0.14 / -0.10) stand; the lambda=1 x256 loss
  (-0.05) and both language gains (+0.03 / +0.02) do not.
- **Conclusion.** Transport: **confirmed for the objective, strengthened** -- the oracle gain is
  not a transport-metric artefact, the code's action content and its transportability rise
  together, and wrong pairs lower both. Equipartition: **negative** -- codebook collapse is not
  a capacity ceiling to be relieved; spreading assignments uniformly splits action-homogeneous
  regions along non-action directions. Language: **provisional / below floor**.
- **Artifacts.** [`claude_survey/results/eval_all.csv`](claude_survey/results/eval_all.csv),
  [`claude_survey/logs/cs_eval_all_20308592.out`](claude_survey/logs/cs_eval_all_20308592.out).
- **Decision.** Retire "codebook collapse caps capacity" as a problem statement (STATUS.md P6
  to be rewritten once `latacc_usage2` returns the dataset-level counts). Run a third `s2_base`
  seed before any further single-seed R2 comparison is reported. The transport line keeps its
  priority: the pairing-precision threshold sweep (oracle pairs corrupted at p) is the next run.

### 2026-09-02 — dataset-level codebook usage; equipartition on the deployed codebook (STATUS P6 → C19/R12)

- **Test.** Replace STATUS.md P6's per-batch code-usage counts with dataset-level counts
  (9,211 held-out transitions), and complete the pre-registered equipartition test on both
  readouts (usage, post-VQ action R2).
- **Result.** Usage 4x16 `16/16`, 4x256 `26/256` (faithful init `21`), 1x65536 `18/65536`.
  With equipartition on `vq_action`: 4x16 lambda=1 `11/16`, 4x256 lambda=1 `18/256`, lambda=5
  `12/256`; `z_q clean_xyz` `0.1827 / 0.2438 / 0.1962` vs `0.3215 / 0.2948`.
- **Control.** Same probe/split as `latacc.csv`; the counting instrument is the one that read
  `s2_base` TC at 16/16 while its training log said 12--13/16.
- **Conclusion — negative (P6 retracted).** Extra codebook entries stay unused (16 -> 65,536
  entries adds two used codes), and forcing uniform usage lowers both usage and action content.
  Collapse is a property of the data's action distribution, not a capacity ceiling.
- **Artifacts.** [`claude_survey/results/latacc_usage.csv`](claude_survey/results/latacc_usage.csv),
  [`claude_survey/results/latacc_usage2.csv`](claude_survey/results/latacc_usage2.csv),
  [`claude_survey/STATUS.md`](claude_survey/STATUS.md) C19 / R12.
- **Decision.** Drop "relieve codebook collapse" from the method. The discrete version of the
  effect latent should quantise a factorised effect map (per-cell small codebooks), not widen a
  single collapsed codebook.

### 2026-09-02 — three-seed R2 floor for the current stage-2 base configuration

- **Test.** `s2_lang2_none` seeds 0/1/2 (identical config; the layer that differs from
  `s2_base` is never called), one probe, one 600-episode split.
- **Result.** `z_q clean_xyz` `0.2659 / 0.3156 / 0.3192` (sd 0.030); `emb` `0.5107 / 0.5121 /
  0.5175` (sd 0.004). Pooled with the two `s2_base` runs: `z_q` `0.305 +- 0.022` over five runs.
- **Conclusion — confirmed.** The quantiser adds ~8x the seed variance of the continuous
  latent. Single-seed `z_q` differences below ~0.07 are not readable; `emb` differences are
  readable from one seed. Under this rule the language-conditioning gains (+0.03 / +0.02), the
  lambda=1 x256 equipartition loss (-0.05) and the DINO-pair decodability loss (-0.04) are
  withdrawn as directions; the oracle-transport gains and the x16 / lambda=5 equipartition
  losses stand. STATUS.md C11 rewritten with the table.
- **Artifacts.** [`claude_survey/results/eval_floor.csv`](claude_survey/results/eval_floor.csv).
- **Decision.** Every future `z_q` comparison ships with three seeds or is labelled
  single-seed and not read for direction; report `emb` next to `z_q` in every table.

### 2026-09-06 — recovered photometric VICReg completion

- **Test.** Audit whether the registered photo-VICReg arm completed and improved action
  content beyond the matched photo cross-reconstruction control.
- **Setup.** Bridge-622, episode-disjoint train/validation windows; continuous 8×32 query
  latent over absolute V-JEPA frame tokens, seed 0, 4,000 steps, batch 96, lambda=0.05
  multiplying VICReg weights 25/25/1. Job 20314851 reaches step 3999, prints the CSV row,
  and ends with `INTERVENTION SSL DONE` on September 2.
- **Result.** Baseline / photo_xrec / photo_vic clean xyz R² =
  `0.0166 / 0.0213 / 0.0017`; clean cont6 = `0.0097 / 0.0128 / -0.0012`;
  moving-patch cont6 = `0.0054 / 0.0103 / -0.0039`. VICReg photo-pair cosine is `0.9638`.
  Final logged weighted VICReg term is `0.3365`, reconstruction `0.2883` (training batch).
- **Control.** Matched seed, model, data, steps and batch. Probe alpha is selected on
  evaluation labels separately per condition, as corrected below; one seed, no held-out
  rank/state/nonlinear-action diagnostic. This is not the proposed structured hybrid.
- **Conclusion — negative.** This photo-VICReg implementation failed the useful linear
  action-content gate. It does not establish collapse of all information or failure of
  VICReg in general; loss-scale imbalance is a hypothesis, not a diagnosed cause.
- **Artifacts.** [`claude_survey/results/intervention_photo_vic_s0.csv`](claude_survey/results/intervention_photo_vic_s0.csv),
  [`claude_survey/logs/cs_intssl_20314851.out`](claude_survey/logs/cs_intssl_20314851.out),
  [`claude_survey/intervention_ssl.py`](claude_survey/intervention_ssl.py), matched
  `claude_survey/results/intervention_{baseline,photo_xrec}_s0.csv`.
- **Decision.** Do not carry this arm forward as positive invariance evidence. Establish a
  useful structured baseline and compare SSL objectives under the same nuisance and budget.

### 2026-09-06 — correction to older clean-probe robustness claims

- **Test.** Inspect whether IRLA and the photometric SSL screen actually reuse one fixed
  clean-trained probe across clean and nuisance views.
- **Setup.** Read-only code audit of `irla.py`, `intervention_ssl.py`, and
  `latent_free/probes_torch.py`; applies to the 8,000-step IRLA and 4,000-step photo screens.
- **Result.** Each condition separately calls ridge fitting/selection;
  `_ridge_std` chooses alpha by that condition's evaluation cont6 R². Recorded IRLA
  lambda=1 clean/patch cont6 `0.1007/0.0943` and photo_xrec `0.0128/0.0103` therefore
  include evaluation-label model selection. Clean fitting rows alone do not make the
  selected probe identical across conditions.
- **Control.** `codex_survey/experiment.py` tunes on inner training episodes and reuses its
  net probe for clean/patch; `photo_gate_nulls.py` also has a separate nested fixed-probe gate.
- **Conclusion — retracted.** The earlier assertion of a single fixed probe in the IRLA/photo
  screen robustness comparisons is withdrawn. Numerical outputs remain provisional
  screening measurements; near-zero absolute VICReg action readout remains negative.
- **Artifacts.** [`claude_survey/irla.py`](claude_survey/irla.py),
  [`claude_survey/intervention_ssl.py`](claude_survey/intervention_ssl.py),
  [`latent_free/probes_torch.py`](latent_free/probes_torch.py),
  [`claude_survey/results/photo_feature_gate_nulls.json`](claude_survey/results/photo_feature_gate_nulls.json).
- **Decision.** Tune only on inner clean training episodes and freeze alpha and weights for
  every nuisance condition. Do not infer InfoNCE > VICReg from different nuisances/budgets.

### 2026-09-06 — recovered cycle-only and direct-effect transport training

- **Test.** Check pair-free transport alternatives alongside the positive oracle and
  continued-reconstruction control, including arms absent from the compact summary table.
- **Setup.** Bridge-big, same stage-2 base checkpoint, one seed per arm, 3,000 fine-tuning
  steps, batch 12; 1,062 held-out endpoints, matched scene subset n=689. Cycle uses a frozen
  base inverse/codebook, two donors per target, CE weight 0.05 and ranking weight 1.
- **Result.** `xs_actmatch`: base `0.5070`, continued reconstruction `0.5114`,
  oracle-positive `0.6558`, oracle positive+rank `0.6628`, rank-only `0.5900`,
  cycle `0.4955`, direct spatial-effect loss `0.3636`. Donor MSE for
  base/control/oracle-positive/cycle/effect = `0.18613/0.18125/0.16883/0.18775/0.23033`.
  Frozen base cycle recovery was already `0.6973` across random scenes versus self `0.7799`
  and chance `0.1584`. Full training/evaluation completion markers exist for both new rows.
- **Control.** Oracle-positive establishes that useful donor supervision can improve this
  architecture. `s2_trctl` is continued self reconstruction, not random-swap training.
  Transport seed uncertainty is unmeasured. No new flow prior was used by the cycle arm.
- **Conclusion — negative.** Neither implemented pair-free arm improves transport at this
  budget. Cycle recovery does not suffice as evidence of shared physical action semantics.
- **Artifacts.** [`claude_survey/results/transport_s2_trcycle.csv`](claude_survey/results/transport_s2_trcycle.csv),
  [`claude_survey/results/transport_s2_treffect.csv`](claude_survey/results/transport_s2_treffect.csv),
  [`claude_survey/logs/tr_cycle_20245783.out`](claude_survey/logs/tr_cycle_20245783.out),
  [`claude_survey/logs/cs_treffect_20238892.out`](claude_survey/logs/cs_treffect_20238892.out),
  [`claude_survey/results/cycle2.csv`](claude_survey/results/cycle2.csv),
  [`claude_survey/transport_train.py`](claude_survey/transport_train.py).
- **Decision.** Evaluate prior-only, cycle-only and prior+cycle separately with an independent
  semantic transport instrument. Do not revive cycle-alone as a solved pairing substitute.

### 2026-09-06 — transport oracle scope and seed interpretation correction

- **Test.** Audit what existing action-matched transport scores actually match and measure.
- **Setup.** Read existing `transport.py`, `transport_train.py`, transport CSVs and the
  September 2 seed-floor results; no new checkpoint evaluation.
- **Result.** Training oracle uses standardized summed seven-dimensional actions; evaluation
  builds donor distance from `tab['act'][:, :3]`, summed xyz only. Its bin agreement/cosine
  `0.9030/0.9796` validate translation similarity. Matched-scene values `0.4719/0.5694`
  yield the historical `18%/82%` residual-to-1 calculation, but no physical ceiling was measured.
- **Control.** Rotation, gripper and within-window profile are not matched in this evaluator.
  Post-VQ seed-floor evidence withdraws the directional reading of DINO-pair action-R²
  `-0.041`; its transport `0.4678` is a separate one-seed negative observation with no
  calibrated seed floor. The same control prevents treating tiny slot/conditioning gains
  as established effects.
- **Conclusion — retracted.** Withdraw descriptions of the evaluation as fully 7-DoF matched
  and of the `18%/82%` split as a causal decomposition. Existing scores remain an operational
  xyz-matched feature-reconstruction transport measurement, not rollout success.
- **Artifacts.** [`claude_survey/transport.py`](claude_survey/transport.py),
  [`claude_survey/transport_train.py`](claude_survey/transport_train.py),
  [`claude_survey/results/transport2.csv`](claude_survey/results/transport2.csv),
  [`claude_survey/results/eval_floor.csv`](claude_survey/results/eval_floor.csv).
- **Decision.** Add rotation/gripper and ordered-profile matching, report raw MSE and pairing
  quality, and use repeated seeds plus episode-level uncertainty for new transport comparisons.

### 2026-09-06 — recovered frozen-Wan flow-decoder screens

- **Test.** Check whether an existing pretrained generative decoder freed z to encode action.
- **Setup.** Bridge-622, 16,945 training windows, episode-disjoint evaluation; frozen Wan 5B,
  trainable IDM/conditioning adapter (53.3M parameters), 4,000 steps, batch default 16;
  original/all-noise, high-noise and high-noise+NCE arms. One initialization per arm;
  full invocation/initialization metadata and NCE coefficient are not saved in the CSVs.
- **Result.** Original/high-noise/high-noise+NCE xyz R² `0.0153/0.0118/0.0220`, cont6
  `0.0050/0.0049/0.0103`, state-PCA prediction R² `0.4830/0.4727/0.5018`. All three logs
  reach step 3999, evaluation, and `DITLAM DONE`.
- **Control.** The NCE arm logs real/shuffled/zero flow loss `0.2223/0.2292/0.9081`.
  This comparison uses different sampled noise/timesteps: resetting the global seed does
  not reset its explicit `gen`. Thus the small real/shuffled gap is not a paired-noise
  estimate; zero context also changes the conditioning distribution. Action probes use
  evaluation-selected alpha, and architecture/budget differ from newer stage-2 experiments.
- **Conclusion — negative.** These completed adapter screens do not deliver a strongly
  linearly action-decodable latent. They do not test or refute the new flow-decoder +
  separate frozen dynamics-prior swap objective, nor establish sample-specific z usage.
- **Artifacts.** `claude_survey/results/ditlam_{bridge,hiN,hiN_nce}.csv`,
  `claude_survey/logs/cs_ditlam_{19708756,19715192,19715193}.out`,
  [`claude_survey/dit_lam.py`](claude_survey/dit_lam.py).
- **Decision.** Make prior calibration and donor-specific control separate gates; hold noise,
  timestep, target state and residual fixed for real/shuffled z comparisons.

### 2026-09-06 — raw pooling matrix confirmation and robustness trade-off

- **Test.** Reaggregate raw pooling results and verify completion rather than reusing an
  earlier summary; read preservation and robustness jointly.
- **Setup.** Bridge-big 4,000/1,000 train/validation episodes; 12k/5k probe windows;
  32 continuous latent dimensions, 456,720 parameters, 6,000 steps, batch 128, seeds 0/1/2
  for five poolers × two targets. All 30 JSONs and result-writing logs are present.
- **Result.** Endpoint global/spatial/temporal/DCT net R²
  `0.1999/0.2565/0.2390/0.2173`, profile `0.0798/0.0782/0.1090/0.1023`.
  Sequence net `0.2252/0.2717/0.2203/0.2164`, profile `0.0894/0.0817/0.1004/0.0993`.
  Sequence spatial/global patch net `0.2023/0.1879`, relative drops `25.5%/16.5%`.
- **Control.** Equal latent width/parameter count, paired seeds, shared rows and inner
  episode tuning. Endpoint/sequence losses have different target scales; equal beta does
  not ensure equal effective reconstruction/KL pressure. No completed spatial×DC+AC1 arm.
- **Conclusion — confirmed.** Spatial/temporal evidence remains complementary under this
  instrument; improved preservation does not imply improved relative nuisance robustness.
- **Artifacts.** [`codex_survey/results/`](codex_survey/results/),
  [`codex_survey/experiment.py`](codex_survey/experiment.py),
  [`codex_survey/logs/`](codex_survey/logs/), recomputed table in
  [`latent_free/EXPERIMENT_REVIEW_2026-09-06.md`](latent_free/EXPERIMENT_REVIEW_2026-09-06.md).
- **Decision.** Test the hybrid under matched total width, report absolute clean/patch
  performance and net/profile separately, and do not call the compressed basis lossless.

### 2026-09-06 — recovered offline policy-side limitation

- **Test.** Determine whether future-derived action decodability survives prediction of z
  from information available to a deployed policy.
- **Setup.** Existing Bridge-big offline ridge experiment, one episode split/label selection,
  9,863 policy-training rows and 5,016 test rows, label budgets 100/300/1000/3000/all;
  inner episode tuning; current-frame-only, no language/history, no rollout.
- **Result.** `s2_base`, all labels: true-z oracle `0.3181`, predicted-z route `0.0759`,
  direct current-frame route `0.1196`; current-frame→z R² `0.0360`. At 1,000 labels,
  predicted-z/direct = `0.0623/0.0932`.
- **Control.** Same test rows, direct prediction and PCA/free-difference routes; both stages
  of the z route are linear. This limitation is not a test of a modern language/history-
  conditioned VLA or the newly registered method.
- **Conclusion — provisional.** Higher future-derived latent readout has not established
  downstream label efficiency; the tested base z route trails direct prediction at these
  budgets in this limited offline instrument.
- **Artifacts.** [`claude_survey/results/policy_eval.csv`](claude_survey/results/policy_eval.csv),
  [`claude_survey/policy_eval.py`](claude_survey/policy_eval.py),
  [`claude_survey/logs/cs_policy_19989512.out`](claude_survey/logs/cs_policy_19989512.out)
  (ends `POLICY policy_eval DONE`, August 17).
- **Decision.** Include matched-label policy learning curves with appropriate task/history
  conditioning and later rollout alongside representation and transport diagnostics.

### 2026-09-06 — IRLA bundled-job failure attribution corrected

- **Test.** Reconcile the lambda=3 result with the previously reported job timeout.
- **Setup.** Bridge-622, job 19690719 runs lambda=3 then lambda=10, intended 10,000 steps
  each, one initialization each; 16,945 training windows, batch default 128.
- **Result.** Lambda=3 reaches step 9999 and prints xyz/cont6/patch-cont6
  `0.0379/0.0169/0.0147`, exactly matching `irla_bridge_inv3.csv`. The log then begins a
  separate lambda=10 arm and times out after its step-9000 report.
- **Control.** Per-arm log boundaries and CSV equality, rather than the bundled job's final
  status. The successful lambda=1 comparison elsewhere used 8,000 steps, so the weights
  cannot be ranked as a controlled sweep. Old probes use evaluation-selected alpha.
- **Conclusion — provisional.** Lambda=3 has a recoverable completed screening result;
  the earlier assertion that this arm never evaluated is retracted above.
- **Artifacts.** [`claude_survey/logs/cs_irla_19690719.out`](claude_survey/logs/cs_irla_19690719.out),
  [`claude_survey/results/irla_bridge_inv3.csv`](claude_survey/results/irla_bridge_inv3.csv).
- **Decision.** Preserve the recovered result; use unique arm identifiers and saved full
  configuration for any future weight comparison.

### 2026-09-06 — actual incomplete IRLA lambda=10 arm

- **Test.** Check the stronger invariance arm that follows lambda=3 in job 19690719.
- **Setup.** Bridge-622, intended 10,000 steps, lambda=10, one initialization.
- **Result.** Last logged step 9000; scheduler cancellation at time limit; no final
  action evaluation for this arm in that log.
- **Control.** The preceding lambda=3 evaluation belongs to a different arm.
- **Conclusion — failed.** The lambda=10 arm in this bundled job did not complete.
- **Artifacts.** [`claude_survey/logs/cs_irla_19690719.out`](claude_survey/logs/cs_irla_19690719.out).
- **Decision.** Do not use this timeout as a method result; no resubmission made in this audit.

### 2026-09-06 — register preservation → invariance/structure → transport candidate

- **Test.** User-proposed quadrants×DC+AC1 + nuisance invariance + algebra + flow-based
  cross-state swaps regularized by frozen dynamics prior and code cycle.
- **Setup.** Planned; no combined training run. Suggested mechanism gates use Bridge-big
  matched-width three-seed pooling controls, then a controlled counterfactual test environment
  with ground truth reserved for evaluation. Final training budget remains to be specified.
- **Result.** No empirical result for the combined method. Mathematical audit: prior+cycle
  can admit opposite action-code conventions in different local state regions even with
  local composition/reversal. For raw temporal DCT of deltas, reversal maps coefficient k
  by `(-1)^(k+1)`: DC negates, AC1 does not. Concatenated-window AC1 is not simple addition.
- **Control.** Separate cycle-only/prior-only/prior+cycle and with/without-algebra arms;
  independently measure feasibility, donor semantics, code recovery and algebra error.
  A demonstration prior includes behavior-policy support, not only physical feasibility.
  Explicitly distinguish stochastic noise from a future-derived residual channel.
- **Conclusion — planned.** This is a testable research hypothesis, not established semantic
  transport or identifiability. Structured pooling preserves selected evidence; augmentation
  targets known nuisances; algebra imposes local geometry; swap/prior/cycle require semantic
  validation beyond their own training losses.
- **Artifacts.** [`latent_free/EXPERIMENT_REVIEW_2026-09-06.md`](latent_free/EXPERIMENT_REVIEW_2026-09-06.md),
  [ALAM primary source](https://arxiv.org/abs/2605.10819),
  [cross-exogenous reconstruction assumptions](https://arxiv.org/html/2605.20223v1#S4.SS4).
- **Decision.** Define duration/basis-aware algebra before implementation; first preserve
  net and profile together, then test matched nuisance objectives and independent transport.
  Reject semantic-transport claims if only prior score or cycle recovery improves. Disclose
  any action-label use in pair-threshold or hyperparameter selection during development.

### Phase 1 — paired by construction

Given one real transition `x=(o_t,o_{t+k})`, generate two nuisance renderings
`x^a=G(x,n_a)` and `x^b=G(x,n_b)` while preserving robot/object motion. Enforce the same
effect latent and cross-reconstruct from the other nuisance view. Begin with mask-free
temporally consistent background perturbations; use SAM3 only as an offline teacher/keyframe
segmenter with cached, flow-propagated masks rather than running it every training step.

The intervention must preserve the robot, gripper/hand, contacted object, and task-relevant
geometry. "Background invariant" is conditional: navigation route geometry and manipulation
targets are not nuisance just because they are not part of the robot.

### Phase 2 — real cross-state expansion with spatial DC+AC1

After Phase 1 is stable, use a spatial `DC+AC1` descriptor to mine real pairs:

1. cross-scene mutual nearest neighbours;
2. a distance/mutuality margin;
3. agreement among DC+AC1, DINO, V-JEPA, and the EMA effect encoder;
4. a labelled-oracle precision--coverage curve on the robot subset;
5. only high-confidence pairs above a pre-registered precision threshold;
6. confidence-weighted loss rather than uniform hard swaps;
7. periodic pair refresh using an EMA teacher.

**Kill criterion.** If spatial DC+AC1 cannot reach high precision at useful coverage, do
not use real-pair swap. Retain paired-by-construction intervention transport only.

## Registered information factorization

Use the causal factorization

`(F0, c_task, z_effect, u_exo)`:

- `F0`: current state, sent directly to the decoder;
- `c_task`: explicit language/task condition;
- `z_effect`: controllable effect predicted by the policy;
- `u_exo`: optional continuous exogenous residual.

Version 1 trains only `z_effect` with action-preserving augmentations. Add `u_exo` only if
real dynamic nuisance creates a demonstrated reconstruction failure. If added, cross-view
swap must keep `z_effect` invariant and restrict `u_exo` to nuisance-specific residuals;
the policy never predicts `u_exo`. Do not begin with two symmetric high-capacity VQ branches,
because either can absorb all future information and empty the effect channel.

## Dataset-scale constraint from original papers

UniVLA's full mixture spans 26 single-arm Open-X manipulation sources, three GNM navigation
sources, and Ego4D human egocentric video; actions/proprioception are excluded during its
pretraining. Olaf-World trains its LAM/world model only on MiraData's `3D Rendering` and
`City Walking` categories and uses MIND for controlled adaptation/evaluation. Consequently,
Bridge is a mechanism test, not the target distribution. Final validation must separately
cover fixed-camera manipulation, wrist/ego manipulation, navigation, human video, and passive
open-world video, including leave-one-dataset/embodiment-out reporting.

### 2026-09-06 — adopted pair-free route and staged optimization recommendations

- **Test.** User confirms intent to use structured AC/DC pooling, cross-state transport,
  and SSL, with dataset acquisition being handled separately by Claude. Assess improvements
  within that route without reinstating an equal-action real-pair prerequisite.
- **Setup.** Planned design advice only; no jobs submitted, downloads modified, or messages
  sent to another agent. Interpret AC/DC as spatial quadrants × (DC+AC1) per the existing
  proposal. Exact dataset mixture, training budget and loss coefficients are not yet set.
  Initial comparisons should preserve episode separation, total latent width and seed budget.
- **Result.** No new empirical result. Relevant measured positives remain spatial/global
  sequence net R² `0.2717/0.2252` (three seeds); frozen aligned AC1 versus equal-width
  shuffled AC1 profile gain `+0.0498` (three resamples of one episode split);
  oracle-positive transport `0.6558` versus continued reconstruction `0.5114` (one seed);
  continuous V-JEPA clip-anchor xyz R² `0.3463→0.4698` (one run per arm, 10,000 steps);
  moving-patch InfoNCE xyz R² `0.0166→0.1739` (one run per arm, 8,000 steps).
  The latter two are provisional and older action probes select alpha on evaluation labels.
- **Control.** These are different instruments/budgets and cannot be ranked as one leaderboard.
  Oracle training uses recorded actions and is a positive control, not an action-free method.
  None establishes a VLA rollout gain for the new combined method. Completed photo-VICReg
  (`xyz=0.0017`), cycle-only transport (`0.4955`), and frozen-Wan adapter screens remain
  negative for their respective implementations, not counterexamples to the new combination.
- **Conclusion — planned.** Proceed with the user's pair-free hypothesis. The old hard-pair
  loss reconstructs destination factual future from donor z and needs equal effects; the new
  swap loss has no such target and avoids that particular wrong-label failure. It still needs
  compatible destination support and an independently verified relation between donor semantics
  and generated effects. Lack of an identifiability guarantee does not imply infeasibility.
- **Artifacts.** [`latent_free/EXPERIMENT_REVIEW_2026-09-06.md`](latent_free/EXPERIMENT_REVIEW_2026-09-06.md),
  [`claude_survey/results/rala2_eval_v.csv`](claude_survey/results/rala2_eval_v.csv),
  [`claude_survey/logs/cs_s2_vae_19999916.out`](claude_survey/logs/cs_s2_vae_19999916.out),
  [`claude_survey/logs/cs_s2_vae_olaf_19999917.out`](claude_survey/logs/cs_s2_vae_olaf_19999917.out),
  [`claude_survey/results/transport_operational_summary.csv`](claude_survey/results/transport_operational_summary.csv),
  [`codex_survey/results/quick_dct_effects.csv`](codex_survey/results/quick_dct_effects.csv),
  [`claude_survey/logs/cs_irla_19694534.out`](claude_survey/logs/cs_irla_19694534.out).
- **Decision.** Recommended refinements, not yet measured or implemented:
  1. Preserve spatial cells before temporal compression; keep DC/AC1 as separate tokens
     through the bottleneck and normalize using training-population statistics without
     discarding each sample's motion magnitude. Condition on elapsed time or resample to
     a common physical horizon; compare hybrid at fixed total width before widening it.
  2. Keep a factual conditional-flow loss on real transitions throughout training. Warm up
     the structured representation with reconstruction and a calibrated nuisance SSL loss
     before freezing/snapshotting an inverse for swap. Ramp swap after the latent carries
     action evidence; SSL remains active rather than being reserved solely for the end.
  3. If using VICReg, compare a small projection-head regularizer with the prior direct-latent
     implementation, monitor variance/rank on the actual policy latent, and keep coefficients
     calibrated to gradient scales. Anti-collapse does not select the action factor. Treat the
     measured continuous clip-effect anchor as an optional same-transition warm-up ablation,
     never as ground truth for arbitrary cross-state effect equality.
  4. Train the dynamics prior on real transitions and freeze it for each swap phase. Treat
     its score as a soft regularizer, not a guarantee of physical validity or a scalar flow
     likelihood. Demonstration density includes behavior-policy preferences; preserve rare
     valid effects and test coverage. A learned flow velocity requires an explicitly defined
     conversion/distillation objective before it can serve as a prior gradient.
  5. Initially sample destinations within compatible embodiment, camera convention, horizon
     and task/effect support while varying endogenous state. This restricts applicability;
     it does not assert equal recorded actions. Expand across embodiments after independent
     transport checks. Never apply arbitrary-donor factual reconstruction to destination
     future, or unrestricted latent attraction between different real transitions.
  6. Prefer independent generation noise initially. An unrestricted residual encoded from
     destination future can carry the answer even if detached; detaching alone does not
     remove information leakage. Hold state/task/noise fixed and change donor z to measure
     its influence; measure physical meaning separately from mere output differences.
  7. Use none/prior-only/cycle-only/prior+cycle swap ablations from a shared warm-up; add
     basis-aware algebra separately. Independent held-out evaluation must include net,
     temporal profile, rotation/gripper, nuisance robustness and semantic transport, followed
     by matched-label policy learning. No numeric promotion threshold for transport is
     invented before measuring its seed/episode uncertainty.

### 2026-09-06 — UniVLA-based full-mixture training target and integration contract

- **Test.** User specifies UniVLA as the base implementation and its full pretraining
  collection as the intended training distribution for the proposed method; Claude is
  handling dataset acquisition. Register the comparison and inspect integration gaps.
- **Setup.** Planned; no training or download mutation in this turn. Target the pinned local
  `omni_magic_soup_plus_plus` definition (26 manipulation + 3 navigation sources + 4 curated
  Ego4D split entries), including its sampling policy and actual episode coverage. Budget,
  seed count, initialization and final policy interface remain to be configured.
- **Result.** No empirical model result. Local code inspection identifies 4 omitted OXE
  sources (`kuka`, `language_table`, `fmb`, `furniture_bench`) plus Ego4D in `oxe_fetch.py`;
  conversion defaults to a 5,000-episode cap and Bridge uses an existing replacement cache.
  These agree with the older reduced-mixture registration, not a verified full-mixture
  acquisition. The LAM YAMLs select `omni_magic_soup_plus_plus`, but the provided VLA shell
  script selects a config whose active `data_mix` is `omni_magic_soup_plus` (no Ego4D).
  VLA pretraining constructs 16 `<ACT_i>` tokens and consumes discrete LAM code indices.
- **Control.** This is a snapshot of local scripts/configs, not a fresh download-status
  audit; acquisition may be changing concurrently. Dataset names alone do not verify
  complete curated splits, weights, conversion coverage or identical effective exposure.
  A continuous effect latent cannot silently replace discrete token labels in the existing
  next-token objective. Changes to the policy head need a matched architecture control.
- **Conclusion — planned.** Adopt full-mixture UniVLA-based training as the user's target,
  while keeping the older reduced-mixture acquisition record historical. Full coverage and
  interface integration are unverified; neither is reported as completed.
- **Artifacts.** [`latent_free/phase1/oxe/oxe_fetch.py`](latent_free/phase1/oxe/oxe_fetch.py),
  [`latent_free/phase1/oxe/oxe_convert.py`](latent_free/phase1/oxe/oxe_convert.py),
  [`latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/mixtures.py`](latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/mixtures.py),
  [`latent_free/third_party/UniVLA/prismatic/conf/vla.py`](latent_free/third_party/UniVLA/prismatic/conf/vla.py),
  [`latent_free/third_party/UniVLA/vla-scripts/train.py`](latent_free/third_party/UniVLA/vla-scripts/train.py),
  [`latent_free/third_party/UniVLA/README.md`](latent_free/third_party/UniVLA/README.md).
- **Decision.** Reuse UniVLA's vision/VLM backbone and data conventions where compatible,
  replace/extend latent representation and training objectives as explicitly registered,
  and recommend a continuous prediction head for the continuous version. Preserve original
  token-based UniVLA as a reference and add a continuous reconstruction-only architecture
  control so the method's effect is separable from the head/bottleneck change. Both method
  and controls use the same actual data manifest, episode splits and sampling exposure;
  report total compute including prior training. Verify the final acquisition manifest and
  set the same explicit mixture in LAM and policy stages before calling a run full-mixture.
  No changes made to Claude's downloader or conversion work, and no messages sent to Claude.

### 2026-09-06 — UniVLA dataset inventory, conditional ETA and readiness audit

- **Test.** Check acquisition against the user's full UniVLA-mixture target and distinguish
  downloaded, converted, queued and unverified data; estimate transfer time from actual logs.
- **Setup.** Read-only file/metadata/code audit; scheduler, login01 process and quota snapshot
  at 05:44 UTC / 01:44 EDT. Target 33 entries (26 manipulation + 3 navigation + 4 Ego4D
  entries); 28 sources have cached public-bucket size metadata. No jobs or downloads changed,
  no model steps run, no training seeds. Current OXE converter cap=5000, seed=0, random 10%
  episode holdout; existing Bridge cache instead has 1000/5000 validation episodes.
- **Result.** 12/28 public sources have all 677 expected nonempty TFRecord shards, totaling
  `64.317927 GB`, zero `.part` files, and logs reporting zero failed fetches. Other 16 source
  directories absent; four omitted by the current 24-source downloader. Its large-source log
  is empty; no downloader seen on inspected login host. Job `20432961` is PENDING (Priority).
  UCSD cache verifies 150 episodes / 3970 frames / 15 validation episodes / 0 skipped,
  matching image/action/index lengths; logged conversion=32 seconds. Bridge raw verifies
  97/1024 train and 0/128 validation shards (11.103 GB); Bridge_big is a 5000-episode cache.
  Expected 28-source raw size=3480.281 GB, excluding Bridge/Ego4D. Completed fetch logs cover
  64.3 GB in 295 seconds (~218 MB/s). Conditional at 150–250 MB/s: remaining selected24
  `742.003 GB` takes `0.82–1.37 h`; remaining full28 `3415.964 GB` takes `3.80–6.33 h`.
  Scratch quota reports 557.7 GB/100 TB and 443.9k/1M files.
- **Control.** File existence/count/nonzero sizes do not establish checksums or complete
  TFRecord/image integrity. Sizes are cached metadata estimates. A process on another host
  is not excluded. Network extrapolations are conditional, not wall-clock completion promises;
  they exclude Bridge/Ego4D, queueing, stalls, decoding and feature extraction. Ego4D access,
  curated clip manifest/size, and exact Bridge_oxe replacement equivalence remain unverified.
- **Conclusion — provisional.** Acquisition and training readiness are incomplete; the
  current script/cache combination is reduced-mixture pilot infrastructure. Full end-to-end
  readiness has no defensible ETA yet, although public-source transfer alone is hours-scale
  if the historical rate holds after a correctly scoped active download starts.
- **Artifacts.** [`latent_free/UNIVLA_DATASET_AUDIT_2026-09-06.md`](latent_free/UNIVLA_DATASET_AUDIT_2026-09-06.md),
  [`claude_survey/results/oxe_mix_sizes.json`](claude_survey/results/oxe_mix_sizes.json),
  [`latent_free/logs/oxe_fetch_small.out`](latent_free/logs/oxe_fetch_small.out),
  [`latent_free/logs/oxe_fetch_big.out`](latent_free/logs/oxe_fetch_big.out),
  [`latent_free/phase1/oxe/oxe_convert.py`](latent_free/phase1/oxe/oxe_convert.py),
  [`UCSD metadata`](latent_free/cache/oxe/ucsd_kitchen_dataset_converted_externally_to_rlds/meta.json).
- **Decision.** Confirm the active acquisition host/log and full manifest, include omitted
  sources, resolve Bridge and curated Ego4D, and benchmark representative conversion/feature
  stages separately. Do not restart another worker's download or modify its scripts during
  this diagnostic-only request. No full-mixture/model-improvement claim follows from this audit.

### 2026-09-06 — retract raw-weight sample-share claims

- **Test.** Verify the earlier reduced-mixture justification that omitted large sources
  represent 3% of UniVLA samples and Ego4D represents 9%.
- **Setup.** Static inspection of local `omni_magic_soup_plus_plus`, RLDS wrapper and
  interleaved sampler; no training, split or seed budget. Applies to the pinned local code
  path, not an independently verified reconstruction of every released paper run.
- **Result.** Literal weights sum to `40.2`, not `44.3`. Wrapper sets `balance_weights=True`;
  sampler computes `p_i = w_i N_i / sum_j(w_j N_j)` with recorded transition counts `N_i`.
  Actual full-mixture probabilities were not computed because required statistics/coverage
  are incomplete. Neither the 3% nor 9% sample-share claim is established.
- **Control.** The sampler call site overrides its function's default balancing flag.
  Raw weights, episode counts and compressed bytes are not interchangeable with transition
  statistics; filtering/caps additionally change effective coverage.
- **Conclusion — retracted.** Retract the earlier numerical sampling shares and the
  negligible-coverage justification based on them. Preserve historical omission decisions
  as history, but do not use those percentages to justify the user's full-mixture plan.
- **Artifacts.** [`mixture definitions`](latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/mixtures.py),
  [`RLDS wrapper`](latent_free/third_party/UniVLA/prismatic/vla/datasets/datasets.py),
  [`interleaved sampler`](latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/dataset.py),
  [`dataset audit`](latent_free/UNIVLA_DATASET_AUDIT_2026-09-06.md).
- **Decision.** Save source statistics, normalized requested probabilities and observed
  batch/window frequencies; use matched actual exposure across controls. The downloader's
  old docstring remains unmodified and its sampling-share statements should not be trusted.

### 2026-09-06 — sequence-preserving full-mixture data contract

- **Test.** Identify preprocessing requirements for spatial × temporal DC+AC1, pair-free
  transport, SSL and subsequent UniVLA-based policy learning on heterogeneous videos.
- **Setup.** Registered recommendations from static code inspection, not implemented or
  trained. Proposed method and mechanism controls share episode/video splits, temporal input
  information, source coverage and budget; original endpoint-only UniVLA remains a reference.
- **Result.** No empirical method result. Pretraining `chunk_act_obs` returns only first/last
  observations, not the full window; LAM YAML `num_frames` alone does not fix this. Current
  converter drops timestamps/FPS, stores only the initial instruction, and uses a capped
  stream/new split. Existing Bridge/OXE holdout fractions differ. Default VLA mixture omits
  Ego4D despite LAM selecting `plus_plus`. Decoded RGB256 costs 196.6 GB/million frames;
  4x4×768 FP16 features cost 24.6 GB/million (layout arithmetic, not measured corpus totals).
- **Control.** Different frame rates and padding can mimic temporal-profile changes;
  duplicate/overlapping episodes can leak across splits. Action-command scale heuristics
  cannot establish realized displacement, units or rotation semantics. Human/navigation
  padded action fields are not common seven-dimensional manipulation labels. Shell DONE
  output without propagated pipeline failure is not a per-source conversion acceptance test.
- **Conclusion — planned.** Build a versioned sequence/time/provenance manifest and real
  multi-frame loader before scale-up; keep the action-free video objective distinct from
  recorded-action diagnostics and target-robot motor supervision. These are prerequisites
  for an interpretable test, not claims that the new method is already successful.
- **Artifacts.** [`dataset audit and proposed contract`](latent_free/UNIVLA_DATASET_AUDIT_2026-09-06.md),
  [`trajectory chunking`](latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/traj_transforms.py),
  [`conversion`](latent_free/phase1/oxe/oxe_convert.py),
  [`conversion job wrapper`](latent_free/phase1/oxe/run_oxe_convert.sh).
- **Decision.** Preserve intermediate observations, real horizon and valid masks, original
  episode/video IDs and time-varying language. Split/group before window extraction; fit
  statistics on train only. Initially restrict swap by compatible embodiment/camera/horizon
  and support without asserting equal factual actions. Share temporal nuisance parameters;
  preserve task-relevant scene geometry. Smoke-test each domain and source frequencies,
  benchmark IO/decode/features, and use sharded storage rather than millions of frame files.
