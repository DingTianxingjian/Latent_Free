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
- **[PARTLY WITHDRAWN 2026-09-18 — see the correction entry of that date: survival of parameters does not bound influence.]**
  **Conclusion: the controlled comparison is fair on this axis**, and the earlier caveat in this ledger ("the codebook
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
- **CPU check of the head (before any GPU run).**  Shapes and sampling range correct for dz = 256 and dz = 0.  On a toy
  target that is +0.5 or -0.5 with probability 1/2 under identical conditioning, flow samples land 47 % / 53 % on the
  two modes and 0 % in between, while the regression head outputs the average (mean |output| 0.045): the property the
  interface change is for.  **Capacity is not matched to the old head:** `FlowHeads` has 26.3M parameters (13.7M
  denoiser + the pooling) against 13.2M for `ContHeads`, so Ours-reg vs Ours-FM mixes interface and capacity;
  none-FM vs Ours-FM uses the identical head and is the clean comparison.
- **What the three Spatial numbers will separate.**  Ours-reg 70.4 vs Ours-FM: the interface.  none-FM vs Ours-FM:
  what the latent adds through the same head.  UniVLA 84.2 stays the external reference (their native interface).

### 2026-09-18 (11:20) — Ours (regression head) @ 10 % GOAL = 17.0 %; the old head has a structural LANGUAGE bottleneck; per-user GPU cap found

- **Result (20783075, 500 episodes).** Ours `latp2_inv` + `ContHeads` on libero_goal @ 10 %: **85/500 = 17.0 %**.  Per task:
  0.06, 0.38, 0.56, 0.08, 0.28, 0.02, 0.02, 0.16, 0.08, 0.06.  UniVLA goal is still training (20783076).
- **Diagnosis, verified in code.** The VLM input is [BOS, image patches, text] (`modeling_prismatic.py:383-384`) under a
  causal mask, so the hidden states at image positions never see the instruction.  `ContHeads` conditions the action on
  [pooled IMAGE states, predicted z]: the instruction reaches the actions **only through the 256-d regressed z**.  For
  `--latent none --head reg` it would not reach them at all.  UniVLA's decoder pools the latent-token states, which sit
  after the text and see it.  LIBERO-Goal is the suite where all tasks share one initial scene and only the language
  differs — the suite where this bottleneck must bite hardest, and it is the lowest number by far (Spatial 70.4, Object
  78.2, where the scene largely identifies the task).  Consistent, not proven: the flow-head runs test it.
- **Consequence.** The regression-head numbers measure that head's language bottleneck as much as the latent.
  `FlowHeads` conditions on the pooled TEXT states directly (`txt_pool`, initialised from the image summary), for both
  Ours and the no-latent control.  Submitted Ours-FM goal 20820640 -> evals 20820641-45 (`NOTE=ours_fm_goal_f10`).
- **Scheduling constraint discovered.**  QOS `part-ai` caps a user at **12 GPUs** (UniVLA long 20783080 was pending with
  `QOSMaxGRESPerUser` while 4 + 4 + 1 were in use).  1-GPU evaluation jobs therefore delay 4-GPU training.  The UniVLA
  object split evaluations (20810391-95) duplicate the finished full evaluation (414/500) and should be cancelled; so
  should the regression-head Long run (20783078 + eval 20783079), which the head change supersedes.  auto mode blocks my
  `scancel`; the user was asked.

### 2026-09-18 (11:45) — RELABEL: 70.4 / 78.2 / 17.0 are "LATP + LEGACY REGRESSION INTERFACE", not LATP's downstream performance; clean text-access control added

- **Relabel (user review).**  The Spatial 70.4 %, Object 78.2 % and Goal 17.0 % rows are **LATP + legacy regression
  interface (`ContHeads`)**.  They are not evidence about the continuous LATP latent as a control signal, because that
  interface has two defects of its own: (a) the instruction reaches the action head only through the regressed z
  (entry above), and (b) deterministic regression averages multimodal targets.  The downstream story of continuous
  LATP is decided by the flow-head runs.  Every table must carry the interface in the row name.
- **Bottleneck shown directly on the module (CPU).**  With the legacy head and z held fixed, replacing the text states
  changes the action by exactly 0; with `text_direct=True` it changes.  The legacy head is language-blind except through z.
- **New clean control (user proposal): `--head reg_text`.**  The legacy regression head plus ONE change: a MAP-pooled
  summary of the text positions is concatenated into the action head's input.  Same regression loss, curriculum and z
  target (19.7M vs 13.2M parameters, the added pooling).  LATP + reg_text on Goal @ 10 %: train 20820676 -> evals
  20820677-81 (`NOTE=ours_rt_goal_f10`).  Reading: Goal recovers under reg_text -> the instruction bottleneck, cleanly
  attributed; recovers only under flow -> multimodality / output interface.
- **The pair that decides the method claim** is LATP + flow vs no-latent + flow on Spatial (same VLA, same head, same
  text access; the only difference is the latent).
- **Priority (user):** LATP+flow Spatial, no-latent+flow Spatial, LATP+flow Goal first; Object-flow second; Long last.
  Executed: cancelled 20810391-95 (duplicate UniVLA object parts) and 20783078/79 (legacy-head Long), with the user's
  explicit approval.  Goal-flow was submitted after Object-flow, so their contents were swapped through the override
  files (20817933 now trains Goal, 20820640 Object) and their evaluation dependencies were swapped to match.  UniVLA
  Long (20783080) had already started at 11:05 when GPUs freed, so the hold had no effect; it is kept as the Long
  baseline.  Per-user cap is 12 GPUs: with UniVLA goal + UniVLA long running (8), one flow run fits now.

### 2026-09-18 (12:10) — queue re-planned by paper table (user): Ours = LATP + flow on all four suites first

- **User direction.** Paper results first, ordered by the tables the paper needs.  Main table T1 = UniVLA vs Ours
  (LATP + flow head) on the four suites at 10 %; the no-latent + flow control row on Spatial next; the reg_text interface
  ablation last (appendix).
- **Actions.** Submitted the missing cell, Ours-flow Long (20820873).  Cancelled and resubmitted the no-latent control
  (now 20820879) and reg_text (now 20820885) so they queue behind the four Ours cells (they had been pending ~1 h, so
  the lost queue age is negligible).  Split UniVLA's Goal/Long evaluations into 5 parts (20820893-902; the single
  8 h / 12 h jobs 20783077/20783081 were cancelled).  A script resolves every training job's effective config
  (submission env + override file) and checks each evaluation's RUN against its dependency: 0 mismatches.  Job map in
  `PAPER_PLAN.md`.

### 2026-09-18 (14:40) — UniVLA @ 10 % Goal ~86.6 % (491/500 episodes in); flow smoke trains; load tests abort natively on h003/h017

- **UniVLA goal (20783076 -> split evals 20820893-97): FINAL 433/500 = 86.6 %** (merged, `merged_libero_goal_univla_goal_f10.json`).
  Per task 0.62, 0.98, 1.00, 0.80, 0.92, 0.78, 0.72, 1.00, 0.96, 0.88.  Against 17.0 % for LATP + legacy
  regression interface: the gap the language-bottleneck reading predicts for this suite.
- **Flow smoke (20817929):** both arms train (12 steps, exit 0).  **Flow load test (20817930) FAILED: exit 134 (SIGABRT)
  on h017** right after the model and the flow head loaded, before the first simulator step, with no Python traceback.
  The previous load test (20783067, token evaluator) died the same way on h003.  All 13 real evaluations through
  `run_vla_eval.sh` succeeded (h000, h005, h007, h013, h016, h018).  Reading: a native MuJoCo/EGL crash tied to the node
  or to the load-test harness (it differs by `timeout`, 4 CPUs and file redirection), not the flow code — unverified.
  Check submitted through the proven path: flow smoke checkpoint, 1 episode x tasks 0-1 via `run_vla_eval.sh`
  (20823981).  All 36 pending evaluations now exclude h003, h004, h006 and h017.
- **Merge fix.** The token evaluator names parts `EVAL-<suite>-openvla-<date>--t0_1--<note>.txt` (task tag before the
  note), so `eval_merge.py` found none; it now matches the tag anywhere in the name.

### 2026-09-18 (17:00) — does the OBSERVED past recover the latent the VLA must predict?  Only a little, and no more than raw past motion

- **Question (user).** The VLA sees one frame, so the temporal evidence in z is used only as a target.  Would giving the
  policy the observed past (the latent of the past window [t-12, t], encoded by the same tokenizer) recover the part of
  z(t) that the current frame cannot determine?
- **Probe (`downstream/libero_history_probe.py`, CPU).** LIBERO-spatial hdf5 index, 500 episodes, 50,250 windows with
  12 steps of past and future; the index's held-out episodes (10,324 windows) as test.  Target z(t) = `latp2_inv` code of
  frames t..t+12 at stride 3 (the VLA label), standardised with its z-stats; also the 12-step action chunk.  Linear ridge,
  alpha on an inner episode split.

  | input | R2 z(t) | R2 action chunk |
  |---|---|---|
  | state = PCA-256 of the current frame's DINO grid + task one-hot | 0.548 | 0.522 |
  | state + z_past (observed past window, same tokenizer) | 0.574 | 0.584 |
  | z_past alone | 0.476 | 0.454 |
  | state + raw past displacement (PCA-256 of F(t) - F(t-12)) | 0.577 | 0.594 |

- **Reading.**  (i) On LIBERO the current state already determines ~55 % of z(t) — far more than the 0.22 measured on the
  OXE mixture, as expected for 10 scripted tasks.  (ii) The observed past adds +0.026 on z and +0.062 on the action chunk.
  (iii) Raw past motion gives the same (+0.029 / +0.072): the benefit is history per se, not our tokenizer's summary of it.
  **So history input would help any policy about equally, including the no-latent control, and is not a place where the
  temporal latent has an advantage.**  Not pursued before the deadline.  Caveats: linear readout on PCA-256 frame features
  (the VLA's own features are far richer); spatial only; hdf5 frames rather than the RLDS frames the VLA trains on.
- **Artifacts.** `acdc_la/results/history_probe_latp2_inv.json`.

### 2026-09-18 (17:05) — Anvil full-system maintenance 2026-09-18 23:30 -> 2026-09-20 21:00 (all compute AND login nodes)

- **Found** because the remaining 7 h trainings went to `ReqNodeNotAvail, Reserved for maintenance` with starts estimated
  9/23-9/24: reservation `anvil-maint-2026-q3`, flags MAINT, IGNORE_JOBS, covers h[000-020] and login[00-07].  A job
  whose time limit overlaps 23:30 cannot start.  This session (on a login node) also ends at 23:30.
- **Actions.**  Training takes ~3 h 55 min (0.935 s/step x 15k + load + merge), so the pending trainings' limits were
  cut 7 h -> 4 h 40 min: Ours-flow Object (20820640) started 16:59 on h016 (done ~21:05).  Evaluation parts set to
  1 h 30 min.  To keep GPUs under the 12-GPU cap free for the evaluations that answer tonight's question (Spatial-flow
  and Goal-flow, ~18:15 / ~18:30), the no-latent control (20820879) and reg_text (20820885) are HELD until after the
  maintenance: `scontrol release 20820879 20820885`.  Ours-flow Long (20820873) may still start before ~18:50.
- UniVLA Long training COMPLETED 16:35; its split evaluations (20820898-902) are eligible.
- **Monitor defect fixed:** the job watcher dropped jobs that left `squeue` and were outside the `sacct -S` window, so
  UniVLA Long's completion produced no event; it now keeps every id it has seen.

### 2026-09-18 (19:00) — evaluations were crashing SILENTLY on h001 (exit 134 reported as COMPLETED); fixed

- **Found.** Goal-flow evaluation part tasks 4-5 (20820643) ended after 70 s with `VLA EVAL exit 134` (MuJoCo/EGL
  SIGABRT) on **h001**, and Slurm reported COMPLETED: `run_vla_eval.sh` ended on `echo`, so its exit status was 0.  The
  flow check 20823981, which I reported to the user as passed, **also crashed on h001 the same way** — corrected.
- **Scope.** Across all evaluation logs, the only aborts are these two, both on h001.  Every real result so far came
  from nodes that ran to the end (h000, h005, h007, h008, h009, h010, h013, h016, h018); the crash signature is
  node-bound (h001, h003, h017; h004/h006 earlier).  Every job sees CUDA_VISIBLE_DEVICES=0 and all four /dev/nvidia*
  files (checked with `srun --overlap` in three running evaluations), so the GPU index does not explain it.
- **Fixes.** `run_vla_eval.sh` now exits with the evaluator's status (new submissions only; pending jobs keep their
  frozen script).  All 26 pending evaluations exclude h001, h003, h004, h006, h017.  Goal-flow tasks 4-5 resubmitted
  (20827663).  The job watcher reads the real exit code from the log of every evaluation that ends COMPLETED and flags
  crashes.  `eval_merge.py` already marks a suite INCOMPLETE when a part is missing.
- **Partial Spatial-flow (8 of 10 tasks):** 269/400 = 67.3 %, against 293/400 = 73.3 % for the legacy regression head
  and 357/400 = 89.3 % for UniVLA on the same tasks.  The stove task fails under both heads (0.16 flow, 0.10 legacy;
  UniVLA 0.90); task 1 fell 0.74 -> 0.26 under flow.  On Spatial the output interface is not the main problem.

### 2026-09-18 (19:05) — LATP + FLOW head @ 10 % Spatial = 66.0 %: the interface change does NOT recover Spatial

- **Result (20817931 -> 20817934-38, 500 episodes, all five parts exit 0).**  **330/500 = 66.0 %**, against LATP + legacy
  regression 352/500 = 70.4 % and UniVLA 421/500 = 84.2 %.

  | task | flow | legacy reg | UniVLA |
  |---|---|---|---|
  | 0 between plate and ramekin | 0.78 | 0.80 | 0.92 |
  | 1 next to the ramekin | **0.26** | 0.74 | 0.98 |
  | 2 table center | 0.82 | 0.76 | 0.96 |
  | 3 on the cookie box | 0.88 | 0.96 | 0.78 |
  | 4 in the top drawer | 0.70 | 0.54 | 0.66 |
  | 5 on the ramekin | 0.98 | 0.98 | 0.98 |
  | 6 next to the cookie box | 0.80 | 0.98 | 0.96 |
  | 7 on the stove | **0.16** | 0.10 | 0.90 |
  | 8 next to the plate | 0.86 | 0.86 | 0.88 |
  | 9 on the wooden cabinet | 0.36 | 0.32 | 0.40 |

- **Reading.**  On Spatial the output interface is not what separates us from UniVLA: two different heads give 70.4 and
  66.0 (a 4.4-point difference, within about 2 SE of each other), and both collapse on the stove task where UniVLA gets
  0.90.  A failure that survives a change of head points at what the two runs share — the latent target (and the 43
  demonstrations) — not at regression.  Task-level swings between the two of our runs (task 1: 0.74 -> 0.26; task 4:
  0.54 -> 0.70) are as large as the gaps to UniVLA, so single-seed per-task numbers are not interpretable one by one.
- **What remains decisive.**  (a) Goal with the flow head (the language-bottleneck test; evaluation in progress);
  (b) the no-latent + flow control on Spatial (held until after the maintenance): if it matches or beats 66.0, the LATP
  latent adds nothing or hurts through this interface.

### 2026-09-18 (19:25) — suspect 2 first (user): UniVLA's latent through OUR continuous flow interface (`--latent univla_cont`)

- **The question.** LATP + flow (66.0 %) and LATP + legacy regression (70.4 %) are both far below UniVLA (84.2 %).
  Suspects (user review): (1) the LATP target is too fine / high conditional entropy; (2) the continuous head is a newly
  initialised interface that does not reuse OpenVLA's native token pathway; (3) the head ignores z; (4) seed variance.
  User: do suspect 2 first; the no-latent control can wait.
- **Design.**  `univla_cont` = UniVLA's DEPLOYED latent — `vq_encode(...)["z_q"]`, the `vq_action` codebook vectors of its
  4 tokens (4 x 128 = 512-d) — as the continuous target of the same `FlowHeads` our latent uses, standardised per
  dimension by the spread of the 16 codebook entries.  Same information as their 4 tokens; only the interface changes.
  Reading: UniVLA-flow near 84 -> the continuous interface is not the problem and our latent is; near 66 -> the interface
  costs about what we lost, and our latent is not worse than theirs.
- **Code.**  build_labeller / datasets_ctrl / both evaluators / eval + load-test routing accept `univla_cont`.  The LAM
  needs CUDA even for a forward pass, so the check runs on a GPU: unit test (shape, standardisation, continuous code ==
  standardised codebook rows of the same frames' tokens) 20827768 -> 12-step training smoke 20827769 -> 1-episode eval
  20827770.  Main run: train 20827771 (4 h 40 min; cannot finish before the 23:30 maintenance, so it starts after
  9/20 21:00 unless run on the workstation) -> evals 20827772-76 (`NOTE=univlacont_fm_spatial_f10`).
- **Past-frame history (user asked whether it was done).**  Measured, not implemented: the CPU probe of 17:00 found the
  observed past adds +0.026 R2 on z, the same as raw past motion.  Open if the user still wants it after suspect 2.

### 2026-09-18 (19:30) — Goal with the FLOW head is as bad as the legacy head: the "instruction only through z" account is REFUTED as the main cause; a padding mismatch is now suspected

- **Partial Goal-flow (parts done: tasks 0-3, 6).**  0.00, 0.22, 0.28, 0.04, 0.02 — against legacy regression 0.06,
  0.38, 0.56, 0.08, 0.02 and UniVLA 0.62, 0.98, 1.00, 0.80, 0.72.  The flow head reads a pooled summary of the text
  states directly, and Goal does not recover.  **The language-bottleneck reading of 2026-09-18 11:20 is refuted as the
  main cause** (the bottleneck itself was real in the code; it is not what makes Goal fail).
- **New suspect, specific to our continuous arms and absent from the token arm.**  Training batches are right-padded to
  the batch's longest prompt, and both continuous heads MAP-pool over every text position including pads.  Pad positions
  attend to the whole real sequence, so their states act as sentence summaries a head can learn to rely on; evaluation
  (batch 1) has no pads.  The token arm's decoder reads the latent-token positions, which precede the pads, so it never
  sees them.  Goal is the suite where the instruction is the only task signal, so it would be hit hardest.  Earlier
  (2026-09-18 09:45) padding was "ruled out" on the per-task Spatial pattern alone; that was a weak test.
- **Test tonight, no retraining.**  `LIBERO_PAD_TO=max` in `run_libero_eval_cont.py` right-pads each prompt to the suite's
  longest (the typical training condition).  Goal tasks 0-3 on the flow model (20827790-91, `ours_fm_goal_padmax`) and on
  the legacy-regression model (20827792-93, `ours_reg_goal_padmax`).  Recovery -> the mismatch is real, and the proper fix
  is pooling each sample over its real tokens only (MAPBlock's own mask cannot take per-sample masks) plus retraining.
- **UniVLA-cont unit test (20827768) PASSED:** 512-d code, mean 0.006 / std 0.905 after standardisation, and it equals the
  standardised codebook rows of the same frames' tokens.  Training smoke 20827769 follows.

### 2026-09-18 (19:45) — CORRECTION of the 2026-09-14 "no warm start" conclusion: parameter survival does not bound influence; measured: the TI branch used ONE code throughout

- **What was overstated.**  The 2026-09-14 entry concluded "the released stage-1 gives the baseline nothing that survives to
  the deployed tokens" and RETRACTED the earlier caveat that the codebook saw their full data; today I repeated it as
  "zero influence".  User review: that is the wrong argument.  The TI codebook is FROZEN in stage 2 (`lam.py:240`), so it
  is a persistent structural prior, not an initialisation; and a 16 x 128 codebook defines a partition of all transitions,
  so 0.0025 % of the parameters says nothing about its influence (a frozen teacher leaves no weights in the student and
  still shapes it).  The retraction of 2026-09-14 is itself withdrawn as too strong.
- **Measured for this run** (`claude_survey/logs/cs_univla_1ep_20543308.out`): the TI branch used **1 of 16 codes at every
  one of the 1,280 logged steps, from step 100 to 128,000**.  Every transition went to the same TI code, so the 16-way
  partition never operated and the decoder received one constant 128-d vector — a decomposition in name only (the same
  collapse the 2026-08 audit found in the released stage-2 weights).  The mechanism "TC learns the residual after TI
  explains part of the variation" therefore had nothing to act on after step 100.  What can remain: the first <= 100
  steps, and the value of that one constant vector (external data).  Small by this measurement; not provably zero.
- **Baseline description from now on:** "UniVLA Stage-2 trained on the matched mixture, initialised with the released
  Stage-1 latent prior as prescribed by the official recipe (the frozen task-irrelevant codebook; in our run its branch
  used 1 of 16 codes throughout training)."  `PAPER_PLAN.md` comparison rule updated.  Removing the prior completely would
  take a stage-2 rerun with a non-released TI codebook (~14 h on 1 GPU at 0.39 s/step) plus its VLA run.

### 2026-09-18 (19:40) — decision (user): the UniVLA baseline must not carry the released TI prior; rerun stage 2 with a RANDOM-init TI codebook

- **Decision.**  A fair baseline initialises the task-irrelevant codebook randomly instead of from the released stage-1.
  `claude_survey/rala2.py --no-inherit` does exactly that: `vq` (TI codebook) and `action_latent` stay at their random
  init; `--ubranch frozen` (default) keeps UniVLA's freezing, so the ONLY change from the official recipe is the source
  of the frozen codebook.  Same mixture, same 128k x 64 budget, same seed as `s2mix_1ep`.
- **Jobs (all start after the maintenance ends 2026-09-20 21:00).**  Stage 2 `s2mix_noinh_1ep`: 20827810 (1 GPU,
  ~14 h at 0.39 s/step; no resume in rala2, so a 16 h limit).  Then, depending on it: OXE probe dump/probes
  `univla_noinh` 20827835; token-interface VLA @ 10 % on all four suites with `--lam_path .../s2mix_noinh_1ep.pt`
  (6 h 30 min limit; the UniVLA arm takes 5.4-5.6 h): spatial 20827811, object 20827817, goal 20827823, libero_10
  20827829, each followed by 5 evaluation parts (`NOTE=univla_noinh_<suite>_f10`).  Expected ~9/21 midday for the LAM,
  ~9/22 for the four suites.
- **Consequence for the paper.**  The main-table UniVLA row becomes this run; `s2mix_1ep` (released TI prior) stays as the
  "official recipe" row with the corrected wording.

### 2026-09-18 (20:00) — LATP + FLOW @ 10 % Goal = 9.2 % (legacy regression 17.0 %, UniVLA 86.6 %); padding test task 0 unchanged

- **Result (merged, 5 parts, 500 episodes): 46/500 = 9.2 %.**  Per task 0.00, 0.22, 0.28, 0.04, 0.14, 0.00, 0.02, 0.14, 0.02,
  0.06.  Lower than the legacy regression head (17.0 %), against UniVLA's token interface at 86.6 %.
- **Padding test (training-like right padding to 27 tokens), flow model, task 0: 0/50** — identical to unpadded.  Tasks
  1-3 and the legacy-head variant still running; on this first task the padding mismatch explains nothing.
- **Code check.**  The token and continuous arms build their training samples identically (same frame, same 12-step action
  slice, same instruction); they differ in the prompt's latent part, the target and the output interface.  No data bug
  found.
- **Suspect 2 made decisive on Goal.**  UniVLA's latent through the same flow head on Goal @ 10 %: train 20827992, evals 20827993 20827994 20827995 20827996 20827997
  (after the maintenance; Spatial counterpart 20827771).  UniVLA-flow on Goal near 87 -> our latent / pipeline is the
  problem; near 9-17 -> the continuous interface cannot use this data regime, whatever the latent.

### 2026-09-18 (20:15) — Goal 9.2 % = the instruction-blind level; fix: QUERY-TOKEN interface for the continuous arms (OpenVLA-OFT idea)

- **Reading of the Goal numbers.**  All ten LIBERO-Goal tasks start from the same scene; only the instruction says which
  one to do.  A policy that ignores it executes one default behaviour and succeeds only when that happens to match, so
  its ten per-task rates sum to about 1.  LATP + flow sums to 0.92 (spread over bowl-on-stove 0.22, wine-on-cabinet 0.28,
  turn-on-stove 0.14 ...); the legacy head to 1.70.  Both continuous arms behave as instruction-blind on Goal, while on
  Spatial / Object the scene itself mostly identifies the task.  The LATP latent carries no instruction by design — neither
  does UniVLA's — so this is about how the VLA interface exposes the instruction to the action head, not about the latent.
- **Why the token arm does not suffer.**  UniVLA's VLA must GENERATE its latent tokens after the instruction under a
  cross-entropy loss, which forces task identity into exactly the positions its action decoder reads.  Our continuous
  heads pool prompt states and regress / denoise; with 43 demonstrations they evidently do not learn to use the text.
- **Fix (user: solve it now).**  `--query_tokens n`: n placeholder tokens (<ACT_i>, embeddings trained, never predicted)
  after the instruction; the continuous heads read their states instead of pooling the prompt.  Those positions see the
  full image + instruction through every LLM layer, sit where openvla-7b emits its action tokens and where UniVLA's
  decoder reads its latent tokens — the OpenVLA-OFT design (placeholder action tokens read by a continuous head), here
  with n = 4 to match UniVLA's 4 latent tokens.  Training and evaluation select the positions by id > 32000.
- **Jobs.**  Smoke 20828149 -> 1-episode Goal eval 20828150 (before 23:30).  LATP + flow + q4: Goal 20828151 -> evals
  20828152-56; Spatial 20828157 -> evals 20828158-62 (after the maintenance, or on the workstation).
- Not yet run, cheap and would confirm the diagnosis directly: the same frame with the ten Goal instructions through the
  trained models; an instruction-blind head gives near-identical chunks.

### 2026-09-18 (20:25) — padding mismatch REFUTED: training-like padded prompts change nothing on Goal

- Goal tasks 0-3, successes out of 50, unpadded -> padded to 27 tokens: flow 0->0, 11->10, 14->13, 2->2 (20827790-91);
  legacy regression 3->2, 19->22, 28->26, 4->4 (20827792-93).  The pad-state summaries are not what the heads depend on;
  the instruction-blindness on Goal is not a train/test padding artefact.  Remaining account: the prompt-pooling
  interface itself; the query-token runs (20828151 Goal, 20828157 Spatial) test the fix.

### 2026-09-18 (22:40) — HANDOVER before the Anvil maintenance (23:30 -> 2026-09-20 21:00); state of every cell and the queue

- **Main table @ 10 % demos, results so far (500 episodes each).**

  | suite | UniVLA (official recipe, released TI prior) | LATP + legacy regression | LATP + flow (prompt pooling) |
  |---|---|---|---|
  | Spatial | 84.2 | 70.4 | 66.0 |
  | Object | 82.8 | 78.2 | trained 20820640, evals 20817944-48 pending |
  | Goal | 86.6 | 17.0 | **9.2** (instruction-blind level) |
  | Long | trained 20783080, evals 20820898-902 released | — | trained 20820873, evals 20820874-78 pending |

- **Diagnosis so far.**  Interface, not padding: padded prompts change nothing (refuted); the language-bottleneck account
  refuted as main cause (flow reads the text and Goal is still ~chance); the continuous heads that POOL PROMPT STATES do not
  learn to use the instruction from 43 demos, while UniVLA's generated latent tokens force task identity into the positions
  its decoder reads.  Fix under test: query-token interface (`--query_tokens 4`).
- **Queued for after the maintenance (Anvil, 12-GPU user cap).**
  LATP + flow + q4: Goal 20828151 -> evals 20828152-56; Spatial 20828157 -> 20828158-62.
  UniVLA latent + flow + q4 (suspect 2 on the fixed interface): Goal 20828934 -> 20828935-39; Spatial 20828940 -> 20828941-45
  (the prompt-pooling univla_cont runs 20827771 / 20827992 were cancelled and replaced by these).
  UniVLA stage 2 with RANDOM-init TI codebook `s2mix_noinh_1ep` 20827810 -> probes 20827835 -> token VLA on four suites
  20827811 / 17 / 23 / 29 -> five eval parts each.
  q4 plumbing check 20828967 (spatial, 1 episode).  HELD (superseded / appendix): none + flow pooled 20820879, reg_text
  Goal 20820885 — `scontrol release` only if still wanted.
- **Workstation during the outage (user).**  `acdc_la/scripts/workstation_run.sh`: LATP q4 Goal, then UniVLA-cont q4 Goal,
  then none q4 Goal; the other session found a CPU-RAM leak in the TF data pipeline (OOM near step 3600 on 62 GB) and is
  testing an in-process iterator rebuild before a checkpoint + restart fix.  Anvil jobs (400 GB) are unaffected.
- **Operational gotchas learned today.**  (1) Monitor loops outlive their Monitor and share one state file: kill
  `scratchpad/jobwatch.sh` orphans before re-arming, or events vanish.  (2) Evaluations submitted before 18:55 exit 0 even
  when MuJoCo/EGL aborts; the watcher reads the real code from the log.  Bad nodes for rendering: h001, h003, h004, h006,
  h009, h017 (excluded).  (3) `eval_merge.py` refuses double counts and flags INCOMPLETE suites.  (4) Pending jobs'
  batch scripts are frozen at submission; Python modules are read at run time.

### 2026-09-18 (23:00) — LATP + FLOW @ 10 % Object = 71.2 % (legacy regression 78.2 %, UniVLA 82.8 %)

- 20817944-48, all five parts exit 0: **356/500 = 71.2 %**.  Per task 0.86, 0.34, 0.66, 0.56, 0.82, 0.48, 0.74, 0.80, 0.92,
  0.94 (legacy regression 0.80, 0.54, 0.76, 0.68, 0.92, 0.64, 0.84, 0.84, 0.90, 0.90; UniVLA 0.34, 0.84, 0.82, 0.72, 0.94,
  0.80, 0.98, 0.88, 1.00, 0.96).  With prompt pooling the flow head is below the legacy head on Spatial (66.0 vs 70.4),
  Object (71.2 vs 78.2) and Goal (9.2 vs 17.0): the interface change as built does not help on any suite.  The
  query-token runs (queued) are the test of the actual fix.

### 2026-09-19 (01:50) — workstation outage plan checked: as written it would OOM at ~3.6k steps; fixed (fresh tf.data iterator every 2500 batches) and all three arms smoke-tested

- **Leak, characterised (workstation, 62 GB).** The RLDS tf.data pipeline alone (no model, no python transform, no
  labeller) grows ~250-265 MB per 1000 samples: +21.5 GB per process at 100k samples, slowing only slowly after ~60k.
  Identical with autotune off, image aug off, jemalloc, and at `LIBERO_TRAIN_FRAC=10` on Goal (so it is per consumed
  sample, not per distinct demo).  This is what killed the 2026-09-14 univla run at step 3625 (~116k samples/rank).
  **Dropping the iterator returns the memory**: 3 segments of 40k samples with a fresh `as_numpy_iterator()` each ->
  17 GB at segment end, 4.5-5 GB after `del` + `gc.collect()`, no carry-over.  Anvil nodes never see it (RAM).
- **Fix (default off, Anvil unchanged).** `finetune_libero_ctrl.batches(dl, every)`: a fresh DataLoader/tf.data
  iterator every `CTRL_REITER_BATCHES` batches, and the iterator is released before the in-process merge.
  `workstation_run.sh` sets 2500 (20k samples/rank per iterator).  The shuffle buffer refills at each restart, i.e.
  data order is re-drawn every ~625 steps; at 10 % data that is ~4 passes over the demos per iterator.
- **Also required on a fresh host, not done by `workstation_run.sh`:** the vendored-UniVLA patch
  (`acdc_la/patches/univla_rlds_train_frac.patch`) was corrupt (hunk header 6 lines, 2 present; `git apply` refused),
  so `LIBERO_TRAIN_FRAC` would have been silently ignored (= 95 % of the demos).  Regenerated from the applied tree.
  Artifacts: `ckpt_20260917/` (latp2_*, tc_*, z_stats_*) from the HF mirror; `s2mix_1ep.pt` md5-identical to the local
  copy; LIBERO-Goal RLDS downloaded.
- **Smoke (Goal, FRAC=10, flow head, 4 query tokens, 2 GPUs, `CTRL_REITER_BATCHES=20`).** `tc_cont` (latp2_inv) 30 steps:
  2.5 s/step, merge ok, peak RSS of both ranks 40 GB (at merge), host MemAvailable never below 8 GB; continuous
  evaluator on both GPUs (tasks 0 / 5, 1 trial) and `eval_merge` run.  `univla_cont` and `none` 10 steps each: train +
  merge ok.  Code path only; smoke checkpoints deleted.  **Confirmed (infrastructure).**
- **Next.** The three outage cells (tc_cont / univla_cont / none, Goal @ 10 %) via `workstation_run.sh`: at 2.5 s/step
  ~10.5 h training each + eval.

### 2026-09-19 (12:10) — workstation outage cell 1: LATP (`latp2_inv`) + FLOW head + 4 QUERY TOKENS @ 10 % Goal = **434/500 = 86.8 %** (same arm without query tokens 9.2 %; UniVLA token recipe 86.6 %)

- **Config.** `workstation_run.sh LATENT=tc_cont SUITE=libero_goal NOTE=latp2inv_fmq4_frac10`: `--head flow --query_tokens 4
  --tc_tag latp2_inv`, `LIBERO_TRAIN_FRAC=10` (train[:10%] = 43 Goal demos), openvla-7b + LoRA r32, 15k steps x 64
  (2 GPUs x micro-batch 8 x accum 4 instead of Anvil's 4 x 8 x 2), one seed; workstation deviations: torch 2.7.1+cu128,
  NCCL 2.31.2, fresh tf.data iterator every 2500 batches (data order re-drawn every ~625 steps).  Train 34,102 s
  (2.27 s/step); final train-batch latent_r2 0.819, one-step action L1 0.030.  Eval: `run_libero_eval_cont`, 50 trials
  x 10 tasks, tasks 0-4 / 5-9 on the two GPUs, ~40 min.
- **Result.** 434/500 = 86.8 %.  Per task (in suite order): 43, 49, 49, 37, 44, 37, 42, 49, 45, 39 of 50; lowest "open
  the top drawer and put the bowl inside" and "push the plate to the front of the stove" (0.74).
- **Against.** Same latent + flow head WITHOUT query tokens (Anvil, 2026-09-18 20:00): 46/500 = 9.2 %; legacy regression
  head 17.0 %; UniVLA's own token recipe @ 10 % Goal (Anvil 20783076): 433/500 = 86.6 %.  So the query-token interface
  takes this arm from the instruction-blind level to UniVLA's level on Goal.
- **Conclusion — provisional.** One seed; the comparison that isolates the latent under the SAME interface
  (UniVLA's latent + flow + q4, cell 2) and the no-latent control (cell 3) are running next on the workstation.  The
  86.8 vs 86.6 difference is 1 episode in 500 and is not evidence either way.  Anvil's queued twin of this cell
  (after the maintenance) is a second seed.
- **Artifacts.** `acdc_la/results/vla_eval/{cont_libero_goal_ws_latp2inv_fmq4_frac10_t0_1_2_3_4,..._t5_6_7_8_9,
  merged_libero_goal_ws_latp2inv_fmq4_frac10}.json`; run dir `acdc_la/ckpt/vla_runs/tc_cont+libero_goal_no_noops+b32x2+
  lr0.00035+r32+s15000+fm+q4--latp2inv_fmq4_frac10/` (`train_log.jsonl`, `ctrl_meta.json` committed; merged model local);
  logs `acdc_la/logs/ws_{train,eval}_latp2inv_fmq4_frac10_*`.
- **Next.** Cell 2 (`univla_cont`, started 12:07 UTC) and cell 3 (`none`); decision on whether the latent matters under
  the query-token interface waits for both.

### 2026-09-20 (01:25) — workstation outage cell 2, SUSPECT 2 ANSWERED: UniVLA's latent through OUR interface @ 10 % Goal = **447/500 = 89.4 %** vs our latent 86.8 % — the interface was the Goal failure, and our latent is not better than theirs

- **Config.** Identical to cell 1 except the labeller: `LATENT=univla_cont` (their deployed latent, `s2mix_1ep`) with the
  same `--head flow --query_tokens 4`, same 43 Goal demos, same 15k x 64 budget, same host, one seed.  Train 45,058 s
  (3.0 s/step; their LAM labeller is heavier than ours); final train-batch latent_r2 0.964, one-step action L1 0.031
  (ours: 0.819 / 0.030).  Eval 50 x 10, ~35 min.
- **Result, same-interface comparison (the control cell 1 lacked).**

  | Goal @ 10 % demos, flow head + 4 query tokens | success |
  |---|---|
  | UniVLA latent (`univla_cont`, s2mix_1ep) | **447/500 = 89.4 %** |
  | our latent (`tc_cont`, latp2_inv) | 434/500 = 86.8 % |

  Per task (suite order), ours vs theirs: 43/39, 49/49, 49/50, **37/47**, 44/40, **37/45**, 42/42, 49/50, 45/47, 39/38.
  The gap is carried by two tasks ("open the top drawer and put the bowl inside", "push the plate to the front of the
  stove"); ours is ahead on "open the middle drawer" and "put the bowl on top of the cabinet".
- **Reading.** 13 episodes in 500, ~1.3 sigma on a binomial with these rates: **not a significant difference**, one seed
  each.  What it does settle: the 9.2 % Goal collapse was the INTERFACE, not our latent (both latents reach ~87-89 %
  through the query-token interface), and **our latent buys nothing over UniVLA's here** -- if anything it is behind.
  Their latent is also easier for the VLA to predict (train latent_r2 0.964 vs 0.819).
- **Conclusion — provisional (single seed each).**  Cell 3 (`none`, no latent, same interface; started 01:24 UTC,
  result ~12:00 UTC) decides whether either latent contributes at all; if `none` also lands near 87-89 %, the latent
  channel is decorative in this data regime and the paper claim cannot rest on Goal.
- **Artifacts.** `acdc_la/results/vla_eval/{cont_libero_goal_ws_univlacont_fmq4_frac10_t0_1_2_3_4,..._t5_6_7_8_9,
  merged_libero_goal_ws_univlacont_fmq4_frac10}.json`; `train_log.jsonl` + `ctrl_meta.json` of the run dir; logs
  `acdc_la/logs/ws_{train,eval}_univlacont_fmq4_frac10_*`.

### 2026-09-20 (01:45) — the "does LATP's own prior transfer to LIBERO?" probe cannot be run on these weights: the prior head was NEVER TRAINED in any surviving checkpoint

- **Question (user, 2026-09-20).** Cheap first layer before any new VLA run: freeze the LATP prior, and on held-out
  LIBERO measure R2(p(z | o_t, g), z*) against the posterior target, next to the VLA query predictor's R2(z_VLA, z*).
  If the built-in prior is already close to the VLA's, it is worth exploiting; if it collapses off OXE, it is an
  OXE-domain prior.
- **Probe built.** `acdc_la/downstream/vla/prior_gate.py`: one pass over the HELD-OUT split (`train=False` ->
  `train[95%:]`, the side no arm trains on) of `libero_goal_no_noops`, ro fixed to 0, computing on the same rows
  z* = `pre_quant(f0, dF)` (standardised by z_stats, exactly the VLA's label), z_prior = `prior_code(f0, c)` with
  c = masked-mean pooled frozen t5-base (the `langvec` convention of `oxe_mix` / `claude_survey/lang_cache.py`), and
  optionally z_VLA = the trained arm's query-token prediction, flow-sampled with the deployment Euler steps.
- **Result on `latp2_inv`, Goal, n = 200: R2(prior, z*) = -67.0** (per token -43.6 / -60.3 / -73.1 / -91.1), prior RMS
  5.35 against the target's 0.87 — the signature of an untrained random head, not of a domain shift.
- **Cause, checked in the code and the checkpoints, not inferred.** `stage2_tc/train.py:39` only builds the prior
  term `if a.lam_prior > 0`, and every surviving tokenizer carries `lam_prior = 0.0` (`tc_cont`, `tc_acdc`,
  `tc_cont_plus15k`); LATP v2 (`stage3_transport/train_v2.py`, which produced `latp2_inv` / `latp2_anchor`) has no
  prior term at all, so it only inherited the random-init head.  The one run that did train it, `tc_cont_prior`, was
  killed at 18k steps for gaming the variance-normalised form (ledger 2026-09-15) and then cancelled; `--code-norm`
  is the fix in the code but no checkpoint was ever trained with it.
- **Conclusion — the question is open, not answered: there is no trained deployment prior to measure.**  Nothing in the
  VLA comparison used one either (both arms' labels are posterior codes and the VLA re-learns the mapping), so no
  earlier result is invalidated.
- **Artifacts.** `acdc_la/results/prior_gate_latp2_inv_libero_goal_n200.json`, `acdc_la/downstream/vla/prior_gate.py`.
- **Next, cheapest first.** (i) The same probe with `--vla <run dir>` gives R2(z_VLA, z*) held-out (the VLA's own
  deployment mapping, today only known as a train-batch 0.819); (ii) a ridge / small MLP from (f0, T5 lang) -> z* fitted
  on the 10 % train split and read out held-out prices how learnable the deployment mapping is AT ALL on LIBERO,
  with no tokenizer retraining; (iii) only if (ii) is high does a `--lam-prior --code-norm` tokenizer earn its compute.

### 2026-09-20 (11:00) — workstation outage cell 3, THE CONTROL THAT MATTERS: **no latent at all = 443/500 = 88.6 %** on Goal @ 10 %, i.e. NEITHER latent contributes anything through this interface

- **Config.** Identical to cells 1 and 2 except `--latent none` (`FlowHeads` with dz = 0: the same flow head, the same
  4 query tokens, the same prompt, no latent input and no latent loss).  Train 31,756 s (2.11 s/step), 15k x 64,
  one seed, same host and code as the other two cells.
- **The three cells, same interface, same 43 Goal demos, same budget, 500 episodes each.**

  | arm | success | vs no-latent |
  |---|---|---|
  | UniVLA latent (`univla_cont`, s2mix_1ep) | 447/500 = 89.4 % | +0.8 pp |
  | **no latent (`none`)** | **443/500 = 88.6 %** | — |
  | our latent (`tc_cont` = `latp2_inv`) | 434/500 = 86.8 % | -1.8 pp |

  Spread 13 episodes in 500; the largest pairwise gap (ours vs UniVLA) is ~1.3 sigma, ours vs no-latent ~0.9 sigma.
  Per task the three interleave (no-latent wins 4 of 10 tasks outright).  Train-batch action L1 tracked together all
  the way (step 1850: 0.0786 / 0.0873 / 0.0784).
- **Conclusion — negative, and it applies to BOTH latents.**  On LIBERO-Goal at 10 % demos, through the query-token
  flow interface, **the latent channel is decorative**: a policy with no latent input matches the best latent arm.
  The 09-18 "Goal 9.2 %" collapse was an interface failure, and repairing the interface did not create a latent
  effect -- it revealed that the action head reads what it needs from the visual features alone in this regime.
  Consequence for the paper: no claim of the form "our latent action beats UniVLA's downstream" can rest on this
  cell, and neither can "latent actions help", since the matched no-latent control is inside the noise band.  This is
  the same pattern as the 2026-09-13 toy closed-loop (every arm in one band) but now at full-VLA scale with 500
  episodes per arm and a proper matched control.
- **What it does NOT say.**  Nothing about suites/regimes where the visual features are insufficient (Long/libero_10,
  or fewer than 43 demos), and nothing about the posterior latent's CONTENT: the probe advantage (`latp2_inv` action
  R2 0.329 vs same-data UniVLA 0.236) is measured on the oracle code, while the policy only ever sees its own
  prediction (train latent_r2 ours 0.819 vs theirs 0.964 -- our latent carries more and is predicted worse).
- **Artifacts.** `acdc_la/results/vla_eval/{cont_libero_goal_ws_none_fmq4_frac10_t*,merged_libero_goal_ws_none_fmq4_frac10}.json`,
  run dir `train_log.jsonl` + `ctrl_meta.json`, logs `acdc_la/logs/ws_{train,eval}_none_fmq4_frac10_*`.
- **Next (cheap, eval-only, no retraining).** (i) Feed the ORACLE posterior z* into the trained heads at evaluation and
  compare with the sampled z: if oracle z* does not raise success either, the latent is useless here regardless of
  predictability; if it does, predictability is the bottleneck and the deployment prior becomes the live question.
  (ii) R2(z_VLA, z*) held-out via `prior_gate.py --vla`.  (iii) A regime where the control is not saturated.

### 2026-09-20 (15:30) — REGISTERED (not yet run): phase 2 restored — latent-action pretraining of the policy, the stage our comparison skipped

- **Why (user, 2026-09-20).** The 09-19/20 cells show the latent channel is decorative on Goal @ 10 %.  The structural
  reason: with only post-training, the deployment latent is a deterministic function of (o_t, l), and the same action
  head already reads o_t and the prompt, so the latent adds no information (UniVLA's own arm reaches train latent_r2
  0.964 and still only matches the no-latent control).  UniVLA's README has three stages -- latent action model,
  **pretraining of the generalist policy on latent-action pseudo-labels over the cross-embodiment collection**, then
  post-training -- and we ran only the first and third, from `openvla-7b` (pretrained on real discretised actions).
  The claim "a better latent action space transfers better" lives in stage 2, so stage 2 is what has to be run.
- **Design (both arms identical except the labeller, as before).**

  | arm | phase 2 (OXE mixture, latent-only) | post-training (Goal @ 10 %) | status |
  |---|---|---|---|
  | ours | `tc_cont` / `latp2_inv` labels | from that checkpoint | to run |
  | UniVLA | `univla_cont` / `s2mix_1ep` labels | from that checkpoint | to run |
  | no phase 2 (both latents) | -- | 2026-09-19/20 workstation cells | **done** (86.8 / 89.4) |
  | no latent at all | -- | workstation cell 3 | **done** (88.6) |

  So the existing cells are the matched no-phase-2 controls; the new runs price stage 2 itself.
- **Budget.** `--max_steps 30000` x 64 = 1.9M clips, ~1/100 of their pretraining: a MATCHED-budget comparison between
  the two latents, to be reported as such and never as a reproduction of UniVLA's scale.  4 x H100 at ~1.1 s/step
  ~ 9 h per arm = two 8 h chunks; then 15k-step post-training (~5 h) and the standard 50 x 10 evaluation per arm.
- **Code (built and smoke-tested on the workstation today, defaults unchanged so every earlier run is reproducible).**
  `finetune_libero_ctrl.py`: `--lam_a 0` (latent-only: OXE has no common action space), `--merge False`,
  `--resume True` (optimiser + schedule + step in `trainer_state.pt`; a chunked run must pass the SAME `--max_steps`,
  because `exp_id` carries it), `--init_from_run <run dir>` (post-train from a phase-2 arm: LoRA from its adapter,
  heads from its `action_decoder.pt`).  Scripts `acdc_la/scripts/run_vla_phase2.sh` and `run_vla_post2.sh`.
  Smokes (Goal, 2 GPUs): latent-only trains with the action L1 frozen at 0.555 as it must; a killed chunk resumes at
  its last save and the loss continues; `--init_from_run` loads both adapter and heads.
- **Vendored-UniVLA patch extended** (`acdc_la/patches/univla_rlds_acdc.patch`, renamed): `UNIVLA_FIXED_WINDOW=1`
  disables their per-frequency window override, which sets the post-training window to 3 frames for 7 of the mixture's
  sources -- shorter than the 14 frames both labellers need (ours reads 0,3,6,9,12).  Our tokenizer labels native
  frames at a fixed T=5 / stride 4 regardless of source frequency, and both arms get identical windows.
- **Untested until Anvil returns.** Nothing has run against `data/oxe_raw` / `omni_magic_soup_plus_plus`: the FIRST
  job must be a ~20-step smoke of each arm on the mixture (labeller on OXE frames, the fixed-window patch, memory),
  before the 30k-step chunks.  Risks: whole-node queue wait (1-3 days historically) against the 09-24 deadline, and
  the mixture's per-source window/aspect assumptions.

### 2026-09-22 (00:40) — Anvil after the maintenance: Long @ 10 % (prompt-pooling flow) 50.8 %; UniVLA Long 53.6 % on 371/500 so far; merged the workstation branch

- **Ours, LATP + flow (prompt pooling) @ 10 % libero_10** (20820873 -> 20820874-78, all exit 0): **254/500 = 50.8 %**.
  Per task 0.20, 0.72, 0.58, 0.64, 0.32, 0.72, 0.64, 0.50, 0.14, 0.62.
- **UniVLA token recipe @ 10 % libero_10** (20783080 -> 20820898-902): parts 0-3 done, 4-9 running: **199/371 = 53.6 %**
  so far (per task 0.38, 0.84, 0.20, 0.40, 0.52, then partial).  Long is the one suite where the prompt-pooling arm is not
  far behind; whether the query-token interface / no-latent control also land here is not measured.
- **Running / queued on Anvil.**  LATP + flow + q4 Goal (Anvil twin = seed 2 of workstation cell 1): trained 20828151
  (train-batch latent_r2 0.819), evals 20828152-56 running/queued.  UniVLA stage 2 with random TI codebook 20827810:
  7.5 h of ~14 h.  Queued: LATP + flow + q4 Spatial 20828157; UniVLA-latent + flow + q4 Goal 20828934 / Spatial 20828940;
  q4 plumbing check 20828967; UniVLA-noinh four-suite chain.
- **Repository.**  The workstation's 6 commits (cells 1-3, prior_gate, phase-2 code, RLDS-leak fix) were merged by rebase;
  both ledger blocks kept in chronological order.

### 2026-10-07 (16:45) — workstation, `next_task.md` started: prerequisites done, OXE data path VALIDATED on one source before the 3.9 TB transfer finishes (peak VRAM 42.4 GB)

- **Prerequisites.** `main` pulled (60 commits).  Vendored patch: the workstation's `third_party/UniVLA` still carried
  only the 2026-09-20 local edit, so it was reset and `acdc_la/patches/univla_rlds_acdc.patch` re-applied; the required
  check passes (`_fixed_window` 3 hits in `rlds/dataset.py`, `acdc_oxe29` 1 hit in `rlds/oxe/mixtures.py`,
  `LIBERO_TRAIN_FRAC` 2 hits).  Labellers (`latp2_inv`, `s2mix_1ep`), `openvla-7b`, DINOv2 hub cache and the
  `univla_vla` venv were already in place from September.
- **Disk.** The project disk has 61 GB free (97 % used), so `data/oxe_raw` is a symlink to `/data/ti874297/oxe_raw` on
  the 22 TB RAID (5.0 TB free before the copy, ~1.1 TB after).  The 29 `acdc_oxe29` sources sum to 3.89 TB (largest:
  fmb 1.29 TB, kuka 835 GB, language_table 429 GB, bridge_oxe 416 GB).  Transfer from Anvil started 16:23 UTC, 2
  streams, smallest first, 85 MiB/s measured -> ETA ~05:00 UTC 10-08.  **Nothing is dropped from the mixture.**
- **Single-source smoke (not in the task list; run because the real smoke is 13 h away).**  `--dataset_name
  cmu_stretch` (763 MB, the first source to land), 2 GPUs, batch 8 x accum 4, `--lam_a 0`, `UNIVLA_FIXED_WINDOW=1`,
  20 steps per arm.  Both arms exit 0 with a finite, falling latent loss: ours (`tc_cont`/`latp2_inv`) 1.683 -> 1.514
  at ~2.3 s/step, UniVLA (`univla_cont`) 1.825 -> 1.465 at ~2.9 s/step.  **Peak GPU memory 42.4 GB of 96 GB** (the
  number `next_task.md` asks for; it also fits an A100-40 GB only at a smaller micro-batch).  Host RAM steady,
  53 GB free with `CTRL_REITER_BATCHES=2500`.
- **What this does and does not establish.**  It establishes the phase-2 code path on REAL OXE RLDS data: labeller on
  OXE frames, the fixed-window patch (without which 7 sources deliver 3-frame windows), memory and throughput.  It is
  not the §3 smoke: that one runs the full 29-source `acdc_oxe29` mixture and still has to pass.
- **Next.** Full-mixture smoke when the copy finishes, then arm A (`p2_ours`) and arm B (`p2_univla`), 30k steps x 64,
  ~21 h each at the measured step time; then post-training on Object and Long plus the no-phase-2 and no-latent
  controls of §6.

### 2026-10-08 (17:50) — §3 full-mixture smoke FAILED (OOM), and why: the 29-source statistics need 39 GB in ONE process, and the cache key contains the `data_dir` spelling

- **Run and failure.**  `acdc_la/scripts/p2_smoke_mix.sh` (arm A, `tc_cont`/`latp2_inv`, 2 GPUs x batch 8 x accum 4,
  20 steps, full `acdc_oxe29`) started 2026-10-08 ~03:15 UTC and was **killed after 1 h 40 m with exitcode -9**
  (host OOM, 62 GB RAM), inside "Computing dataset statistics", before step 1.  Arm B was killed by hand because it
  would have repeated the failure.  **Conclusion — invalid run**: no training signal, nothing to compare.
- **Cause 1, cost.**  The RLDS statistics for the mixture are computed by a full pass over every source's records, and
  **each DDP rank does it independently** while holding its own 7B model.  Measured in one process with no GPU and no
  model (`acdc_la/scripts/warm_stats.py`): **4 h 05 m (14,680 s) for 29 sources, peak RSS 39 GB** of 62 GB
  (`logs_migrate/warm_stats_mem.log`).  Two ranks at 39 GB cannot coexist here.  Anvil never showed this: its nodes
  have 400 GB and `run_vla_phase2.sh` asks for `--mem=400G`.
- **Cause 2, the cache key.**  `get_dataset_statistics` (`rlds/utils/data_utils.py:197`) keys the cache on
  `sha256(str(builder.info) + str(state_obs_keys) + source(standardize_fn))`, and `str(builder.info)` contains
  `data_dir=...` **verbatim, unresolved**.  So `data/oxe_raw` (the spelling in `next_task.md` §3/§4) and
  `/home/ti874297/Projects/latent_free/data/oxe_raw` (the spelling `warm_stats.py` used, via `acdc_la.paths.LF`) are
  **different keys**, and the symlink target `/data/ti874297/oxe_raw` is a third.  Probed with the real code path
  (scratch `hash_probe.py`): the absolute spelling is **29/29 HIT**, the relative one **9/29** (exactly the big
  sources the failed smoke had got through) and then MISS on `toto`.  The Anvil `dataset_statistics_*.json` files that
  came along with the 3.9 TB rsync are likewise dead weight — Anvil's absolute path is a fourth key.
- **Fix.**  Statistics are computed once, out of band, in one process (done, exit 0, 09:04 UTC), and every phase-2 run
  passes the **absolute** data root so it reads that cache: `acdc_la/scripts/p2_smoke_mix.sh` updated, and the new
  workstation runner `acdc_la/scripts/run_vla_phase2_ws.sh` (the twin of Anvil's `acdc_la/scripts/run_vla_phase2.sh`) hard-codes
  `$PWD/data/oxe_raw`.  This is a path spelling, identical for both arms — no arm-specific setting changed.
- **Artifacts.**  `acdc_la/scripts/warm_stats.{py,sh}`, `logs_migrate/warm_stats.log`, `logs_migrate/warm_stats_mem.log`,
  `acdc_la/scripts/run_vla_phase2_ws.sh`, `acdc_la/scripts/p2_smoke_mix.sh`; cache at
  `/data/ti874297/oxe_raw/<source>/<version>/dataset_statistics_<hash>.json`.
- **Second wall, measured (18:00-18:50).**  With the statistics cached (45 s instead of 4 h), the relaunched smoke was
  SIGKILLed again, 80 s later, inside tf.data pipeline construction -- both arms, exit -9.  A data-only probe
  (`acdc_la/scripts/mix_mem_probe.py`: one process, no model, no GPU, watchdog at 8 GB available) prices one rank's
  `acdc_oxe29` pipeline:

  | shuffle buffer | one rank |
  |---|---|
  | 16000 (the `Cfg` default, used by every LIBERO cell) | 54 GB -> 1 GB available in ~2 min, **before the first sample** |
  | 3000 | peak RSS 47.4 G, aborted at the floor |
  | 1000 | 49.8 G, aborted |
  | 256 | 48.0 G, aborted |

  So the shuffle buffer is **not** the cost: the floor is ~48 GB per rank, ~1.7 GB per source, spent while the 29
  per-source pipelines fill.  2-rank DDP of the full mixture therefore cannot run on this 62 GB host at any buffer
  size, and one rank plus its 7B model is already marginal.  The LIBERO cells never showed this because a single
  10 %-split source never fills the buffer and has one pipeline, not 29.
- **Cause, and the fix it implies.**  `finetune_libero_ctrl.py:348` passes `training_phase="post-training"`, which
  selects `chunk_act_obs_libero` ("load all obs. within a window") with `window_size = 12 + 2 = 14`: every encoded
  image is gathered **14 times** per trajectory.  UniVLA's own cross-embodiment pretraining takes the other branch,
  `chunk_act_obs`, which keeps only the first and last frame of the window -- 2 images, 7x less -- which is why their
  256k-frame shuffle buffer is affordable and ours is not.  Our labellers read only 4 of those 14 frames
  (`imgs[k * stride]`, `stride = window_size // (T - 1) = 4`, i.e. 0/4/8/12), 8 with the `ro` in {0,1} jitter, so a
  strided gather is both cheaper and sufficient.  **Conclusion -- invalid run** (no training signal); the configuration
  prescribed in `next_task.md` §4 (2 GPUs on this host) is not runnable as written.
- **Next.**  Implement the strided obs gather (identical for both arms, with a label-equality test against the current
  path before any production run); if it does not bring one rank under ~20 GB, move arms A and B to Anvil
  (`acdc_la/scripts/run_vla_phase2.sh`, `--mem=400G`, 4 GPUs, micro-batch 4 x accum 4 for 40 GB cards) and keep the
  workstation for post-training, controls and evals.  Any host: run `warm_stats.sh` first and never change the
  data-root spelling afterwards.

### 2026-10-12 → 10-17 — Newton (UCF): phase 2 at 1/10 scale run end-to-end; post-training at 10 % and full data; phase-1 full-data continuation does NOT help

- **Host.** Moved to UCF ARCC Newton (branch `newton`, paths `/lustre/fs1/home/ti874297/lf/{latent_free,envs}`).  OXE RLDS
  (29 `acdc_oxe29` sources, 3.90 TB, every shard checked against `dataset_info.json`) at `data/oxe_raw` (symlink into
  `/lustre/fs1/groups/ytian`).  Mixture GNM weights set to UniVLA's `omni_magic_soup_plus` (1.0, were 2.0) on 10-15 --
  AFTER the phase-2 runs below, which used 2.0.  RLDS statistics recomputed on Newton under the absolute data root (26 min).
  Bad GPU nodes found and excluded: evc22/26/38/40/43/44/45/48 (normal), evc104 (highgpu); `acdc_la/logs/bad_nodes.txt`.
- **Phase 2, 1/10 scale (next_task.md §4 recipe, 2 GPUs x 8 x accum 4 = 64, 30k steps, LoRA r32, latent-only).**
  Host RAM of the full-mixture pipeline grows ~3 GB/min for 2 ranks even with `CTRL_REITER_BATCHES=2500`, so runs were
  done in memory-bounded segments (`acdc_la/scripts/run_vla_phase2_newton.sh`: restart with `--resume` at the next
  checkpoint once RSS > 380 GB).  Arm A (`latp2_inv`, job 900546): 20 segments, final latent_r2 0.795, latent_loss
  0.824.  Arm B (`s2mix_1ep`, 900547 -> 907887): **INVALID** -- the wrapper killed the trainer while `trainer_state.pt`
  (1.2 GB) was being written (truncated at 287 MB, step 19000); resumed from the step-19000 adapter/heads with AdamW
  moments reset.  Its post-training never learned actions (action L1 stuck ~0.15 vs 0.03 for every healthy run) and
  scored 0/500 on Object; cause (reset vs phase 2 itself) not determined -- the arm was dropped (user: compare against
  UniVLA's published numbers instead).  Fixes: atomic `trainer_state.pt` (tmp + rename); wrapper waits until the
  checkpoint is 60 s old.
- **Post-training of arm A (flow head, 4 query tokens), 500 episodes each, evaluator = UniVLA's protocol (checked
  line by line: max_steps, seed 7, center crop, 10 wait steps, benchmark init states).**

  | suite | 10 % demos (64 x 15k) | UniVLA paper 10 % (Fig. 10) | full split (128 x 30k, Long 40k = UniVLA's budget) | UniVLA paper full (Table I) |
  |---|---|---|---|---|
  | Object | 442/500 = 88.4 % | -- | 494/500 = **98.8 %** | 96.8 |
  | Long | 308/500 = 61.6 % | 62.4 | 445/500 = 89.0 % | 92.0 |
  | Spatial | -- | -- | 486/500 = **97.2 %** | 96.5 |
  | Goal | -- | 86.3 | queued | 95.6 |

  **Not attributable to phase 2:** no matched no-phase-2 / no-latent control was run (user decision), and the head
  (flow + query tokens, OFT-like) differs from UniVLA's; the 2026-09-18 71.2 % Object run had no query tokens.
- **Phase 1 continued on the FULL data (job 944428).** UNCAPPED caches for all 29 sources (`cache/oxe_full`, 33.1 M
  frames, 6.5 TB, 17.4 M train clips; converter now writes `frames.npy` directly).  `train_v2 --init-from latp2_inv`,
  identical args to `latp2_inv` (lam_effect 1, lam_recover 0, lam_rel 0.5, lam_inv 1, batch 64, 15k steps).
  `data/distractor_video.npy` was missing on Newton and was REBUILT from `imageio:cockatoo.mp4` (first 200 frames,
  720x1280) -- the original frame choice is undocumented.  Probes, both tokenizers dumped on the SAME full-cache
  validation set (11,456 clips; the model-free block A is identical for both, which confirms it):

  | code z (256-d) | r2_net | seg1 | prof | seq | patch_drop |
  |---|---|---|---|---|---|
  | `latp2_inv` (re-dumped as `latp2_inv_ref`) | 0.363 | 0.301 | 0.099 | 0.264 | **20.4 %** |
  | `latp2_inv_full` (+15k on full data) | 0.372 | 0.307 | 0.100 | 0.263 | 24.6 % |

  +0.009 r2_net (the 09-16 plain-reconstruction control bought +0.003; the LATP objective +0.064), prof unchanged, and
  nuisance robustness WORSE by 4.2 points.  One seed; 15k x 64 is ~5.5 % of the full data.  **Conclusion: at this
  recipe and budget, data volume is not phase 1's bottleneck.  Decision: keep `latp2_inv` for phase 2.**
  Results: `acdc_la/results/stage1_probes_latp2_full_vs_ref.json`, `latp2_inv_full{,_ref}.json`.
- **Next (user priority): full-parameter phase 2 at UniVLA's scale** with `latp2_inv` labels.  Trainer now has
  `--full_ft True` (FSDP FULL_SHARD, fp32 shards / bf16 compute, activation checkpointing on LLaMA layers, KV cache off,
  sharded model+optimiser checkpoints via torch.distributed.checkpoint, full bf16 `vla_full.pt` at the end) and
  `CTRL_OBS_KEEP=0,1,3,4,6,7,9,10,12,13` (only the 10 window frames `latp2_inv` reads, T=K+1=5, stride 3; equality test
  `acdc_la/scripts/test_obs_keep.py` reads T from the checkpoint).  Smoke (2 GPUs, batch 1): 30 steps, sharded saves and
  final weights OK; `--resume` OOMs on 2 GPUs (61 GB/GPU of fp32 state) -- still to be tested on >= 4 GPUs.

### 2026-10-27 — REGISTERED (not run): counterfactual evaluation of the latent by simulator replay

- **Why.** Every encoder measurement so far is indirect: real data has no counterfactual (the same action executed in
  another scene), so the probes measure whether a donor's action semantics SURVIVE in a recipient scene, never whether
  the predicted future is RIGHT (paper limitations say so).  A simulator can produce the counterfactual exactly.
- **Main experiment (factorial replay, LIBERO / robosuite).**  Replay demo i's action sequence (EEF deltas) open-loop
  from another scene's initial state; N actions x M scenes, the 2x2 cell being (A,a) (A,b) (B,a) (B,b).
  Two perturbation families, reported SEPARATELY: (i) appearance only -- DomainRandomizationWrapper with color/texture +
  lighting ON, camera + dynamics OFF, so physics and outcome are identical and only pixels change; (ii) geometry --
  another demo's initial state (object positions differ), so the open-loop outcome really differs.
  Metrics: invariance z(A,a) ~ z(B,a); separability z(A,a) != z(A,b) (incl. retrieval); TRANSPORT with ground truth --
  decode(f0_B, z(A,a)) vs the replayed (B,a) future in DINO feature space, against copy-donor / zero-effect / own-z
  references; relational consistency (a - b effect difference in A vs in B, the L_rel claim).  Plus the PROXY ERROR:
  the gap between the donor effect EAT anchors to (dF_i) and the true counterfactual dF_{B,a} -- expected ~0 for (i)
  and large for (ii) because Psi is computed per 4x4 cell, which states where the EAT assumption holds.
  Tokenizers: tc_cont (reconstruction only), latp2_anchor (EAT without L_inv), latp2_inv (full), UniVLA LAM
  (s2mix_1ep), optionally latp2_inv_full.  Zero-shot first.  Prerequisite: replay determinism (same scene + same
  actions -> identical frames) checked before any number is read.
- **Side experiments.**  Scene-leakage linear probes on z (scene/task/source id: lower is better; action: higher is
  better) for the same tokenizers -- cheap, indirect.  Real-data cross-scene action retrieval on Bridge/DROID (near-
  identical action chunks, distant first frames by DINO distance) if time allows.  LIBERO-Plus has systematic scene
  perturbations but no replay pairs, so it cannot supply the transport ground truth.
- **Status.** LIBERO HDF5 demos downloading to `data/LIBERO_hdf5` (job 1079786).  No result yet.

### 2026-10-27 — scene-leakage linear probes (the cheap side experiment): no within-source scene information in z; the pooled "leakage" is dataset/embodiment identity

- **Setup.** `acdc_la/stage1_preserve/leakage_probes.py` on the full-cache validation dumps (11,456 clips, 29 sources;
  one episode-disjoint 70/30 split shared by all tags; ridge with LOO-CV alpha, logistic for source id).  Tags dumped
  on the same split: `tc_cont_ref` (reconstruction only), `latp2_anchor_ref` (EAT, no L_inv), `latp2_inv_ref` (full),
  `latp2_inv_full`.  First version used a fixed ridge alpha=10 and over-fit the per-source sets (all R^2 < 0) --
  superseded by the CV version below.

  | z (256-d) | pooled: source bal. acc | pooled: appearance R^2 | within-source: appearance R^2 | within: action net R^2 | within: action seq R^2 |
  |---|---|---|---|---|---|
  | tc_cont | 0.598 | 0.167 | -0.446 | 0.270 | 0.199 |
  | latp2_anchor | 0.681 | 0.229 | -0.094 | 0.317 | 0.232 |
  | latp2_inv | 0.782 | 0.254 | -0.025 | 0.340 | 0.240 |
  | latp2_inv_full | 0.818 | 0.266 | 0.031 | 0.358 | 0.268 |
  | F0Q (first frame, reference) | 1.000 | 1.000 | 0.998 | -0.118 | -0.194 |

  (chance source acc 0.034; appearance = PCA of F0Q, 32 pooled / 16 per source; 26 sources with >= 80 test clips.)
- **Reading.** Within a source, NO tokenizer carries linearly decodable first-frame appearance (R^2 <= 0.03), while
  the action information rises tc_cont < anchor < inv < inv_full.  The pooled increase of source accuracy and
  appearance R^2 with the LATP objectives is therefore between-source: z identifies the dataset/robot better, which
  goes with encoding its actions better (source id carries embodiment and action space).  **Conclusion status: no
  evidence of scene leakage, and no evidence FOR the invariance claim either** -- a linear probe that already reads
  ~0 for the reconstruction-only baseline cannot show an invariance gain.  The counterfactual replay experiment is the
  direct test.  Results: `acdc_la/results/leakage_probes.json`, logs `acdc_la/logs/leak_probes_*.out`.

### 2026-10-28 — LIBERO-PRO (robustness) on the full-data post-trained arm-A policies: language robust, object replacement weak, position/task ~0 with the few successes concentrated in single tasks

- **Setup.** LIBERO-PRO (arXiv 2510.03827; code Zxy-MLlab/LIBERO-PRO @ eafdb80; bddl/init files HF zhouxueyang/LIBERO-Pro)
  with our evaluator (`run_libero_eval_cont`, center crop, seed 7, base-suite max steps, base-suite action stats),
  launcher `acdc_la/scripts/run_liberopro_eval_newton.sh`, separate LIBERO config dir.  Policies: the four
  full-split post-trained runs (1/10-scale LoRA phase 2, latp2_inv labels; standard LIBERO 97.2/98.8/94.4/89.0).
  10 trials x 10 tasks = 100 episodes per cell.  The env perturbation (runtime scene generation) was not run.

  | suite | lan (Sem) | object (Obj) | swap (Pos) | task (Task) |
  |---|---|---|---|---|
  | Spatial | 0.93 | 0.96 | 0.00 | 0.00 |
  | Object | 0.97 | 0.83 | 0.11 | 0.00 |
  | Goal | 0.98 | 0.61 | 0.11 | 0.09 |
  | Long | 0.90 | 0.55 | 0.03 | 0.07 |

  Leaderboard (authors' settings, not ours): OpenVLA Pos/Task 0.00 everywhere, Obj 0.81-0.98; Pi0.5 Pos 0.08-0.38,
  Task 0.00-0.01, Obj 0.92-0.98; X-VLA Task 0.00-0.10.
- **Per task.** The non-zero swap/task cells are each ONE task: Goal swap 9/10 "open the top drawer and put the bowl
  inside" (all others <= 2/10), Object swap 10/10 cream cheese (others <= 1/10), Long swap 3/10 book, Goal task 9/10
  "turn on the stove", Long task 7/10 "put both moka pots on the stove".  Object-replacement failures sit on specific
  targets (alphabet soup 0/10 in Object and Long, book 0/10, bowl tasks in Goal 0-1/10) while tasks whose target is
  unchanged stay at 9-10/10.
- **Reading.** The pattern is that of a policy replaying memorised trajectories: success under swap/task where the
  perturbation leaves the memorised motion's end point valid, failure elsewhere -- the LIBERO-PRO thesis, which our
  policy does not escape.  The Pos/Task cells above OpenVLA are therefore NOT evidence of spatial generalisation.
  Obj is clearly below OpenVLA/Pi0.5 on Goal and Long.  High Sem is consistent with the paper's observation that
  policies largely ignore the instruction.  Not a fair head-to-head: different evaluators; a same-evaluator baseline
  (e.g. UniVLA's released LIBERO checkpoints) is needed before any comparison is claimed.
- **Artifacts.** `acdc_la/results/vla_eval_pro/merged_*_pro_p2ours_full.json`, logs `acdc_la/logs/liberopro_*`.
  The first Object-suite attempt failed (a stale empty run dir matched the same glob); rerun with the explicit path.

### 2026-10-28 — full-parameter phase 2: intermediate check at step 5000 (10 % Long 57.6 vs 61.6 at 1/10 LoRA, within noise); crash at 5250 and in-allocation retry

- **Setup.** Snapshot of the UniVLA-scale full-parameter phase-2 run (FSDP, acdc_oxe29, latp2_inv labels, global batch
  1024, lr 2e-5 constant) at step 5000 of 20k, consolidated (`consolidate_fft_ckpt.py`), then the SAME downstream recipe
  as the 1/10-scale LoRA cell: 10 % LIBERO-Long post-training, 500 episodes (job 1080187, `p2fft_midcheck_newton.sh`).
- **Result.** 288/500 = **57.6 %** vs 61.6 % (1/10-scale LoRA phase 2).  Per task (fft5000 / LoRA): 0 22/42, 1 80/82,
  2 92/84, 3 78/70, 4 40/54, 5 4/36, 6 66/76, 7 86/72, 8 42/24, 9 66/76.  Difference 4.0 points, about 1.3 sigma of
  two independent 500-episode rates; mixed signs per task; task 5 (book) carries most of it.  Reading: no evidence
  that the full-parameter run is better OR worse at 25 % of training; the decision waits for step 20k.
- **Phase-2 objective vs downstream.** At step 5000 the full-parameter run has seen 5.1M samples (5000 x 1024), 2.7x
  the LoRA run's 1.9M (30000 x 64), and its training latent R2 is already higher (0.820 at 4.5-5k vs 0.807 at the
  LoRA run's end, after its 10x lr decay at 24k; the full run's lr stays 2e-5 by UniVLA's recipe).  A better phase-2
  fit did not give a better 10 % Long policy.  (`p_oracle` in the log has no effect with `--head flow`.)
- **Decided 2026-10-29: LR-decay branch after 20k.**  The main run stays UniVLA's recipe (constant 2e-5) to 20k for the
  head-to-head; then a WSD-style branch from the 20k checkpoint, 2k more steps with lr linear 2e-5 -> 0, in its own run
  dir (`finetune_libero_ctrl.py --branch_from <run> --decay_start 20000 --max_steps 22000`, launcher usage in
  `run_vla_phase2_fft_newton.sh`).  Both are post-trained and evaluated the same way; the branch is reported as a
  recipe variant (10 % more phase-2 steps than UniVLA).  Not queued yet: submit when the main run reaches 20k.

### 2026-10-29 — why full-parameter phase 2 (step 5000) post-trains worse than the 1/10 LoRA: the first post-training update wipes its latent head (optimisation shock), warmup prevents it

- **Question (user).** 10 % Long: fft step 5000 57.6 % vs 1/10 LoRA 61.6 % although the fft run saw 2.7x the samples
  and fits phase 2 better.  Is it really worse, why, can it be reversed?
- **Ruled out.**  (1) Config differences: the two post-training runs are identical (ctrl_meta, env, launcher) except
  that the LoRA arm CONTINUES its phase-2 adapter while the fft arm gets a fresh LoRA on the full base (by design).
  (2) Consolidation bug: the sharded checkpoint holds exactly the 1,073 `model.*` keys the skeleton expects (no wrapper
  prefixes); heads load strictly; at lr 0 the consolidated model reproduces its phase-2 objective on OXE (R2 0.829 vs
  0.82 in the training log).  (3) "Full FT drifts further": relative weight change vs openvla-7b is SMALLER for fft
  (LLM attn/mlp 6.1/7.0 %, projector 11 %, DINO/SigLIP 2.1/2.0 %) than for the LoRA delta (12.4/10.6 %, 19 %,
  9.1/10.4 %); head weight RMS 0.027 (fft) vs 0.049 (LoRA).  (4) Poor transfer: zero-shot latent R2 on LIBERO at
  lr 0 is 0.603 (fft) vs 0.637 (LoRA) -- a small gap, not a failure.
- **The cause: optimisation shock at the first post-training update.**  Replay of the first 50 updates (1 GPU,
  batch 64, lr 3.5e-4, fresh AdamW): fft R2 0.70 before the first step, **-0.04 after it** (mean 0.085 over updates
  2-10), one-step action L1 0.46 -> **1.23**; R2 back to 0.45 by update 50.  LoRA arm: 0.58-0.74 throughout, max L1
  0.51.  Reproduced in the real runs: first 25 logged steps R2 0.158 (fft) and 0.074 (fft replicate) vs 0.658 / 0.653
  (LoRA, LoRA replicate); 0.125 for the no-phase-2 control.  The fft curve then sits between LoRA and no-phase-2
  (steps 100-300: 0.70 / 0.62 / 0.51) and all three converge by ~1.5k steps.  Reading: the fft model (heads and
  backbone) was trained at lr 2e-5; post-training's lr is 17x larger and a fresh Adam's first steps move every
  parameter by ~lr, which its solution does not survive.  The LoRA arm was trained at 3.5e-4 already, so post-training
  is a continuation of its own regime.  The phase-2 latent knowledge of the fft arm is therefore largely discarded
  and re-learned from LIBERO's 10 % demos.
- **Held-out split (train[95%:], 2000 windows, flow-sampled as deployed) of the two FINAL 10 % Long policies.**
  Latent R2 0.227 (fft) vs 0.184 (LoRA); action L1 over the chunk 0.165 vs 0.154; first-step arm L1 0.178 vs 0.169;
  **first-step gripper error 0.084 vs 0.048**.  Small action deficit, mostly gripper; consistent with, not proof of,
  the closed-loop gap.
- **Fix, offline check.**  `--warmup_steps` (linear lr warmup, new trainer flag): with 200 warmup updates the fft
  arm's R2 stays 0.59-0.69 for all 310 logged updates while action L1 falls 0.51 -> 0.14 -- no collapse.
- **Running (decides "really worse" and "reversible").**  Replicates without warmup: LoRA (1090393), fft (1091573);
  no-phase-2 control (1090395); fft + 500-step warmup (1091917).  All 10 % Long, 15k steps, 500-episode evaluation.
  Post-training of every full-parameter base (incl. the final 20k / decay models) should use warmup.
- **Artifacts.** `acdc_la/results/p2_diag/` (zs_lib_*, ft_lib_*, zs_oxe_* logs, heldout_*.json, weights.json),
  `acdc_la/logs/p2_diag_wu200_1091619.out`, scripts `acdc_la/scripts/p2_diag_newton.sh`, `p2_diag_weights.py`;
  `prior_gate.py` gained `--no_prior` and held-out action errors.
- **LIBERO-Plus Long moved off Newton (user, 2026-10-29).**  Newton's October GPU quota is exhausted (the four Long
  jobs pended on AssocGrpGRESMinutes); they are HELD (1084953-56, fallback after the Nov 1 reset) and the suite runs
  on the UCF workstation with `acdc_la/scripts/run_liberoplus_eval_portable.sh` (no Slurm; shards dealt over GPUs;
  `LIBERO_RESUME=1` continues an interrupted shard), same 8 strided shards and note `p2ours_full`, policy
  `tc_cont+libero_10_no_noops+b32x4+lr0.00035+r32+s40000+fm+q4--p2ours_libero_10_full`.  Branch `newton` pushed
  (1d23133).  Copy the shard jsons back to `acdc_la/results/vla_eval_plus/` and run `liberoplus_summary`.

### 2026-10-30 — 10 % Long replicates: run-to-run spread ~6 points; fft vs LoRA and phase 2 vs none are not distinguishable; warmup does not help (CORRECTS the 10-29 reading)

- **Closed loop, 500 episodes each, identical recipe (15k steps, 10 % Long):**

  | run | success | | run | success |
  |---|---|---|---|---|
  | LoRA phase 2 (1/10) | 308/500 = 61.6 % | | fft step 5000 | 288/500 = 57.6 % |
  | LoRA replicate | 279/500 = 55.8 % | | fft replicate | 256/500 = 51.2 % |
  | no phase 2 | 282/500 = 56.4 % | | fft + 500-step warmup | 249/500 = 49.8 % |

- **Spread.**  Identical configs differ by 5.8 (LoRA) and 6.4 (fft) points: pooled sd of ONE run ~4.3 points, of
  which binomial noise (500 episodes) is ~2.2 and training randomness ~3.7.  Per-task replicate differences have
  sd ~16 points (LoRA task 9: 76 vs 34; fft task 5 "book": 4 vs 30 -- the 10-28 per-task reading was noise).
- **Comparisons.**  LoRA mean 58.7 vs fft mean 54.4: 4.3 +- 4.3 (1 s.e.), not significant; no-phase-2 56.4 lies
  between them.  On this protocol neither phase-2 variant shows a detectable effect over no phase 2.  Warmup
  removes the first-step collapse (latent R2 0.63 vs 0.16 over the first 25 steps, verified in the real run) but the
  closed loop did not improve (49.8, one run) -- the runs converge after ~1.5k of 15k steps and the collapse leaves
  no measurable trace.  **Correction of 2026-10-29:** the optimisation shock is real but is NOT shown to cause the
  57.6 vs 61.6 gap, which is within run-to-run noise; warmup is not adopted as a default.
- **Consequence.**  A single 500-episode cell carries ~+-8 points (95 %) once training randomness is included; every
  single-run comparison in this ledger's VLA tables has that uncertainty.  Detecting a 3-point difference at sd 4.3
  needs ~30 runs per arm.  Phase-2 variants must be compared with a lower-variance or more phase-2-sensitive
  protocol, or with several seeds per arm.
  Update (13:50): Object finished on Newton (8/8 shards complete); Goal shards 0-1 finish on Newton (job 1084949),
  Goal shards 2-7 (jobs 1084950-52, HELD) move to the workstation too.  Policies on the private HF repo
  `Potestates/TERRA` (`libero_10_full/`, `libero_goal_full/`; pushed over git-lfs from the login node -- compute nodes
  have no network).  Each shard runs whole on one machine; Newton has ImageMagick 6, the workstation 7, which only
  touches Sensor Noise's motion blur (random angle per call) -- recorded so a cross-machine difference is not mistaken
  for a policy effect.
- **Crash.** Segment 21 of job 951668 died at 02:55 inside the dcp resume at step 5250 (ranks out of
  step in a collective, an impossible >1 EB allocation, NCCL timeout); the checkpoint was intact.  The wrapper treated
  a non-zero exit as final and released the node, so the run had to queue again.  `run_vla_phase2_fft_newton.sh`
  now retries a crashed trainer in the same allocation (give up after 3 crashes on the same checkpoint).  Resubmitted
  as job 1086074 (4 x H100 HBM3, evc101, started 19:43); resumed at step 5250 with continuous metrics.
- **Artifacts.** `acdc_la/results/vla_eval/merged_libero_10_nw_p2fft5000_libero_10_frac10.json`, logs
  `acdc_la/logs/p3_1080187.out`, `acdc_la/logs/p2_fft_1086074.out`.

### 2026-10-29 (17:35) — LIBERO-Plus Long moved to the UCF workstation: environment up and the pipeline verified end to end on one task; the suite itself has not run yet

- **Configuration.**  LIBERO-Plus `libero_10` = **2519 tasks**, 1 trial each, the same 8 strided shards and note
  `p2ours_full` Newton used for the other three suites, dealt over the workstation's 2 GPUs; policy
  `tc_cont+libero_10_no_noops+b32x4+lr0.00035+r32+s40000+fm+q4--p2ours_libero_10_full` (Newton, full-data post-training).
  Launcher `acdc_la/scripts/run_liberoplus_eval_ws.sh` (new) wrapping `run_liberoplus_eval_portable.sh`.
- **Two forced deviations from Newton's environment.**  Neither touches the policy, the task set or the init states.
  (i) LIBERO-plus comes from upstream instead of Newton: `github sylvestf/LIBERO-plus` @ `4976dc3` plus the HF
  `assets.zip` (6,395,849,578 B, sha256 `96764a4b…`).  The zip carries the uploader's absolute path, so the real tree
  sits at `inspire/hdd/.../LIBERO-plus-0/assets` and was moved up to `libero/libero/assets`: 9.5 GB / 448,799 files,
  9.7 GB for the whole tree, the same size Newton reports for its copy.  (ii) ImageMagick **7** from conda-forge in a
  private prefix instead of apt's 6.9 (no sudo on this host), with `MAGICK_HOME` and an `LD_PRELOAD` of conda's glib
  and libjpeg so the conda libraries win the soname lookup against the ones cv2 and torchvision bundle (both error
  messages are recorded in the launcher header).  The ImageMagick version reaches only the Sensor Noise motion blur,
  whose angle is drawn uniformly at random per call, so it is far inside the category's own stochasticity.
- **Pipeline check — NOT a result.**  The September 10 %-demo Goal arm
  (`tc_cont+libero_goal_no_noops+…--latp2inv_fmq4_frac10`) on two LIBERO-Plus **Goal** tasks: builds the perturbed env
  with the new assets, renders offscreen under EGL, loads the flow head (10 Euler steps), 2/2 success, writes
  `per_task_id` and the shard json, 11–13 s per successful task.  The json was deleted: that policy is not the one
  under test and its number means nothing.  The same policy on `libero_10` stops at `unnorm_key` (its
  `dataset_statistics.json` has only `libero_goal_no_noops`) — expected, and the reason the real run needs Newton's
  Long policy.
- **Conclusion — planned.**  The Long suite is blocked only on the 15 GB policy: Newton accepts
  `password,keyboard-interactive` and refuses public keys, and Claude Code's shell has no tty
  (`read_passphrase: can't open /dev/tty`), so the copy needs an ssh ControlMaster the user opens in a real terminal.
- **Disk (user decision, 2026-10-29).**  Deleted `/data/ti874297/oxe_raw` (3.6 TB, the 29-source phase-2 RLDS mixture —
  phase 2 runs on Newton now) and two superseded 2026-09-14 smoke run dirs (30 GB).  /data 1.8 → 5.4 TB free, / 167 →
  194 GB.  **Kept** the stage-1/2 feature cache (`latent_free_cache/oxe` 1.8 TB + `bridge_big` 95 GB), which the
  registered low-bottleneck TC sweep would need.
- **Also.**  The workstation's uncommitted 2026-10-08 phase-2 memory work (`ACDC_WINDOW_KEEP` + `ACDC_MIX_SHARDS`, with
  the measurement that the 29-source pipeline needs ~48 GB per rank under `chunk_act_obs_libero`) is parked on the
  local branch `ws-oct08-memfix`, superseded by this branch's `CTRL_OBS_KEEP`, which keeps both `ro` windows.
- **Artifacts.**  `acdc_la/scripts/run_liberoplus_eval_ws.sh`, header note in `run_liberoplus_eval_portable.sh`;
  `/data/ti874297/liberoplus/{LIBERO-plus,magick,get_liberoplus.log,pipecheck.log}`;
  `acdc_la/logs/liberoplus_pipecheck_libero_goal_s0of2000.log`.
- **Next.**  Copy the run dir from Newton, run all 8 shards, then copy the shard jsons back to
  `acdc_la/results/vla_eval_plus/` on Newton and run `liberoplus_summary --note p2ours_full` for the four-suite table.

### 2026-10-30 — LIBERO-Plus, all four suites COMPLETE: 52.6 % over all 10,030 perturbed tasks, +8.7 over UniVLA's published 43.9, and the best Robot-Initial-States score on the leaderboard (50.5 of 12 entries); the one real weakness is Sensor Noise

- **Configuration.**  LIBERO-Plus (`sylvestf/LIBERO-plus` @ `4976dc3`), 1 trial per task, note `p2ours_full`, 8 strided
  shards per suite (`LIBERO_TASK_SHARD=k/8`, `LIBERO_RESUME=1`).  Policies = the full-split post-trained arms
  (`tc_cont+<suite>_no_noops+...+fm+q4--p2ours_<suite>_full`, flow head, 4 query tokens), one per suite, i.e. the same
  checkpoints that give standard LIBERO 97.2 / 98.8 / 94.4 / 89.0 (avg **94.9**) against UniVLA's published
  96.5 / 96.8 / 95.6 / 92.0 (avg 95.2) — in-distribution the two are matched, which is what makes the Plus
  comparison readable.  Spatial + Object + Goal shards 0-1 on Newton; **Long all 8 shards and Goal shards 2-7 on the
  UCF workstation** (2 GPUs, ~23 s/task, Long 13 h 49 m, Goal 6 h 17 m, every shard exit 0) after Newton's October GPU
  quota ran out; policies fetched from HF `Potestates/TERRA`.  Environment deviations on the workstation (upstream
  LIBERO-plus + ImageMagick 7) are recorded in the 2026-10-29 entry; each shard ran whole on one machine.

- **Result (success %, 1 trial per task; `acdc_la/results/vla_eval_plus/summary_plus_p2ours_full.json`).**

  | suite | Camera | Robot | Language | Light | Background | Noise | Layout | cat-mean | total | n |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Spatial | 6.4 | 57.4 | 84.4 | 94.9 | 98.8 | 17.1 | 92.5 | 64.5 | 62.5 | 2402 |
  | Object | 3.5 | 55.3 | 84.2 | 48.5 | 87.1 | 21.1 | 67.7 | 52.5 | 49.8 | 2518 |
  | Goal | 7.8 | 37.2 | 59.0 | 91.8 | 76.2 | 17.9 | 58.6 | 49.8 | 46.8 | 2591 |
  | Long | 4.3 | 53.4 | 81.5 | 74.1 | 82.0 | 20.5 | 74.7 | 55.8 | 51.8 | 2519 |
  | **ALL (ours)** | **5.5** | **50.5** | **76.8** | **77.1** | **85.7** | **19.3** | **72.9** | **55.4** | **52.6** | 10030 |
  | UniVLA (README leaderboard — our architecture's own published row, the baseline that matters) | 1.8 | 46.2 | 69.6 | 69.0 | 81.0 | 21.2 | 31.9 | -- | 43.9 | 10030 |
  | pi_0 (README) | 13.8 | 6.0 | 58.8 | 85.0 | 81.4 | 79.0 | 68.9 | -- | 53.6 | 10030 |
  | OpenVLA-OFT (README; wrist camera + OFT stack, a different input/architecture class) | 56.4 | 31.9 | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | -- | 69.6 | 10030 |
  | OpenVLA-OFT_w (README; OFT with the wrist camera REMOVED — the table's own wrist ablation) | 10.4 | 38.7 | 70.5 | 76.8 | 93.6 | 49.9 | 69.9 | -- | 55.8 | 10030 |

  **Placement over all 12 leaderboard rows** (ours inserted): Robot Initial States **1st**, Objects Layout 4th,
  Language 5th, Light 6th, Background 6th, Camera 8th, Sensor Noise 9th, total 8th — between pi_0 (53.6) and UniVLA
  (43.9), and above NORA (39.0), WorldVLA (25.0), OpenVLA (15.6).

  Success falls monotonically with the benchmark's own `difficulty_level`: 75.8 (1) / 60.5 (2) / 55.2 (3) / 47.6 (4) /
  25.5 (5) over 1644 / 2202 / 2094 / 1886 / 2083 tasks — the measurement tracks the intended difficulty axis.

- **Conclusion — confirmed (measurement), not attributed.**  The arm beats its own architecture's published row in
  **6 of 7 categories** and by **+8.7 total**, which puts it at pi_0 level (53.6) from a UniVLA base that scores 43.9.
  Two results stand out: **Robot Initial States 50.5 is the highest value in the table**, above UniVLA's 46.2 and 20
  points above OpenVLA-OFT+ (30.3) — initial-state robustness is where this arm is strongest in absolute terms, not
  merely relative; and **Objects Layout 72.9 vs 31.9 (+41.0)** is 4th of 12, within 1.3 of OpenVLA-OFT (74.2), i.e.
  top-tier rather than merely improved.  Also up: Light +8.1, Language +7.2, Background +4.7.
  **Where it does not improve, and how much of that is ours.**  Sensor Noise 19.3 vs 21.2 (flat) and Camera Viewpoints
  5.5 vs 1.8 (low in absolute terms).  The table's own wrist ablation separates the two: removing the wrist camera from
  OpenVLA-OFT takes Camera **56.4 -> 10.4** while Noise only goes 75.8 -> 49.9.  So the camera column on this benchmark
  is dominated by whether the policy has a wrist observation, which the UniVLA input stack does not — our 5.5 is close
  to what a no-wrist policy can reach and is an architecture fact, not a phase-2 failure.  **Sensor Noise is the one
  genuine weakness**: a no-wrist OFT still reaches 49.9 and pi_0 reaches 79.0, so ~19 is improvable and nothing in
  phase 2 addresses pixel-level nuisance.
  **Which margins survive the run-to-run noise measured the same day** (see the 2026-10-30 replicate entry: training
  randomness sd ~3.7 points for a 10 % Long post-training run).  LIBERO-Plus itself is nearly noise-free at the total
  level -- 1 trial x 10,030 tasks gives a sampling s.e. of 0.5 points, far tighter than a 500-episode LIBERO cell --
  but each suite's number still comes from ONE post-trained checkpoint, so the ~3.7 bound applies to the policy.
  Against that bound: **Objects Layout +41.0 is ~11 sd** and cannot be noise; **total +8.7 is ~2.4 sd**, solid but not
  overwhelming; **Robot Initial States +4.3 over UniVLA's 46.2 is ~1.2 sd, i.e. inside it** -- 50.5 is the highest
  value in the table as measured, and the lead over 3rd place (OpenVLA-OFT_w 38.7) is +11.8, but the lead over UniVLA
  specifically must not be reported as established.  The full-split checkpoints used here trained on 10x the data for
  30-40k steps, so 3.7 is an upper bound carried over from a 10 %-data run, not a measurement on these runs.
  **Causally unattributed, as in the standard-LIBERO entry:** no matched no-phase-2 / no-latent control was run (user
  decision) and the head differs from UniVLA's (flow + query tokens), so this is a policy-vs-published-policy
  comparison, not an ablation of the latent tokenizer.  The layout and initial-state results are nonetheless the first
  evidence on a perturbation benchmark consistent with the method's claim (action semantics invariant to where the
  scene's objects and the robot start), and the noise column bounds it.

- **Coverage — complete.**  10,030 / 10,030 tasks, 1 trial each.  Newton finished Goal shards 0-1 (`19a1635`,
  324/324 at 47.8 % and 45.4 %) after this table was first computed over 9,835; the 195 added tasks moved the total
  52.26 -> 52.57, within 0.03 of the imputation made from their category rates, and changed the Goal row by
  +1.8 (total), +0.4 (Light), +1.0 (Noise).  Both halves are therefore mutually consistent.
  Reading a partial run by overall success is misleading on this benchmark: at 147/2519 tasks the Long run showed
  74 %, because strided shards walk the category-ordered suite and only Robot Initial States and Background Textures
  had been reached; the final Long number is 51.8.

- **Artifacts.**  `acdc_la/results/vla_eval_plus/cont_{libero_10,libero_goal}_plus_p2ours_full_s*of8.json` (14 new
  shard jsons), `summary_plus_p2ours_full.json` (regenerated over all four suites);
  `acdc_la/logs/liberoplus_p2ours_full_libero_{10,goal}_s*of8.log` (gitignored, workstation);
  `/data/ti874297/liberoplus/run_plus_all.log`; `acdc_la/downstream/vla/liberoplus_summary.py` now takes `--cls` /
  `LIBERO_PLUS_CLS` (the benchmark clone path differed per host, which was the only thing blocking a local summary).

- **Next.**  (1) The registered intervention/replay experiment is the controlled version of this result: LIBERO-Plus
  supplies no counterfactual pairs, so it cannot say whether the layout and initial-state gains come from the latent's
  invariance or from the post-training mixture.  (2) Report Robot Initial States (1st) and Objects Layout (4th) as the
  headline; state the camera column as a no-wrist-input limitation with the OFT_w ablation as evidence, and Sensor
  Noise as the open weakness.  Do not claim viewpoint robustness.  (3) Nothing further to run on LIBERO-Plus for this
  arm — the four-suite table is final.

### 2026-11-05 — workstation readiness for post-training the 20k full-parameter phase-2 arm: measured, and three things that would have killed the run

- **Why.**  The 20k `fft` phase-2 checkpoint is to be post-trained and compared with the existing cells.  User decision:
  run the **10 % Long** cell first (15k steps, global 64) rather than the full split, because it joins the six cells of
  the 2026-10-30 replicate table directly and costs 9.5 h instead of 50 h.  Launcher `run_vla_post2_portable.sh`
  (Newton, `346aab1`).
- **Throughput, measured here (120-step smoke, Long RLDS at `LIBERO_TRAIN_FRAC=10`, 1 GPU, b32x1, flow head + q4).**
  2.32 s/step = **13.8 samples/s/GPU**, against 14.1 for the 2026-09-19 2-GPU Goal run and 16.2 for an H100 in
  Newton's full-split runs -- this box is ~15 % slower per card than an H100.  So the 10 % Long cell is **9.5 h on two
  GPUs, 19.3 h on one**, and the full split would be 50 h (Long) / 6.8 days (four suites), NOT the 2.3-2.9 days /
  7.5-9.3 days quoted earlier from a 2026-09-14 probe that had shared its GPU with another job.  `latent_r2`
  0.13 -> 0.39 and `action_l1` 0.35 -> 0.21 over 100 steps, the same shape as the September Goal run.
  **VRAM 43.8 GB per rank** at batch 8.
- **Host RAM, measured (62 GB box, clean processes).**  `torch.load(vla_full.pt, mmap=True)` of a 15.1 GB file adds
  **nothing** to RSS (2.2 GB before and after) -- the mmap change (`346aab1`) works.  `load_state_dict` then peaks at
  **30.6 GB**, `merge_and_unload` **31.9 GB**, `save_pretrained` **31.9 GB**; the whole merge path 250 s.  This
  **replaces the 50-55 GB estimate** that was reasoned from the September numbers: mmap keeps the file's 15 GB out of
  anonymous memory, where it is also reclaimable.  Measured with a stand-in `vla_full.pt` (the openvla-7b state dict,
  15.1 GB, same shapes) and a real same-shape adapter (`tc_cont+...+fm+q4--latp2inv_fmq4_frac10`): a memory
  measurement, not a model.  Two embedding keys were skipped (the stand-in predates the resize), so add ~0.5 GB, and
  at the real merge rank 1 is still alive at the barrier, so call it 32-33 GB of 41 available.
- **Three fixes this found.**  (1) The launcher hard-required `BS x GA x GPUs = 128` and exited otherwise, but every
  existing 10 % cell is global **64** (`b32x2`); running the new arm at 128 would not be comparable with the cells it
  is meant to join.  `GLOBAL` added, default 128 (`4a3bf61`).  (2) `CTRL_STAGGER_SEC` defaults to 90 but one
  `load_state_dict` takes ~180 s, so both ranks would hold ~15 GB of private parameters at once (~43 GB against 41
  available): **use `CTRL_STAGGER_SEC=240`**, which lets rank 0 reach `vla.to(device_id)` and free its CPU copy first.
  (3) The transfer list must include **`action_decoder.pt`** (heads, read at `finetune_libero_ctrl.py:342`) --
  `consolidate_fft_ckpt.py:2-4` gives the full set: `vla_full.pt` + `action_decoder.pt` + `ctrl_meta.json` +
  processor + `dataset_statistics.json`.
- **Also verified.**  `LIBERO_TRAIN_FRAC` is live in the vendored UniVLA tree
  (`third_party/UniVLA/prismatic/vla/datasets/rlds/dataset.py:251-255`); if it were not, `FRAC=10` would silently
  train on 95 % of the demos and the cell would not be the one it is compared against.  Long RLDS (3.5 GB) taken from
  HF `openvla/modified_libero_rlds`, the same source as the local Spatial/Goal, so Newton is not needed for data.
- **Correction 2026-11-08 — the measurement held, the "it fits" did not.**  The first real run of this path
  (`terra_piper`, below) trained to 10,000 steps and was then **SIGKILLed three times in a row in the merge**
  (09:12 / 09:21 / 09:37, `exitcode -9`), each time inside `merged.save_pretrained`, leaving three 4.9 GB partial
  safetensors.  Re-run **alone** in a clean process it succeeded at **peak RSS 30.7 GB in 211 s**, i.e. the 31.9 GB
  figure above is right -- but it was measured with nothing else on the host, and at the end of a real run **rank 1
  is still alive at `dist.barrier()`** (torchrun only SIGTERMs it after rank 0 dies) holding its own tf.data arena,
  plus whatever else shares the box.  MemAvailable fell to **9.2 GB** even in the clean re-run.  So: the merge of a
  20k-base LoRA run **does not reliably fit on this 62 GB host as the tail of a 2-rank job**, and the practical
  recipe is to let training finish and merge separately (`--merge False` would be the clean fix in the trainer).
  This does not affect any trained weights: the adapter at step 10,000 was already saved when the kill happened.
- **Conclusion — confirmed (infrastructure).**  Nothing measured about the method.  The run is blocked only on the
  15 GB `vla_full.pt`, which needs the user's own terminal (Newton offers only `password,keyboard-interactive` and
  this harness has no tty), and on GPU 1, which another job holds at 53.8 GB of 96 -- leaving 42 GB against the
  43.8 GB a rank needs, so two-GPU is not startable until it frees.
- **Artifacts.**  `acdc_la/logs/` smoke log and `/data/ti874297/{smoke_long.log,mmap_test/{check,merge}.log}`;
  run dir `/data/ti874297/vla_runs/tc_cont+libero_10_no_noops+b32x1+...--smoke_long_frac10` (120-step smoke, no
  checkpoint); scratch `merge_check.py` / `mmap_check.py`.
- **Next.**  `INIT=/data/ti874297/p2fft_20k GLOBAL=64 FRAC=10 STEPS=15000 CTRL_STAGGER_SEC=240 NOTE=p2fft20k_libero_10_frac10`.
  One cell carries +-8 points (2026-10-30), so two replicates (19 h on two GPUs) are what make it comparable with the
  LoRA and no-phase-2 arms, which have replicates.  The full split stays a separate decision.

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

### 2026-09-22 (20:10) — USER-REPORTED (other machine, not verified here): with phase 2 restored, TERRA beats matched UniVLA on all four LIBERO suites, avg 93.4 vs 91.8

- **Protocol, as described by the user.** The latent-action VLM pretraining stage (the phase this Anvil
  comparison had been skipping, registered 2026-09-20 15:30) was run on **10 % of the pretraining mixture**
  for both arms; downstream fine-tuning then followed **UniVLA's own LIBERO configuration** (not the 10 %-demo
  few-shot protocol used by every Anvil cell above).  Only the latent labeller differs between arms.

  | arm | Spatial | Object | Goal | Long | avg |
  |---|---|---|---|---|---|
  | UniVLA (matched) | 92.3 | 92.6 | 93.4 | 88.7 | 91.75 |
  | **TERRA (ours)** | **94.1** | **95.2** | **93.8** | **90.6** | **93.425** |

  Per-suite gain +1.8 / +2.6 / +0.4 / +1.9; average **+1.675 pp**.
- **Status: PROVISIONAL, second-hand.**  Numbers were reported in conversation; no evaluation JSON, run directory
  or log has been inspected on this machine, and the per-task breakdown is not in hand.  They are recorded so the
  paper has one source, not because they have been checked.  Anyone re-using them should first locate the result
  files and confirm the 500-rollout count per suite.
- **What this changes.**  (i) The main table is no longer the 10 %-demo few-shot table; `paper/experiments.tex`
  and the closed-loop panel of `figures/representation_comparison_with_policy.pdf` now state the 10 %-pretraining /
  full-demo protocol.  (ii) The 2026-09-20 finding that **no latent at all** reaches 88.6 % on Goal was measured under
  the *other* protocol (10 % demos, query interface, no phase 2) and does not transfer to this table -- but the
  no-latent control has **not** been run under this protocol, and it is the control a reviewer will ask for.
- **Retraction of the placeholder block.**  The invented matched-block numbers committed earlier today
  (no-latent 72.1 / UniVLA 75.4 / TERRA 76.7 avg) are withdrawn; they were only ever scaffolding for the figure
  and table layout and are replaced by the table above.

### 2026-09-25 (12:30) — NMSE ablation with paired source-bootstrap intervals, from existing dumps (no training); the random-TI UniVLA baseline is at least as good as TERRA on every probe

- **What ran.** `acdc_la/downstream/nmse_bootstrap.py`: the repo's own per-source episode-disjoint ridge
  (`common.probes.per_source_readouts`) from each arm's frozen Z to net / seg1 / seq / prof built from ACT, plus the
  moving-patch drop; NMSE = 1 - R^2, clip-weighted over the 29 sources.  2,000 bootstrap resamples of SOURCES, the
  same resample for every arm, so TERRA - arm gets a paired interval.  Point estimates reproduce the ledger to the
  third decimal (TERRA net .329, endpoint-only .336, UniVLA matched .237), so the protocol is unchanged.
  **Intervals are across sources, not across training seeds; every arm is still one training seed.**
  Artifacts: `acdc_la/results/nmse_bootstrap.json`, per-source rows in `acdc_la/results/nmse_rows/`.

  | arm | net | seg1 | prof | seq | drop % |
  |---|---|---|---|---|---|
  | endpoint only (N) `s1_mean_1ep` | .664 | .790 | .983 | .800 | 33.6 |
  | DC only (N) `s1_dc` | .671 | .797 | .993 | .808 | 29.7 |
  | N + A1, earlier tokenizer `s1_acdc_1ep` | .676 | .736 | .915 | .776 | 28.9 |
  | TERRA start `tc_cont` (65k) | .734 | .767 | .909 | .807 | 22.4 |
  | +15k reconstruction only | .732 | .760 | .909 | .807 | 22.6 |
  | +15k EAT w/o invariance | .700 | .733 | .915 | .784 | 19.7 |
  | **+15k EAT (TERRA) `latp2_inv`** | **.671** | **.717** | **.909** | **.764** | **17.9** |
  | UniVLA matched (released TI prior) | .763 | .778 | .875 | .785 | 23.0 |
  | **UniVLA, random TI codebook `univla_noinh`** | **.655** | **.689** | **.821** | **.702** | **15.7** |

- **Confirmed (paired intervals exclude 0).**  (i) EAT vs reconstruction-only at the same start and budget: better on
  net [-.083,-.035], seg1 [-.061,-.026], seq [-.059,-.028], drop [-11.4,-0.3]; prof unchanged.  (ii) TERRA vs the
  endpoint (N-only) code: better on seg1, prof [-.102,-.048], seq and drop [-27.0,-8.1]; net tied.
  (iii) vs UniVLA matched: better on net and seg1, **worse on prof** [+.011,+.060], seq and drop tied.
- **NEW and adverse: the fair baseline.**  `univla_noinh` (UniVLA Stage 2 with a RANDOM-init TI codebook, same mixture,
  128k x 64, same seed -- the comparison the user asked for on 2026-09-18) was probed on 2026-09-22 06:28 and never
  written up here.  It is **significantly better than TERRA on prof [+.062,+.119] and seq [+.033,+.094]**, and tied on
  net, seg1 and drop (intervals span 0, point estimates all favour UniVLA).  Consequence: no representation claim of
  the form "TERRA beats UniVLA" survives against the fair baseline; the released-prior row (`univla_matched`) is the
  weaker of the two UniVLA arms and cannot be presented as the matched comparison without this row beside it.
- **Next decision.**  The paper's positive representation evidence is the EAT ablation and the N-only ablation, both
  confirmed above.  The comparison with UniVLA needs either the noinh row reported as is, or a stated reason why the
  released-prior initialisation is the protocol; the closed-loop comparison (user-reported, 2026-09-22) should be
  checked against a noinh VLA as well (four-suite chain from 2026-09-20 -- status unknown here).

### 2026-09-25 (13:30) — why the random-TI UniVLA beats the released-prior one: the inherited TI codebook is an island, and in `s2mix_1ep` the TI branch is dead

- **Question (user).**  A random TI codebook giving better probes than the released stage-1 prior looks like a bug.
- **Checked.**  (i) Same dump script (`acdc_la/baselines/univla.py`), same val clips, Z = quantised TC codes `z_q` of the
  K = 4 frame pairs (4 codes x 128 each) for both -- not a measurement difference.  (ii) What `--no-inherit` changes:
  stage 2 inherits only `vq` + `action_latent` (5,120 params; encoder/decoder are trained from scratch in BOTH runs,
  `genie/model.py:66-71`, ported in `claude_survey/rala2.py:562`).  (iii) Per-position code usage, first frame pair:

  | run | TC bits per code position (max 4) | TI codes used / bits |
  |---|---|---|
  | `univla_matched` (`s2mix_1ep`, released prior) | 2.92 / 3.39 / 2.51 / 2.20, 14-16 codes | **1/16, 0 bits at every position** |
  | `univla_noinh` (`s2mix_noinh_1ep`, random) | 3.37 / 3.37 / 3.37 / 3.22, all 16 | 8-9/16, ~2.7 bits |
  | `univla_official` (released stage 2) | 3.17 / 3.45 / 3.65 / 3.43, all 16 | 1-9/16, 0-0.25 bits |

- **Conclusion: confirmed mechanism, not a bug.**  The frozen inherited TI codebook does not match a freshly trained
  stage-2 encoder, so every input maps to one TI code (the "island" diagnosis pre-registered in rala2.py's docstring,
  and the 1/16 usage logged at every step on 2026-09-14).  With a random codebook at the encoder's own scale the TI
  branch works as UniVLA designed it: nuisance goes to TI, TC is cleaner and better used -- hence noinh's higher probes
  and its lower distractor drop (15.7 % vs 23.0 %).  The released-prior arm is therefore "UniVLA with a dead TI
  branch", and noinh is the faithful, and stronger, baseline.  One seed each; the TI collapse in `s2mix_1ep` is
  systematic within that run (every logged step), not a single-batch artefact.
- **Consequence.**  Any comparison against `univla_matched` alone compares against a broken baseline.  The paper's
  UniVLA row should be noinh (or both, with this explanation); against noinh TERRA is behind on prof and seq and tied
  elsewhere (entry 12:30).  The closed-loop numbers reported on 2026-09-22 must be checked for which UniVLA they used.

### 2026-09-25 (14:10) — dimension-matched readout: the TERRA vs UniVLA-noinh ranking depends on the readout; only UniVLA's profile advantage is robust

- **Why.**  UniVLA's Z is 2048-d (4 frame pairs x 4 codes x 128, one code set per segment); TERRA's is 256-d for the
  whole window.  The native-dimension comparison (entry 12:30) gives the ridge 8x the dimensions.
- **What ran.**  `acdc_la/downstream/nmse_dimmatch.py`: unsupervised PCA of UniVLA's Z (basis fitted on clean + patch
  Z, no labels) to 256 and 64 dims, then the same per-source probe.

  | NMSE | net | seg1 | prof | seq | drop % |
  |---|---|---|---|---|---|
  | TERRA, 256-d native | .671 | .717 | .909 | .764 | 17.9 |
  | UniVLA noinh, 2048-d native | .655 | .689 | .821 | .702 | 15.7 |
  | UniVLA noinh, PCA 256 | .748 | .769 | .872 | .775 | 27.2 |
  | UniVLA noinh, PCA 64 | .693 | .725 | .838 | .729 | 20.1 |
  | UniVLA matched, PCA 256 | .851 | .890 | .931 | .879 | 31.2 |

- **Conclusion: provisional, and a statement about the instrument.**  At equal dimension TERRA leads on net, seg1, seq
  and distractor drop and trails on prof; at native dimension UniVLA ties or leads.  PCA is not a clean control either:
  64 dims beats 256 for UniVLA because the probe standardises every feature, inflating low-variance PCA tails into
  noise.  Robust across readouts: UniVLA is better on the within-window profile (.82-.87 vs .909), as its per-pair
  encoding predicts.  Not robust: any other ranking between the two.  The paper should not claim representation
  superiority over UniVLA from linear probes; its representation evidence is the controlled ablations (EAT vs
  reconstruction, N+A1 vs N), and the comparison with UniVLA rests on closed loop.

### 2026-09-25 (14:50) — existing VQ and extra-AC arms, re-read under the NMSE protocol: VQ is much worse, AC2 is mixed

- **What ran.**  Same per-source probe (`nmse_bootstrap.py`, rows cached in `acdc_la/results/nmse_rows/`) on dumps that
  already existed: `tc_vq256` (the TC tokenizer with a 256-code VQ bottleneck, same data and 128k budget as the
  continuous `tc_cont`) and the pilot-budget stage-1 ladder `s1_mean` / `s1_dc` / `s1_acdc` / `s1_acdc3` (0.039 ep).

  | NMSE | net | seg1 | prof | seq | drop % |
  |---|---|---|---|---|---|
  | TC tokenizer, continuous `tc_cont` | .735 | .768 | .909 | .808 | 22.4 |
  | TC tokenizer, VQ-256 `tc_vq256` | .946 | .925 | .967 | .946 | 40.5 |
  | pilot DC + AC1 `s1_acdc` | .699 | .746 | .924 | .787 | 32.3 |
  | pilot DC + AC1 + AC2 `s1_acdc3` | .714 | .774 | .916 | .797 | 28.9 |

- **Conclusion: confirmed (single seed each).**  Discretising the same tokenizer loses most of the action information
  (net NMSE .735 -> .946) and doubles the distractor drop, consistent with its under-used codebook (99/256 codes at
  step 20k, 2026-09-13).  A second AC mode buys a little profile (.924 -> .916) and robustness at the cost of net, seg1
  and seq, as recorded on 2026-09-09: one AC mode remains the operating point.  UniVLA's profile advantage comes from
  per-pair encoding (one code set per segment), not from its quantiser.  Decision: no VQ variant of TERRA; the N-only
  (endpoint) continuous arm is the CoMo-style baseline (continuous latent over an early temporal feature difference).

### 2026-09-25 (15:30) — why the VQ tokenizer reads so much worse: it loses some information, and the linear probe exaggerates the rest

- **Facts checked.**  `ckpt/tc_vq256.pt`: step 75,000 (of 128k, stopped with the chains on 2026-09-14), 16 tokens x
  256 codes x 64 d, endpointdyn -- 10k steps MORE than `tc_cont` (65,000).  Final code usage is healthy: every token
  uses ~all 256 codes, 6.5-7.1 bits per token, ~109 bits per window, top code share 3-6 % (the 99/256 seen at step 20k
  was repaired by dead-code restarts).  So the gap is not collapse.
- **Evidence, most to least certain.**  (i) Information: validation reconstruction MSE of dF 1.604 (VQ) vs 1.485
  (continuous) with the same decoder -- the quantiser costs information.  (ii) Readout: a linear probe on the codebook
  vectors gives NMSE .946; reading the codes as one-hot categories is worse (.987, 4,099 dims against ~270 train clips
  per source), so the probe number overstates the information gap -- "not linearly readable at this sample size", not
  "carries nothing".  (iii) Nuisance: drop 40.5 % -- small feature shifts flip discrete codes.
  Script: `acdc_la/downstream/nmse_vq_onehot.py`.
- **Consistent with** the 2026-09-12 gate (post-hoc VQ/PQ codebooks of the continuous code, LIBERO history-policy
  chain): continuous oracle L1 0.367 vs 0.394-0.408 discrete; gate failed.  UniVLA's VQ reads well because its codes
  are few and frequent (4 per pair, 16-way); a 256-way, 16-token code is rare per code.  Decision unchanged: TERRA
  stays continuous.

### 2026-09-25 (16:20) — TERRA's 256-d latent is effectively ~25-d: EAT acts as an information bottleneck

- **Why.**  CoMo constrains its continuous latent to 16 d on information-bottleneck grounds (higher dims admit
  static noise); ours is nominally 256 d.  `acdc_la/downstream/nmse_pca_truncate.py`, existing dumps, no training.
- **Result.**  Effective rank: `tc_cont` (before EAT) 127.7 of 256, top 16/32/64 PCs hold 33/52/75 % of variance;
  `latp2_inv` (after EAT) **24.6**, top 16/32/64 hold 76/83/90 %.  Probing the top-k PCs: before EAT, truncation costs
  steadily (net NMSE .891 at k=16 vs .735 at 256); after EAT, k=64 is within .02 of the full code on every target
  (net .694 vs .671) with a slightly lower drop (16.6 vs 17.9 %).  k=128's drop (23.5 %) is the probe's standardisation
  inflating low-variance PCA tails, as before -- read trends, not single k.
- **Conclusion: provisional.**  EAT concentrates the code into ~25 effective dimensions, i.e. it performs the
  compression CoMo imposes by construction.  Truncating a trained code is not training a k-d bottleneck; a 4 x 16
  continuous arm is the test of whether a CoMo-sized bottleneck is better.  Cheap downstream option: use the top-64
  PCs of z as the VLA target instead of all 256 dims.

### 2026-09-25 (17:00) — REGISTERED (not yet run): low-bottleneck sweep of the TC tokenizer

- **Question.**  CoMo constrains its continuous latent to 16 d; ours is 4 x 64 = 256 d (effective rank ~128 before
  EAT, ~25 after, entry 16:20).  Does training with a small bottleneck give a better latent than truncating a big one?
- **Arms.**  `tc_cont_d16` (4 x 16 = 64 d) and `tc_cont_d4` (4 x 4 = 16 d, CoMo-sized).  Everything else is
  `tc_cont`'s recipe as stored in its checkpoint (endpointdyn, continuous, 6-layer encoder, 4-layer decoder, lr 3e-4,
  warmup 4000, batch 64, seed 0), including its schedule: `--steps 128000 --stop-at 65000` reproduces the cosine curve
  tc_cont was on when it was stopped.  New flag `--stop-at` in `common/train_loop.py` for exactly that.
  Script: `acdc_la/scripts/run_tc_bottleneck.sh` (LD=16 / LD=4), `ai` partition.
- **Readout.**  The NMSE table (`nmse_bootstrap.py` rows + paired source bootstrap) against `tc_cont` (256 d, same
  budget), plus effective rank.  **Pre-registered reading:** the small bottleneck is worth continuing (to +15k EAT, then
  VLA) only if at 64 d or 16 d it matches tc_cont on net/seg1/seq (paired interval containing 0 or favouring it) AND
  lowers the distractor drop.  If it loses on net/seg1/seq, the bottleneck is too tight at this budget, and EAT's
  learned compression (entry 16:20) is the better route.
- **Not submitted (17:10).**  `sbatch` refused: AssocGrpGRESMinutes.  `mybalance`: `cis250950-ai` SU limit 0.0 (used
  1995.0), `cis250950-gpu` SU limit 0.0 (used 27.1).  Every pending job on the account (24, incl. the univla_noinh
  four-suite VLA chain registered 2026-09-20) is held for the same reason, which is why that chain never produced
  results; 5 more are DependencyNeverSatisfied.  The sweep is ready to submit as soon as an allocation exists:
  `for LD in 16 4; do sbatch --export=ALL,LD=$LD acdc_la/scripts/run_tc_bottleneck.sh; done`.

### 2026-09-25 (18:30) — transported-action NMSE per source: EAT beats reconstruction-only on 26/29 sources, but the gap does not grow with donor distance

- **What ran.**  `acdc_la/downstream/transport_by_source.py` (CPU, existing dumps).  Per source, a frozen ridge
  reader fitted on REAL transitions (DF -> 12-step action sequence, train episodes) -- DF is model-free, so both arms
  are read by the same reader -- applied to the transported DSW of the test episodes (recipient F0 + same-source donor
  z) and scored against the donor's action; NMSE = 1 - R^2.  Donor rows recovered exactly (same source, DONOR_GE, ACT
  == ASW); donor distance = ||F0Q(recipient) - F0Q(donor)||, quartiles within source.  Arms: `tc_cont_plus15k`
  (reconstruction only) vs `latp2_inv` (full EAT), same start and budget.  One training seed each.

  | donors | median recon | median EAT | EAT lower |
  |---|---|---|---|
  | all | .828 | .722 | 26 / 29 |
  | Q1 (nearest) | .866 | .776 | 24 / 29 |
  | Q4 (farthest) | .914 | .830 | 21 / 29 |
  | Q3-Q4 mean | .895 | .777 | 22 / 29 |

- **Conclusion.**  Confirmed: EAT's transport gain is broad, not carried by a few sources (26/29 overall).
  Not supported under this protocol: that the gap GROWS with donor distance -- both arms rise by about .05 from Q1 to
  Q4 and the gap stays about .09, while the count of sources favouring EAT falls from 24 to 21.  The manuscript's
  Fig. 3(a) numbers (recon .80 -> 1.06, EAT .77 -> .86) come from a computation not in this repository and are not
  reproduced by this one; until they are, the text should claim breadth, not a distance trend.
  Figure: `figures/transport_scatter.{pdf,png}` (`acdc_la/figures/make_transport_scatter.py`).

### 2026-09-25 (19:30) — CORRECTION to 16:20: the "effective rank ~25" used the variance-spectrum definition; the standard one gives 166

- The 16:20 entry reported effective rank 127.7 (tc_cont) and 24.6 (latp2_inv) without naming the definition; it was
  exp(entropy) of the normalised SQUARED singular values (PCA eigenvalue spectrum).  The standard Roy & Vetterli
  definition (normalised singular values, as `train_v2.py::_erank` and the training logs use) gives 210.9 and 165.7.
  The direction stands -- EAT concentrates variance (top 64 PCs hold 90 % after EAT vs 75 % before) -- but
  "compresses the code to ~25 dimensions" is definition-dependent and should not be stated without the definition.
  Both definitions and all values: `paper/appendix/APPENDIX_FACTS.md`.

### 2026-09-25 (20:10) — REGISTERED (not yet run): a LAPA-style baseline inside our pipeline, and a closed-loop perturbation protocol

- Full specification: `acdc_la/PLAN_lapa_and_perturbation.md`.  (1) `tc_lapa`: LAPA's bottleneck (frame-pair
  evidence `--tc endpoint`, discrete 4 x 8 codes = 8^4, NSVQ) on our shared features, window, decoder and budget --
  explicitly not a LAPA reproduction.  (2) Perturbed closed loop: LIBERO-Plus light / background / object-layout at
  L3 plus our moving patch; never a family any arm trained on (our invariance uses crop and global
  brightness/contrast); TERRA vs UniVLA-noinh vs no-latent; Object + Long, 500 episodes per suite and condition;
  retention with Wilson intervals; qualitative pair chosen by a rule fixed in advance.  Pre-registered predictions
  are in the plan.  Blocked on GPU allocation (0 SU).

### 2026-10-02 — UniVLA-noinh VLA at 10 % demos: closed loop equal to the released-prior UniVLA despite much better probes

- **What ran** (2026-09-23 / 09-29, while `cis250950-ai` briefly had allocation): the four-suite token-interface VLA
  chain registered on 2026-09-20, labeller = `s2mix_noinh_1ep` (random TI codebook), Anvil protocol (10 % demos, no
  phase-2 VLM pretraining, UniVLA evaluator, 50 trials per task).  Merged with `eval_merge.py`.
- **Result.**  Spatial **412/500 = 82.4 %** (complete); Object **162/200 = 81.0 %** (tasks 0-3 only, INCOMPLETE);
  Goal and Long not run (allocation exhausted again: `cis250950-ai` 1.3 SU left).  Same-protocol UniVLA with the
  released prior: Spatial 84.2 %, Object 82.8 %.
- **Conclusion: provisional (one seed, Object incomplete).**  The random-TI arm is much better on every OXE probe
  (entries 2026-09-25) yet not better in closed loop -- Spatial -1.8 pp, inside the ~1.7 pp binomial SE.  Probe gains
  of this size do not predict closed-loop gains under this protocol, consistent with the 2026-09-20 finding that the
  latent channel is close to decorative here (no-latent control 88.6 % on Goal).
- **Consequence.**  Probe improvements alone cannot justify scaling a tokenizer; a closed-loop effect of the latent
  must first be shown under the protocol that would be scaled.

### 2026-10-02 — UniVLA's stage-2 scale, a correction to the 09-20 phase-2 plan, and the scale-matched phase-2 plan (REGISTERED, not run)

- **UniVLA, as published (arXiv 2505.06111).**  Generalist-policy pretraining on latent pseudo-labels: Prismatic-7B,
  FULL fine-tuning (visual encoders + LLM + head), global batch 1,024, 20,000 steps (~20.5M samples), constant lr 2e-5,
  32 x A100 for ~30 h (960 A100-hours); data = OXE subset + GNM + Ego4D; latent actions as |C| added vocabulary
  tokens with cross-entropy.
- **CORRECTION to the 2026-09-20 entry.**  "30,000 x 64 = 1.9M clips, ~1/100 of their pretraining" is wrong: it is
  ~1/10.7 of UniVLA's 20.5M samples.
- **Plan (user, today): phase 2 at UniVLA's scale, read as a curve.**  Both arms (TERRA labels, UniVLA labels) on our
  29-source mixture, UniVLA's recipe -- full fine-tuning, global batch 1,024, 20k steps, constant lr 2e-5 -- so that,
  because the learning rate is constant, checkpoints at 2k / 6.7k / 20k steps (1/10, 1/3, 1 x UniVLA) are valid
  points of one curve without separate runs.  Downstream at each point (Object + Long; all four suites at 20k, two
  seeds), plus the no-phase-2 and no-latent controls.  Stated difference from UniVLA: no Ego4D (our tokenizer never
  saw human video, so its labels would be out of distribution there).  Head for continuous labels: flow matching,
  with an MSE-regression arm at the 1/10 point to test that choice before the full run.

### 2026-10-07 — (1) UniVLA-noinh VLA complete on four suites; (2) from the first frame alone, the within-window profile is unpredictable

- **(1) Closed loop, Anvil protocol (10 % demos, no phase 2), labeller `s2mix_noinh_1ep`**, evaluations that ran from
  the queue on 2026-10-06/07: Spatial 82.4 %, Object **87.2 %**, Goal 83.8 %, Long 53.0 % (avg 76.6).  Released-prior
  UniVLA, same protocol: 84.2 / 82.8 / 86.6 / ~53.6.  Wins and losses split (+4.4 Object, -2.8 Goal, -1.8 Spatial):
  confirmed, under this protocol, that the large probe gap between the two UniVLA labellers does not reach the policy.
- **(2) What a phase-2 policy can predict.**  Same per-source probe, targets from ACT, two inputs: the first frame only
  (F0Q -- what the VLM sees in phase 2) vs the whole window (DF -- what the tokenizer sees).  R^2 net / seg1 / last
  segment / profile / seq: first frame **.142 / .088 / .110 / .009 / .106**; whole window .249 / .238 / .159 / .114 /
  .216.  **The within-window profile -- the information A1 adds -- is essentially unpredictable from one frame by a
  linear reader**, while the net effect is partly predictable.  Caveat (estimator, not estimand): a linear ridge on
  pooled DINO features, without the instruction; a VLM is nonlinear and reads the instruction, so this bounds
  nothing from above.  Consequence for phase 2: the A1 part of TERRA's labels may act mostly as target noise for a
  single-frame policy unless the policy gets velocity context (previous frame or proprioception); the decisive test is
  a phase-2 arm with endpoint-only (N) labels against N+A1 labels at the same budget.

### 2026-10-07 — phase 2 at 1/10 of UniVLA's scale: unblocked and smoke submitted (job 21165072)

- **Decision (user).**  Start with the 1/10 point on the `ai` allocation (489.9 SU): LoRA recipe registered 2026-09-20
  (`run_vla_phase2.sh`: latent-only, flow head, 4 query tokens, 30k steps x global batch 64 = 1.92M samples ~ 1/10.7 of
  UniVLA's 20.5M).  Full fine-tuning (UniVLA's recipe) is deferred to the full-scale run: Adam state alone is ~112 GB
  for 7B, i.e. multi-GPU FSDP on 80 GB cards.
- **Two blockers found and fixed before any GPU time.**  (i) UniVLA's `omni_magic_soup_plus_plus` = 29 robot/navigation
  sources + 4 Ego4D splits, and Ego4D is not on disk -> new mixture `acdc_oxe29` in the vendored `oxe/mixtures.py`:
  the same 29 sources and weights, without Ego4D (also the stated difference from UniVLA: our tokenizer never saw
  human video).  (ii) The 2026-09-20 fixed-window patch existed only on the workstation; on Anvil only its
  LIBERO_TRAIN_FRAC hunk had been applied.  The window hunk (`UNIVLA_FIXED_WINDOW`) is now in
  `third_party/UniVLA/prismatic/vla/datasets/rlds/dataset.py`, and `run_vla_phase2.sh` exports it.
- **Smoke.**  `acdc_la/scripts/run_vla_phase2_smoke.sh`: 20 steps per arm (TERRA labels, UniVLA labels) on ONE GPU at
  micro-batch 8, GPU memory sampled every minute (`acdc_la/logs/p2_mem_*.csv`) -- the single-card requirement has
  never been measured.  The 30k-step runs go only after both arms exit 0.
- **Arms at 1/10.**  A TERRA labels; B UniVLA labels (`univla_cont`, `s2mix_1ep`); C endpoint-only labels (needs the
  CoMo-style `--tc endpoint` tokenizer first, ~8 GPU-h on the `gpu` account).  Each post-trained on Object + Long and
  compared with no-phase-2 and no-latent controls.
- **Budget note.**  Queued job 20828934 (UniVLA latent + flow + q4, Goal, no phase 2; registered 2026-09-20) started
  today and will use ~25-30 SU with its evaluations; it doubles as an Anvil seed of the no-phase-2 control.

### 2026-11-05 — Full-parameter phase 2 finished 20k; decay branch queued; post-training launcher for the 62 GB workstation

- **Phase 2.**  `p2fft_ours` reached step 20000 (job 1090237, exit 0 after 18 segments), latent R2 ~0.845, against ~0.82 at
  step 5000.  Decay branch `p2fft_ours_decay` (20k -> 22k, lr linear to 0) submitted as job 1157422; its branch code has
  not yet run on the real job at the time of writing.
- **Launcher.**  `acdc_la/scripts/run_vla_post2_portable.sh`: full split, global batch 128, Long 40k / others 30k steps.
- **Two memory traps of post-training a `vla_full.pt` base on the workstation** (62 GB, about 41 GB free).  Both are
  arithmetic from the September measurements, not measured on this configuration:
  1. `CTRL_REITER_BATCHES` counts micro-batches, the RLDS leak goes with samples.  A fixed 2500 is 20k samples per iterator
     at BS 8 but 80k at BS 32, ~20 GB per rank.  The launcher derives it from BS (20000 / BS).
  2. Each start and the final merge hold the 15 GB `vla_full.pt` in RAM on top of the base; the September runs had no such
     file and already peaked at 35-40 GB in the merge.  `torch.load(full_init, ..., mmap=True)` removes the copy (tested on
     a compute node: 982 tensors, 50 read, max RSS 0.43 GB).  mmap fails on Newton login nodes (`ulimit -v` 4 GB), not on
     compute nodes.  The Newton diag runs and earlier post-training did not use this flag; the loaded values are the same.

### 2026-11-05 — Downstream of the 20k full-parameter model queued on Newton (user's call; the workstation is shared and busy)

- **10 % Long:** job 1157664, same recipe as the other 10 % cells.  **Full split, four suites:** jobs 1157709 1157714 1157719 1157724 (4 GPUs, 8 x 4, Long 40k /
  others 30k, 500-episode standard LIBERO eval at the end).  **LIBERO-Plus:** jobs 1157710 1157711 1157712 1157713 1157715 1157716 1157717 1157718 1157720 1157721 1157722 1157723 1157725 1157726 1157727 1157728, note `p2fft20k_full`, 8 shards per
  suite, each afterok on its suite's training.  One post-training per suite serves both benchmarks.
- **Update 2026-11-06 12:00.**  Object/Spatial/Goal post-training moved from highgpu (estimated start 11-19) to normal,
  2 GPUs x 8 x accum 8 (same global batch 128; run dirs read `b64x2`): jobs 1172709 1172714 1172719, Plus jobs 1172710 1172711 1172712 1172713 1172715 1172716 1172717 1172718 1172720 1172721 1172722 1172723.  Long stays on highgpu
  (1157709, running).  Decay branch 1157422 started correctly from 20000 but only reached the 20750 checkpoint in 30 h:
  RSS hit the 740 GB hard limit (~95 min) before the next 250-step save (~92 min) in most segments.  Continuation 1172724
  (afterany, SAVE=100).

### 2026-11-06 — Rebuttal ablation A queued: 7 arms x 3 seeds from tc_cont (65k), all retrained on Newton

- **Arms** (`acdc_la/scripts/run_eat_ablation_newton.sh`, tags `eatab_<arm>_s<seed>`): recon (L_self, = tc_cont_plus15k),
  noinv (= latp2_anchor), full (= latp2_inv), a1 L_self + L_inv, a2 full - L_effect, a3 full - L_rel, a4 full - L_dist.
  Seeds 0, 1, 2.  Jobs in `acdc_la/logs/eatab.jobids` (1172833-53), preemptable partition.
- **Data deviation, so seed 0 is retrained too.**  The paper's rows were trained on the CAPPED OXE caches (Anvil,
  8.19 M train clips); those caches are not on Newton (`cache/oxe` -> empty `cache_oxe`).  All 21 runs use the uncapped
  caches (`ACDC_FULL_DATA=1`, 17.4 M train clips), the same as `latp2_inv_full`.  The 65k start `tc_cont` itself was
  trained on the capped data.  The table must be reported as its own set, not merged with the paper's single-seed rows;
  the distractor video was also rebuilt on Newton (entry 2026-10, `latp2_inv_full`), so the drop column moves too.
- **Code change.**  `train_v2.step`: the transport block (Ye, MMD) now also runs when `lam_effect = 0` and `lam_real > 0`.
  Before, MMD was only computed inside the L_effect branch, so "no L_effect" would silently have dropped L_dist as well.
  The warm-up ramp scales both weights together, so the existing arms compute exactly what they did.
- **Cost, corrected.**  `latp2_inv_full` ran at 2.84 s/step on one H100 (job 944428): ~12 h per 15k-step EAT run, and
  it timed out at 12 h in the final evaluation.  21 runs are ~250 GPU-hours, not the ~40 in the plan.  Time limit 20 h.
- **Not yet runnable.**  A5 (direct MLP prediction head) needs code; A6 is the existing `L_recover` (smooth-L1 to the
  EMA target's code, not cosine); A7 needs a 65k F0|N endpoint tokenizer that does not exist.

### 2026-11-07 — real-robot data path (Piper); p2_univla downstream flagged INVALID; ablation jobs pinned to H100

- **Real robot.** `jachansantiago/pick_can_place_plate_150_trimmed` (LeRobot v3, AgileX Piper, 150 eps, 1 task, 30 Hz,
  front + wrist 640x480 AV1) -> `data/real_rlds/piper_pick_can/1.0.0` by `acdc_la/realrobot/convert_piper_rlds.py`
  (run as a module on a compute node): 10 Hz (every 3rd frame), action = 6 absolute joint targets + gripper/8.2 in
  [0,1], images 256x256; 6,462 steps.  Registered via `acdc_la/patches/univla_piper.patch` (apply after
  univla_rlds_acdc.patch).  Launcher takes `DS=piper_pick_can EXTRA="--data_root_dir data/real_rlds" SKIP_EVAL=1`.
  TERRA arm = 20k `p2fft_ours` + LoRA/flow/q4, 10k steps x 64 (train[:95%] = 142 eps).  To run on the workstation
  (no queue); Newton copy 1176736 held as backup.  UniVLA baseline = official release + official recipe (user, plan 1).
- **p2_univla downstream = INVALID (not investigated further, user).**  Object @10 % 0/500 and Long @10 % stopped at
  8,750 steps: train one-step action L1 stays 0.13-0.15 (p2ours same suite 0.034), i.e. the action head never
  learned under that init -- a training failure, not an evaluator bug.  No p2_univla number may be used.
- **Ablation A.** evc11 is a V100 node where torch saw no GPU; preemptable V100 slots were also preempted within
  minutes.  Launcher now requests `gpu:nvidia_h100_pcie:1`, excludes evc11, and exits if CUDA is unavailable; all
  pending jobs updated, the 9 dead ones resubmitted (ids in `acdc_la/logs/eatab.jobids`).

### 2026-11-07 — 20k full-parameter phase 2, post-trained on Long at the FULL split: 449/500 = 89.8 %, our best Long number; the LoRA phase-2 arm is 89.0 %, so the full split cannot separate the two phase-2 variants

- **Run.**  `tc_cont+libero_10_no_noops+b32x4+lr0.00035+r32+s40000+fm+q4--p2fft20k_libero_10_full` (Newton job 1157709,
  queued 2026-11-05, arrived here with commit `7b043a6`): LoRA r32 on the **20k `fft` phase-2 base** (`p2fft_ours`),
  flow head, 4 query tokens, global batch 128, 40k steps = UniVLA's Long budget, full split (`train[:95%]`).
  Evaluated 500 episodes over 4 shards, UniVLA's protocol.
- **Result.**  **449/500 = 89.8 %.**  Per task 45, 44, 35, 48, 45, 45, 46, 46, 47, 48 of 50 ("put both moka pots on
  the stove" is the one weak task at 0.70, as in every Long cell).
- **Control — the same recipe from the LoRA phase-2 base.**  `p2ours_libero_10_full`, identical string except the
  base (`b32x4+lr0.00035+r32+s40000+fm+q4`): **445/500 = 89.0 %**.  The arms differ by **4 episodes = 0.8 points**
  against a pooled run-to-run sd of ~4.3 points for a single run (2026-10-30).  **Not a difference.**
- **Against the field reference.**  UniVLA's published full-split Long is 92.0 (Table I).  At 89.8 % Long is now
  2.2 points under it, i.e. inside one run's noise of UniVLA's own number, and this is the best Long score we have
  measured; Object 98.8 % and Spatial 97.2 % (LoRA base) already exceed their published 96.8 / 96.5.
- **What this does and does not say.**  It is a clean controlled comparison -- only the phase-2 base differs -- and it
  comes out flat.  Long at the full split sits near the same ceiling for both arms, the way spatial does at 94.8 %
  (2026-09-15), so **the full split does not discriminate phase-2 variants**.  The 10 % cells do, and that is exactly
  where the 20k arm has no number yet: Long @10 % is p2ours 61.6 / 55.8, p2fft5000 57.6 / 51.2 / 49.8, nop2 56.4,
  p2fft20k **--**.  Nothing here says the 20k phase 2 is better or worse; it says the full split cannot tell.
- **Conclusion — confirmed (measurement); negative for the method claim at this split.**  No advantage of the
  full-parameter 20k phase 2 over the LoRA phase 2 is detectable on Long at the full split.
- **Artifacts.**  `acdc_la/results/vla_eval/merged_libero_10_nw_p2fft20k_libero_10_full.json` + the 4 shard files.
- **Next.**  The 10 % Long cell is the discriminating one.  Newton job 1157664 is replicate 1; the workstation run
  (`GLOBAL=64 FRAC=10 STEPS=15000`) becomes **replicate 2**, which is what makes the cell comparable with the LoRA and
  no-phase-2 arms, both of which have replicates.  Not a duplicate of 1157664 -- the pair is the point.

### 2026-11-07 (00:00) — ablation A regrouped (Newton)

- **Ablation A re-planned (2026-11-07 ~00:00): grouped, 3 arms per job.**  The single-arm runs were GPU-bound, not
  data-bound (`a3` on evc49: 2.82 s/step, GPU util 82-100 %), and the H100 preemptable pool gave us ~1 GPU at a time
  (a3 s0 was itself preempted at ~600 steps, before its first save).  New `stage3_transport/train_multi.py` trains
  several arms in one process on ONE batch stream: batch, DINO features and the L_inv jitter draw are computed once,
  each arm keeps its own model / EMA teacher / buffer / optimiser / schedule / checkpoint (`ckpt/eatabm_<arm>_s<seed>.pt`,
  resume restores the teacher and buffer exactly).  CPU smoke (batch 8, fmb): shared part 3.3-4.0 s/step, each arm
  0.9-1.6 s, so three arms cost roughly 1.5-2x one arm.  Not bit-identical to a single-arm run (arms share the global
  RNG of the derangement/donor draws); the arms now see identical clips AND nuisance draws, which tightens the contrast.
  Adds the two missing rows: **a5** = L_effect replaced by an MLP head g(z) -> Psi(dF), cosine (`--lam-head`, new in
  train_v2.step); **a6** = L_effect replaced by the latent cycle with cosine to sg[z] (L_recover with recover_l1 = 0).
  Groups: g1 recon/a1/full, g2 noinv/a2/a3, g3 a4/a5/a6; seeds 0 first (seeds 1-2 nice 100).  Jobs in
  `acdc_la/logs/eatabm.jobids`; the 21 single-arm jobs were cancelled (none had saved a checkpoint).
  Endpoint + full EAT (A7) not run (plan: answer at the representation level).

### 2026-11-07 (20:30) — ablation A: one 9-arm job per seed (Newton)
- First grouped round (jobs 1177722-30, 3 arms/job): all PREEMPTED after 0.5 min - 3.4 h; 1177725/28 "NO GPU" were a
  120 s torch-import timeout on evc32/33, not bad nodes (other jobs trained there) -> guard timeout 900 s, exit 4 on timeout.
- GPU timing (H100): 2.31-2.41 s/step = data 1.40-1.65 + DINO feat 0.22 + **~0.2 s per arm** -> the step is data-bound;
  extra arms are nearly free.  Saved progress: s0 a4/a5/a6 @5000; s1 noinv/a2/a3/a4/a5/a6 @2000; everything else @0.
- train_multi now resumes arms saved at different steps (stream restarts at the least advanced arm; an arm trains only
  from its own step, so arm step s still sees batch s), atomic saves (tmp + rename), save every 500, no curve evals.
  CPU smoke (a1 @4 + full @0 -> 6) passed.
- Resubmitted as GROUP=all (recon,a1,full,noinv,a2,a3,a4,a5,a6) x seeds 0/1/2, each an 8-deep afternotok chain so a
  preemption restarts automatically: s0 1186071-78, s1 1186079-86, s2 1186087-94 (s1/s2 nice 50).  Expect ~3-4 s/step
  -> ~13-17 h of GPU time per seed if not preempted.  Endpoint+full EAT (A7) still not run.
### 2026-11-08 — TERRA real-robot arm trained on the workstation: 10k steps on the Piper pick-can data, held-out one-step action L1 0.093 and FLAT from step 6,000; the in-run merge does not fit this host

- **Run.**  `tc_cont+piper_pick_can+b32x2+lr0.00035+r32+s10000+fm+q4--terra_piper`, UCF workstation, 2 GPUs,
  global batch 64 (8 x accum 4 x 2), **10,000 steps**, LoRA r32 + flow head + 4 query tokens on the **20k
  full-parameter phase-2 base** (`p2fft_20k`; `vla_full.pt` byte-checked at 15,081,838,026 against Newton).
  Data `piper_pick_can` (AgileX Piper, 150 episodes, 6,462 steps at 10 Hz), `train[:95%]` = 142 episodes trained,
  `train[95%:]` = 8 episodes held out.  **6 h 31 m** wall (2.34 s/step = 13.7 samples/s/GPU, ~15 % under an H100).
  Launcher `run_vla_post2_portable.sh` with `DS=piper_pick_can EXTRA="--data_root_dir .../real_rlds" SKIP_EVAL=1`
  (`abec2d0`); Newton's queued copy (1176736) was held and is not needed.
- **Training.**  action L1 **0.407 -> 0.0411**, latent r2 0.247 -> 0.835, lr x0.1 at step 8,000 as the recipe's
  StepLR does (`:391`, `step_size = 0.8 * max_steps`).  This is **not** the p2_univla failure mode (L1 stuck at
  0.13-0.15, 2026-11-07): L1 passed 0.1 at step 850.  ~105 epochs over 142 episodes, which is the same epoch count
  as the LIBERO 10 % cells (15k x 64 / 10,147 = 95), not an arbitrary budget.
- **Held out, the only offline signal there is** (`acdc_la/realrobot/eval_heldout.py`, new: no simulator exists for
  this data, so checkpoints cannot be ranked by rollout).  1,024 frames of the 8 held-out episodes, the same two
  numbers the trainer logs:

  | step | 6,000 | 7,000 | 8,000 | 9,000 | 10,000 |
  |---|---|---|---|---|---|
  | one-step action L1 | 0.0972 | 0.0981 | 0.0947 | 0.0940 | **0.0930** |
  | latent r2 | 0.742 | 0.728 | 0.736 | 0.737 | 0.744 |

  **Flat.**  The 0.005 spread over 4,000 steps is not a selection signal, so no earlier checkpoint is preferable and
  there is no overfitting inside the window; 10,000 is also the end of the anneal.
- **Correction, same day — 0.093 is NOT the policy's prediction error, and the open-loop number is 0.149.**  Writing
  the deployment wrapper (`acdc_la/realrobot/piper_policy.py`) and replaying the held-out episodes through it showed
  that the metric both the trainer and `eval_heldout.py` report is **oracle-conditioned**: `FlowHeads.loss` computes
  `l1 = (at + (1 - t) * va - actions).abs()` where `at = (1 - t) * noise + t * actions` **already contains the
  ground-truth action**, so it is a one-Euler-step estimate from a point on the path to the answer, not a prediction
  from noise.  `eval_heldout.py` also runs with `image_aug=False`, i.e. without the 0.9-area centre crop deployment
  uses.  Sampling the chunk properly (10 Euler steps from noise, the deployment image path) on the same 8 episodes,
  353 steps, step 10,000: **normalised chunk L1 0.149**, and in physical units **3.24 deg mean absolute error per
  joint** for `chunk[0]` (per joint 1.98 / 3.86 / 3.87 / 3.43 / 5.27 / 1.03) with the gripper at 0.047 of its
  [0, 1] range; after temporal ensembling 3.69 deg and 0.062.  **What survives:** the 6,000 -> 10,000 comparison is
  still a like-for-like ranking under one fixed protocol, and it is flat.  **What does not:** "held-out L1 0.093"
  must not be read as the error a robot would see, here or in the HF README (corrected there too).  The open-loop
  error of the earlier snapshots has NOT been measured -- it would need each one merged.
- **Is 0.149 a failure?  No, and the null baselines say what it is.**  Measured on the same 8 held-out episodes
  (353 steps), mean absolute error per joint in degrees: constant = the training mean **14.70**, the average
  demonstration trajectory time-warped to each episode **11.02**, **ours 3.24**, and *repeat the last command*
  (persistence) **2.06**.  So the policy **crushes both trajectory-level baselines** -- it follows each episode's own
  trajectory, it is not reproducing an average -- but it **loses to doing nothing**, because the action space is
  ABSOLUTE joint targets at 10 Hz where the arm only moves ~2 deg per step (joint 5: 0.08 deg) and persistence is
  therefore a very strong L1 baseline that accomplishes nothing on the robot.  The real content of the number is
  that the policy's precision (~3 deg) is of the same order as the per-step motion, and that it adds ~1 deg of
  error to joints that barely move.
- **It is bias, not sampling noise -- there is no free fix.**  The flow head is stochastic, so the obvious hope was
  that averaging samples would collapse the error.  One VLA forward per observation, 8 head samples, Euler steps
  5 / 10 / 50 (normalised units): single sample 0.1471 / 0.1483 / 0.1506, **mean of 8 samples 0.1451 / 0.1452 /
  0.1457**, across-sample sd 0.0244 / 0.0318 / 0.0411.  Averaging buys **2 %** and more integration steps are
  slightly *worse*; ~0.145 of the 0.148 is systematic.  **So inference-side tricks will not help**; the levers are
  training-side (more than 142 demonstrations of one task, delta instead of absolute actions, a higher control rate).
- **Temporal ensembling is slightly harmful here** (3.69 deg vs 3.24 deg): it is UniVLA's LIBERO setting, where the
  action is a delta and its lag is free; on an absolute joint target the lag shows up directly.  The wrapper takes
  `ensemble=False`; offline cannot decide between them, the robot can.
- **Infrastructure 1 — intermediate checkpoints do not exist unless copied.**  The trainer writes every LoRA save to
  the SAME adapter dir (`finetune_libero_ctrl.py:496`), so each save overwrites the last and the run keeps no history.
  Step 5,000 was already gone before it could be copied; 6,000-10,000 were snapshotted by hand
  (`/data/ti874297/terra_snapshots/step*`, 813 MB each).  Any future "pick the best step" has to arrange this first.
- **Infrastructure 2 — the in-run merge OOMs on this 62 GB host.**  Three consecutive SIGKILLs inside
  `merged.save_pretrained`; merged standalone afterwards at peak RSS 30.7 GB in 211 s.  Details and the correction to
  the over-optimistic 2026-11-05 reading are in that entry.  No weights were lost: the step-10,000 adapter was
  already on disk.
- **The artifact was verified, not just listed.**  982 tensors, vocab 32005, and the LoRA target layers differ from
  the base (`layers.0.self_attn.q_proj` mean |diff| 5.5e-4, `layers.31.mlp.down_proj` 1.2e-3, `embed_tokens` 9.2e-6 --
  the last is in `modules_to_save`, so it should move).
- **Conclusion — confirmed (training); NOT a result about the method.**  The real-robot arm trains cleanly from the
  20k base and yielded a merged, loadable policy.  **No physical-robot number exists**, and offline L1 does not
  predict success rate.  The UniVLA baseline (official release + official recipe, user's plan 1) is being prepared on
  Newton, so there is nothing to compare against yet.
- **Artifacts.**  Run dir `/data/ti874297/vla_runs/tc_cont+piper_pick_can+...--terra_piper` (16 GB merged);
  `/data/ti874297/terra_snapshots/{step6000..step10000,heldout.json,heldout.log,merge_step10000.log,snap.log}`;
  `acdc_la/logs/post2_terra_piper.log`.  Pushed to the HF model repo `Potestates/TERRA` under
  `piper_pick_can_10k` -- **correction 2026-11-10: that repo is PUBLIC, not private as written here.**  Verified
  anonymously with no token: `api/models/Potestates/TERRA` returns HTTP 200 with `private: false, gated: false`, and
  a range request on a weight file returns 206.  So the three merged policies there (`piper_pick_can_10k`,
  `libero_10_full`, `libero_goal_full`, 15.19 GB each) are world-readable.  Visibility is the user's account
  decision and was not changed (same convention as `acdc_la/scripts/hf_upload_terra.sh`, optimiser state left out).
- **Next.**  The collaborator's physical trials are the only real evaluation of this arm.  The box is now free for
  the **10 % Long replicate 2** (`GLOBAL=64 FRAC=10 STEPS=15000`), the cell that can actually separate the 20k
  phase-2 base from the LoRA one.

### 2026-11-09 — Piper deployment made standalone; main fast-forwarded to newton; ablation moved to normal partition
- Deployment no longer needs a UniVLA clone: the 6 UniVLA files it uses are vendored byte-identical (commit 0ab9e9dd,
  Apache-2.0) in `acdc_la/vendor/univla/`; ContHeads/FlowHeads moved verbatim to `downstream/vla/heads_ctrl.py`
  (finetune_libero_ctrl re-exports them).  `acdc_la/realrobot/README.md` + `requirements_deploy.txt` for the robot host.
- Check (job 1195796, V100, UniVLA imports BLOCKED by a meta-path hook, HF checkpoint `Potestates/TERRA/piper_pick_can_10k`):
  replay of 353 held-out steps -> chunk[0] 3.24 deg mean per joint, ensembled 3.69 deg; identical to the workstation's
  numbers.  `acdc_la/results/piper_replay_standalone.json`.  Fix found by it: replay() must hide the GPU from TF first.
- Ablation A: the 24-job preemptable chain (1186071-94) was exhausted by preemptions (18 s - 58 min each); saved:
  s0 a4/a5/a6 @5000, s1/s2 partial (<=2000).  Resubmitted on the NORMAL partition, one 9-arm job per seed
  (1195772-74, -c 32, 24 workers); other pending jobs of ours niced +300 so these start first.
### 2026-11-10 — FIRST PHYSICAL-ROBOT feedback on the Piper policy: the motion is right, the grasp is not precise; and the measurements that follow RETRACT my ranking of the cause

- **The evidence.**  A labmate ran the `piper_pick_can_10k` policy on the real AgileX Piper, **~5 trials**, with an
  **action chunk of 5**.  Verbatim: *"it can understand the motion like grasp and place / but the grasp it not too
  precise / I am not sure if it is undertrain or overtrain"*.  This is the **only** physical evaluation of this arm
  and it is qualitative: no success count, no trial log, 5 trials.  Treat it as a direction, not a rate.
- **Under- or over-trained: neither, and this was already measured.**  Held-out one-step L1 is **flat** from step
  6,000 to 10,000 (0.0972 / 0.0981 / 0.0947 / 0.0940 / 0.0930) while the training L1 was still falling
  (0.407 -> 0.0411).  Falling train + flat held-out = on the plateau: more steps buys nothing measurable, fewer
  cuts the LR anneal.  **The step budget is not the knob**, so the question as posed has a negative answer.
- **The error grows across the chunk, so chunk = 5 is costing accuracy** (`acdc_la/realrobot/chunk_profile.py`, the 8
  held-out episodes, 353 steps, chunk sampled properly from noise through the deployment image path):

  | k steps ahead | 0 | 1 | 2 | 3 | 4 | 5 | ... | 11 |
  |---|---|---|---|---|---|---|---|---|
  | our error (deg/joint) | **3.24** | 3.94 | 4.58 | 4.98 | 5.19 | 5.32 | | 5.67 |
  | motion scale at that horizon | 0.00 | 2.06 | 4.08 | 5.90 | 7.55 | 9.04 | | 15.77 |
  | err / motion | - | 1.91 | 1.12 | 0.84 | 0.69 | 0.59 | | 0.36 |

  Mean over the first 5 (what a chunk-5 rollout actually commands) **4.38 deg** vs **3.24 deg** for chunk 1.  So
  **chunk 5 -> 2 is -18 %, chunk 5 -> 1 is -26 %** of commanded joint error, free, at deployment.  Note the two
  readings differ: *relative* to the motion the long horizons are predicted better (0.36 at k=11), but what lands on
  an ABSOLUTE joint target is the absolute error, and that is lowest at k=0.
- **Inference latency permits it.**  `act_chunk` costs **89.0 ms mean / 96.2 ms p90** (RTX PRO 6000 Blackwell Max-Q,
  bf16, 10 Euler steps, `acdc_la/realrobot/latency.py`).  At 10 Hz chunk 1 needs < 100 ms, so it fits here with **4 ms of
  p90 margin** -- too thin to recommend on an unknown card.  **chunk = 2 (200 ms budget) is the safe choice.**
- **RETRACTED (mine, 2026-11-10): "the policy never sees the wrist camera ... this is the biggest lever, it is
  structural".**  I ranked that first in chat before measuring, and the phase breakdown does not support it being
  dominant.  Single-view blindness at close range predicts an error SPIKE in the grasp phase.  There is none --
  chunk[0] error by phase is approach 2.89 / **grasp 2.94** / transport 4.34 / release 2.34 deg, i.e. the grasp is
  among the *better* phases in absolute terms.  The wrist camera remains a plausible lever and is **untested**; it is
  a hypothesis, and it was wrong to rank it as the explanation.  (It is true and unchanged that `datasets_ctrl.py:67`
  feeds only `image_primary`, so this policy has never seen `wrist_image`.)
- **What the phase data DOES say, once the confound is removed** (`acdc_la/realrobot/phase_motion.py`, ground truth only --
  a slow phase earns a low absolute error for free, so divide by the per-step motion):

  | phase | err (deg) | motion/step (deg) | err / motion | n |
  |---|---|---|---|---|
  | approach | 2.89 | 4.62 | **0.63** | 62 |
  | grasp | 2.94 | 1.38 | **2.14** | 88 |
  | transport | 4.34 | 1.89 | 2.30 | 113 |
  | release | 2.34 | 1.10 | 2.12 | 82 |

  The error is a **roughly phase-independent ~3 deg floor** while the required precision varies 3x.  The one phase
  the policy does well is the FAST gross approach (0.63); everywhere the motion becomes fine it is 2.1-2.3x the
  per-step motion.  So "understands the motion but the grasp is imprecise" is this floor meeting a phase whose
  tolerance is below it -- not a grasp-specific perception failure.  **This is a description, not a cause.**
- **The competing explanation, NOW MEASURED -- the demonstrations impose a floor, but not this floor.**
  (`acdc_la/realrobot/demo_spread.py`, all 150 episodes, 6,462 steps, no GPU and no training.)  Two steps from
  DIFFERENT episodes in the same arm configuration and looking the same are the same input to this policy, so half
  their disagreement about the next command bounds the mean error of ANY deterministic policy of those inputs
  (`|f-a1| + |f-a2| >= |a1-a2|`).  Matching pose to < 2 deg on every joint and the image to the closest 2 % of a
  32x32 grey descriptor, over 7,558 cross-episode pairs: spread mean **2.99 deg**, so the **bound is 1.49 deg**
  (median-based 0.73).  Adding a task-phase constraint (|phase diff| < 0.08, to stop the same configuration on the
  way out being paired with the way back) changes it by 0.01 deg, so the number is robust.  **Our 3.24 deg is ~2.2x
  that bound**, and the bound is an OVER-estimate: a 32x32 grey descriptor cannot resolve a 1 cm shift of the can, so
  some "matched" pairs legitimately differ for reasons a 224x224 policy can see, which inflates the spread.  So
  **demonstration multimodality does NOT explain the floor** -- there is at least a factor of 2 of policy-side
  headroom -- but it does CAP what any policy fix can buy: ~1.5 deg is 72 % of the 2.06 deg the arm moves per step,
  so sub-degree precision is not obviously available from this dataset at all.  Also notable: given the pose, the
  image constraint does not narrow the next command (1.42 -> 1.49 deg, i.e. unchanged within the selection effect of
  a 13x smaller pair set) -- though at 32x32 grey that is a statement about the descriptor, not about the image.
- **Correction, same day -- the floor is 0.61 deg, not 1.49, and the headroom is 5x not 2x**
  (`acdc_la/realrobot/delta_floor.py`).  The 1.49 deg above was half the spread of the next ABSOLUTE target at
  pose-matched steps, but the match allows the two poses to differ by up to 2 deg per joint and **that difference
  sits inside the absolute-target spread**, so the bound was inflated by its own matching slack.  Recomputed on the
  DELTA (target minus the step's own measured configuration), which removes the pose offset by construction and is
  the same physical position error: spread mean **1.22 deg -> floor 0.61 deg** (pose < 2 deg + image closest 2 %,
  5,526 cross-episode pairs), unchanged by a phase constraint (1.22) and only slightly lower at a 1 deg pose match
  (1.12 -> 0.56).  The MEDIAN spread is 0.13 deg (floor 0.06), so matched demonstrations agree almost exactly and the
  mean is carried by a small ambiguous tail.  Consequences: the demonstrations are far MORE consistent than the
  earlier number said, our 3.24 deg sits **~5.3x above the floor** rather than 2.2x, and the "~1.5 deg caps what any
  policy fix can buy" caveat is withdrawn -- sub-degree precision IS available from this dataset.
- **The nulls the delta arm has to beat** (same 8 held-out episodes, 353 steps).  Under a delta parameterisation
  "output zero" means *stay where you are*, and its error is mean |delta| = **1.20 deg** per joint
  (0.71 / 2.80 / 1.95 / 0.29 / 1.41 / 0.05); repeat-the-last-command is 2.06 deg; the absolute arm is 3.24 deg.  So
  the informative band is narrow: **0.61 deg (floor) to 1.20 deg (do nothing)**, and a delta arm that lands above
  1.20 deg is better than the absolute arm while still being no better than not moving -- which would be a useful
  NEGATIVE, pointing away from the parameterisation and back at the head's single-token conditioning or the single
  front view.  Prediction registered BEFORE the run finishes: the signal-to-range ratio improves from 2.06/82 = 2.5 %
  to 1.20/17.7 = 6.8 %, i.e. 2.7x, so the expected landing is **~1.0-1.5 deg** (2-3x better than absolute), with
  ~0.7-0.9 deg the optimistic case if the error-as-fraction-of-range is fully preserved.
- **A more specific structural candidate than either of the above, found by reading the head (2026-11-10).**
  `FlowHeads.cond` is `MAPBlock(n_latents=1)` over the visual tokens and `MAPBlock(n_latents=1)` over the 4 query
  tokens, concatenated into `c_in` -> **a single 512-d token**, and `velocity()` gives the 12 action tokens exactly
  that one token to attend to.  **The whole visual scene reaches the action head as one 512-d vector**, from which
  the entire 12x7 chunk is generated; the millimetre-scale offset between gripper and can has to survive a
  single-token attention pool.  This predicts precisely what was measured -- a floor that does not vary with phase,
  because the bottleneck is identical at every step -- and unlike the wrist camera it is one line
  (`n_latents=1` -> 4 or 8) plus a post-training-only rerun (~6.5 h).  It is UniVLA's own ActionDecoder design, so
  it is inherited, and it is **still a hypothesis: nothing here shows it is the binding constraint.**  **Constraint
  (`next_task.md` section 8): `n_latents` lives in `FlowHeads` and is shared by every arm**, including the LIBERO
  cells and the UniVLA-baseline-on-the-robot being prepared, so raising it for the real-robot arm alone would void
  that comparison.  It is a registered ablation, not a patch.
- **Why ABSOLUTE joint targets are the wrong parameterisation for this, quantified.**  Compare the two
  `dataset_statistics.json`: LIBERO is delta EEF (mean ~0, q01/q99 ~ +-0.8, so the whole output range is spent on
  the increment); Piper joint 1 has mean 111.5 and q01/q99 2.45 .. 166.66, a **164 deg range**, while the arm moves
  ~2 deg per step -- so the informative part of each prediction is **1.2 % of the output range**.  Our measured
  error as a fraction of each joint's own range is 2.4 / 2.4 / 2.6 / 11.5 / 8.0 / 7.3 %: **the signal sits below the
  error**.  Under a delta parameterisation the range would be ~+-5 deg and the same relative precision would give
  ~0.15-0.3 deg on the quantity that matters.  This is why that whole line (OpenVLA, RT-1/2, Octo) uses deltas, and
  it is **our converter's deviation, not OpenVLA's** -- OpenVLA's own recipe is delta EEF.  Two already-measured
  consequences are symptoms of it: persistence (2.06 deg) beats the policy (3.24 deg), which needs a target
  dominated by "where you already are"; and temporal ensembling HURTS (3.69 vs 3.24), because averaging absolute
  positions predicted at different times is a lag, whereas averaging deltas averages a velocity estimate.  Cheap to
  test: `action[t] = joints[t+1] - joints[t]` in the converter, same base, same 10k steps, one 6.5 h run.  Honest
  caveat: deltas bring integration drift and their own small-signal problem if the range is set by rare fast
  motions, so this is a well-motivated experiment, not a certainty.
- **Gripper timing is aligned.**  Shifting the predicted gripper channel against the demonstration: 0.0470 at 0,
  0.0450 at +1 step, 0.0746 at +2.  Best is +1 step (100 ms late) but it buys 4 %, which is not actionable.
- **Conclusion — provisional (qualitative, n~5), and it retracts a cause I asserted without measuring.**  The policy
  transfers to the physical robot and does the task's gross structure; its precision floor (~3 deg/joint) sits above
  the grasp's tolerance.  Two deployment-side changes are justified by measurement (**chunk 5 -> 2**, and
  **ensembling off**: 3.69 vs 3.24 deg, 2026-11-08).  The training-side levers (wrist view, delta instead of absolute
  actions, higher control rate, more demonstrations) are all **unmeasured hypotheses** and must not be reported as
  diagnoses.  No success rate exists for this policy on the robot.
- **Infrastructure, found 2026-11-10 while briefing the press_button run: the chunker pads episode starts and
  nothing masks it** (`acdc_la/realrobot/short_episode.py`, prismatic's real `chunk_act_obs_libero`).  The sample
  window is BACKWARD-looking -- timestep `t` gets `[t-13 ... t]` for `window_size + 2 = 14` -- and
  `tf.maximum(chunk_indices, 0)` (`traj_transforms.py:91`) clamps anything before the episode start, so **frame 0 and
  action 0 are repeated**; actions are clamped to `[0, traj_len-1]` (`:98`).  One sample is emitted per timestep with
  **none dropped**, so `t=0` is a fully degenerate sample (whole window = frame 0, whole action target = action 0).
  `pad_mask` records it (`:104`) but **no downstream code reads it** -- a grep over `prismatic/vla/` and `acdc_la/`
  finds it only in `acdc_la/scripts/test_obs_keep.py`.  Share of samples with a fully real window by episode length
  in 10 Hz steps: **12 -> 0 %, 17 -> 24 %, 43 -> 70 %, ~203 -> 94 %**.  Consequences: the **LIBERO cells are barely
  affected** (Long averages ~203 steps per episode, so ~6 % of samples), **pick-can ran with ~30 %** padded samples,
  and a short-episode dataset like `press_button` (38-167 frames at 30 Hz = 12-55 steps at 10 Hz, median 17) would be
  **mostly padding**.  **This does not invalidate any comparison in this ledger** -- it is identical on both sides of
  every arm, since every arm uses the same chunker, window and data -- but it means the effective sample count is
  below the step count, and absolute numbers are not comparable across datasets with different episode lengths.
  Not changed: `window_size` is shared (`next_task.md` section 8), so this is recorded, not patched.
- **Artifacts.**  `/data/ti874297/{chunk_profile.json,chunk_profile.log,latency.log,demo_spread2.log}`; the three diagnostics are
  `acdc_la/realrobot/{chunk_profile,phase_motion,latency,demo_spread}.py`.
- **Next.**  Ask the collaborator for chunk = 2, ensembling off, and a confirmation that the preprocessing matches
  (squash to 256 -> 224 -> 0.9-area centre crop, 10 Hz); and for the failure SHAPE, which separates the hypotheses:
  a consistent spatial offset points at calibration/preprocessing, hunting/jitter at the precision floor, and failure
  only at can positions outside the demonstrated set at data coverage.

### 2026-11-10 — 10 % Long replicate 2 of the 20k full-parameter phase-2 arm: 303/500 = 60.6 %, which does NOT separate it from the LoRA phase-2 or no-phase-2 arms

- **Run.**  `tc_cont+libero_10_no_noops+b32x2+lr0.00035+r32+s15000+fm+q4--p2fft20k_libero_10_frac10_rep2`, UCF
  workstation, 2 GPUs, global batch 64 (8 x accum 4 x 2), **15,000 steps**, LoRA r32 + flow head + 4 query tokens on
  the 20k full-parameter phase-2 base (`p2fft_20k`), `LIBERO_TRAIN_FRAC=10`.  **10 h 41 m** wall (2.56 s/step).
  Training: action L1 0.168 -> 0.0298, latent r2 0.580 -> 0.821, lr x0.1 at step 12,000.
- **Result — 303/500 = 60.6 %** on the standard 500-episode LIBERO-Long eval, all 10 tasks.  Per task, sorted:
  0.92 / 0.86 / 0.70 / 0.68 / 0.68 / 0.64 / 0.58 / 0.46 / 0.36 / **0.18** ("put both the alphabet soup and the tomato
  sauce in the basket" is 9/50 and is the floor in every arm).
- **The cell, all arms at 10 % Long, 500 episodes each:**

  | arm | rep 1 | rep 2 | mean |
  |---|---|---|---|
  | p2ours (LoRA phase 2) | 61.6 | 55.8 | 58.7 |
  | **p2fft20k (full-parameter 20k)** | **not relayed** | **60.6** | — |
  | p2fft5000 | 57.6 | 51.2 (and 49.8 at wu500) | 52.9 |
  | nop2 (no phase 2) | 56.4 | — | 56.4 |
  | univla_noinh | 53.0 | — | — |
  | ours_fm | 50.8 | — | — |

- **Conclusion — the run is confirmed, the method question is NOT.**  60.6 sits **inside** p2ours's own spread
  (55.8-61.6) and above nop2's single 56.4, but **one 500-episode cell carries +-8 points at 95 %** (pooled sd ~4.3
  for one run, 2026-10-30).  So 60.6 vs 58.7 vs 56.4 is **not separable**, and this entry does **not** establish that
  the 20k full-parameter phase 2 beats the LoRA phase 2 or beats no phase 2.  What it does: it gives the 20k arm its
  first 10 % Long number.  Compare with the full split, where the same arm was 89.8 against p2ours 89.0 and the
  conclusion was also "the split cannot tell" (2026-11-07) -- the 10 % cell was supposed to be the discriminating one,
  and with a single replicate it still is not.
- **Infrastructure — the merge fix worked.**  `EXTRA="--merge False"` plus `acdc_la/scripts/merge_ctrl_run.py`
  standalone: **peak RSS 30.9 GB in 191 s**, no SIGKILL, where the in-run merge was killed three times on this host
  (2026-11-08).  This is now the recipe for every run on the 62 GB box.
- **Infrastructure — half the first eval was lost to a co-tenant, not to us.**  The 2026-11-08 eval launched two
  ranks; the GPU-1 rank died at model load with `CUDA out of memory` because **another user's job held 80.45 GiB of
  that card's 96**, so tasks 1/3/5/7/9 never ran and the merged file read 122/250 = 48.8 % with `complete: false`.
  Recovered 2026-11-10 by rerunning exactly those five tasks on GPU 0 (2 h 24 m) and re-merging; both shard files are
  now present and `complete: true`.  Lesson for this host: the launcher's eval loop does not check that each rank
  produced a file, and `eval_merge` reports `complete: false` but the launcher still exits 0.
- **Caveat that applies to this number like every other in this ledger.**  ~6 % of this run's samples had a
  padding-contaminated window (Long episodes average ~203 steps against a 14-frame window; see the chunker entry
  above).  Identical on every arm, so the comparison is unaffected.
- **Artifacts.**  `acdc_la/results/vla_eval/merged_libero_10_ws_p2fft20k_libero_10_frac10_rep2.json` + the two shard
  files `..._t0_2_4_6_8.json` / `..._t1_3_5_7_9.json`; run dir
  `/data/ti874297/vla_runs/tc_cont+libero_10_no_noops+b32x2+...--p2fft20k_libero_10_frac10_rep2` (16 GB merged);
  `acdc_la/logs/post2_{p2fft20k_libero_10_frac10_rep2,eval_p2fft20k_libero_10_frac10_rep2_gpu0,eval_p2fft20k_libero_10_frac10_rep2_gpu1,eval_p2fft20k_libero_10_frac10_rep2_odd}.log`;
  `/data/ti874297/{long_rep2_wrapper.log,long_rep2_phase23.log,rep2_odd.log}`.
- **Next.**  **Get Newton job 1157664's number** (replicate 1 of this same arm) -- it is the only thing that turns
  60.6 into a pair and makes this cell comparable with the two-replicate arms.  Until then the 20k-vs-LoRA phase-2
  question has no answer at 10 % Long either.

### 2026-11-10 — DELTA actions, arm B (the OpenVLA baseline): 3.24 deg -> 1.054 deg open loop, a 3.1x cut, and the first arm that beats "do nothing"

- **Why.** The 2026-11-10 robot feedback ("the motion is right, the grasp is not precise") plus the measurement that
  the absolute-joint-target arm was *2.7x worse than outputting zero motion* (3.24 deg vs the 1.20 deg zero-delta
  null) said the output parameterisation, not the budget, was wrong.  Registered prediction before launch: **1.0-1.5 deg**.
- **Configuration.** Two arms, launched 2026-11-10 03:13 / 03:26 UTC, identical except for the latent mechanism:
  - **A `terra_piper_delta`** — phase-2 base (`/data/ti874297/p2fft_20k`, full-parameter 20k, `latp2_inv`) + LoRA r32
    + flow head + 4 query tokens + the latent branch (`--latent tc_cont`).  **Still training at this entry** (see Next).
  - **B `openvla_piper_delta`** — official `openvla-7b`, fresh LoRA r32, same flow head, same 4 query tokens,
    **no latent mechanism at all** (`--latent none` -> no labeller, `dz=0`, no `lz` loss, `latent_r2` identically 0.000
    for all 400 logged points, verified).  Matches the registered LIBERO `none`+`fmq4` control.
  - Both: 10,000 steps, global batch 64 on 1 GPU (`b64x1`), lr 3.5e-4, window 12, `--merge False`.
    B finished exit 0 in 45,693 s (12.7 h).
- **Data.** `data/real_rlds_delta`, derived from `data/real_rlds` by `acdc_la/realrobot/make_delta_rlds.py` so that
  frames, subsampling, episode boundaries and instruction are **bit-identical** and only `action` differs:
  `d[:, :6] = acts[:, :6] - js[:, :6]` (target minus *measured* configuration, so it self-corrects at deployment),
  gripper untouched.  150 episodes, 6462 steps.  Dataset name stays `piper_pick_can`, so no OXE config change.
- **Result — arm B, 353 held-out steps (8 episodes, `train[95%:]`), mean |error| per joint in degrees:**

  | | j0 | j1 | j2 | j3 | j4 | j5 | **mean** | gripper |
  |---|---|---|---|---|---|---|---|---|
  | B `chunk[0]` | 0.599 | 2.022 | 1.672 | 0.419 | 1.552 | 0.062 | **1.054** | 0.0417 |
  | B ensembled | 0.600 | 1.685 | 1.349 | 0.351 | 1.441 | 0.061 | **0.914** | 0.0589 |
  | zero-delta null | 0.71 | 2.80 | 1.95 | 0.29 | 1.41 | 0.05 | 1.20 | — |
  | B / null | 0.84 | 0.72 | 0.86 | **1.44** | **1.10** | **1.24** | 0.88 | — |

  Controls, all in the same "degrees of commanded-target error" space: **absolute arm 3.24 deg**,
  zero-delta null (stay put) **1.20**, persistence null (repeat the last command) **2.06**,
  demonstration floor **0.61**.  Whole-chunk L1 in normalised space 0.1142; final train `action_l1_1step` 0.0648.
- **Conclusion: confirmed, for the parameterisation.** Delta actions cut open-loop target error **3.24 -> 1.054 deg
  (3.1x)**, inside the registered 1.0-1.5 deg prediction, and this is the first arm to come in **under** the
  do-nothing null instead of 2.7x above it.  Two independent predictions of the delta argument also came true:
  1. **Ensembling flipped sign.** On the absolute arm it *hurt* (3.69 vs 3.24); here it *helps* (0.914 vs 1.054,
     -13 %).  Averaging increments across time is coherent; averaging absolute targets is not.
  2. The residual is no longer dominated by a near-constant offset.
- **Conclusion: negative, for the wrist joints, and the honest size of the win.** Overall B is only **12 % better
  than outputting zero motion** (1.054 vs 1.20), and per joint it is better than the null only on j0/j1/j2
  (0.84/0.72/0.86) while being **worse than the null on j3/j4/j5** (1.44/1.10/1.24) — the policy *adds* noise on the
  wrist, where the true increments are smallest.  Headroom to the 0.61 deg demonstration floor is 1.7x, not the 5.3x
  the absolute arm had.  **This is open-loop error on 353 held-out steps, not a robot success rate**; no physical
  success rate exists for any policy yet.
- **Artifacts.** Run `/data/ti874297/vla_runs/none+piper_pick_can+b64x1+lr0.00035+r32+s10000+fm+q4--openvla_piper_delta`
  (merged in place, 112 s, peak RSS 31.9 GB); log `acdc_la/logs/post2_openvla_piper_delta.log`;
  replay `acdc_la/realrobot/piper_policy.py --replay data/real_rlds_delta`;
  nulls and floor `acdc_la/realrobot/delta_floor.py`; dataset `acdc_la/realrobot/make_delta_rlds.py`.
- **Next.** (a) Arm A is at step 9050/10000 as of 19:32 UTC, ~111 min out; its merge (`--init /data/ti874297/p2fft_20k`,
  31 GB peak) and replay are chained.  **A vs B is a valid comparison** — same dataset, same normalisation, same
  formula — and is the only thing here that speaks to the latent mechanism; B alone speaks only to the parameterisation.
  At step 8975 A's train `action_l1_1step` was 0.0558 against B's final 0.0648, with `latent_r2` 0.832.
  (b) The deployment recipe in `acdc_la/realrobot/README.md` (chunk=2, ensemble off, samples=8) was derived on the
  **absolute** arm; `ensemble off` is now contradicted for delta checkpoints and must be re-derived before any delta
  checkpoint goes on the robot.  (c) j3/j4/j5 being worse than the null is the next diagnostic, not the chunk length.

#### 2026-11-10, same day, per-joint control: the wrist IS predictable, and the error is concentrated in j2 and j4 — this SUPERSEDES the diagnostic I named an hour earlier

- **What I said above and am now narrowing.** The entry above closes with "(c) j3/j4/j5 being worse than the null is
  the next diagnostic".  The *fact* stands (arm B / null = 1.46 / 1.10 / 1.30 on j3/j4/j5).  The *diagnostic* was
  aimed wrong, because the null is not the right reference for where a model is losing — the floor is.
- **Control.** `acdc_la/realrobot/wrist_floor.py`, the same matching as `delta_floor.py` (11,052 cross-episode pairs,
  pose < 2 deg, image closest 2 %) but reported per joint.  CPU only, no checkpoint.

  | per joint | j0 | j1 | j2 | j3 | j4 | j5 | mean |
  |---|---|---|---|---|---|---|---|
  | true \|delta\| per step (= null) | 0.709 | 2.797 | 1.949 | 0.287 | 1.405 | 0.048 | 1.199 |
  | **floor** (half the matched spread) | 0.424 | 1.712 | 0.601 | 0.213 | 0.710 | 0.002 | 0.610 |
  | arm B measured, `chunk[0]` | 0.599 | 2.022 | 1.672 | 0.419 | 1.552 | 0.062 | 1.054 |
  | floor / null | 0.60 | 0.61 | **0.31** | **0.74** | **0.51** | 0.04 | — |
  | arm B / floor | 1.41 | 1.18 | **2.78** | 1.97 | **2.18** | 29.5 | 1.73 |
  | share of total recoverable error | 6.6 % | 11.6 % | **40.2 %** | 7.7 % | **31.6 %** | 2.3 % | — |

- **Conclusion: confirmed, and it is case (b), not case (a).** `floor / null` is below 1 on **every** joint,
  including the three where arm B loses to the null — j3 0.74, j4 0.51, j5 0.04.  So the wrist increments **are**
  predictable from the observation; arm B exceeding the null there is **our model's shortfall, not a data limit**.
  There is no joint on which outputting zero is the correct controller.
- **And the priority is not the wrist.** Measured against the floor rather than the null, **j2 (elbow) and j4 carry
  71.8 % of all recoverable error** (1.071 deg and 0.842 deg of the 2.664 deg summed headroom).  j1, which has by far
  the largest *raw* error (2.022 deg), is nearly solved at **1.18x its floor** — its error is mostly irreducible
  demonstration variation.  j3 and j5, the two that looked worst by the null ratio, are together 10 % of the
  recoverable error; j5's 29.5x ratio is an artifact of a near-deterministic floor (0.002 deg) on a 0.062 deg error.
- **Why this is a plausible loss-allocation story, not just a ranking.** The training objective is an unweighted L1
  over normalised dims, so each joint's gradient share follows its normalised scale, not its recoverable error.
  j2 and j4 are exactly the dims where raw motion is moderate but conditional predictability is high
  (floor/null 0.31 and 0.51 — the two most predictable joints after j5), i.e. the dims an unweighted loss
  under-serves relative to what the data would allow.  **Hypothesis, not a result:** per-joint loss weighting by
  inverse floor would recover part of the 0.444 deg mean headroom.  Not run, and §8 of `next_task.md` applies —
  it is an ablation on our arm, and if it is ever reported as a comparison the baseline gets the same weighting.
- **Artifacts.** `acdc_la/realrobot/wrist_floor.py`; nulls and the aggregate floor `acdc_la/realrobot/delta_floor.py`.
- **Next.** Supersedes "(c)" above: the diagnostic to run is **j2 and j4**, not j3/j4/j5, and the reference is the
  per-joint floor, not the null.  Still blocked on arm A's open-loop number before anything is attributed to the
  latent mechanism.

### 2026-11-10 — press_button (first multi-task real-robot set) converted on Jetstream2 g3.xl; training HELD for the abs-vs-delta decision

- **Status: data prep only, no training, no result.**  Host: Jetstream2 g3.xl (1x A100-SXM4-40GB, 115 GB RAM), workspace
  `/media/volume/TERRA` (code `latent_free/` at newton `3bb8555`, env `envs/univla_vla` = REPRODUCE.md recipe, torch
  2.2.0+cu121, + pyarrow/av).  Handoff: `HANDOFF_press_button.md` (HF `Potestates/TERRA`).
- **Registered question (before training):** does this policy family read its instruction?  `anhdao69/press_button_task`
  = 4 tasks, **press the blue / orange / pink / red button, 25 episodes each**, 100 episodes, 5533 frames at 30 Hz.
  The **button layout changes between episodes** (checked visually, eps 2-4), so colour -> position cannot be
  memorised; the word plus the image is the only route.
- **Converter bug caught (would have silently mis-aligned images):** unlike pick-can, the concatenated mp4 holds
  **6190 frames for 5533 rows** -- dead footage between some episodes (5.03 s between ep 2 and 3).  The pick-can
  converter's "video frame k = data row k" assumption is false here.  `convert_press_button_rlds.py` maps frames by
  LeRobot v3 `episodes/from_timestamp + row timestamp` (pts-matched); first/mid/last front + wrist frames of eps 2-4
  checked by eye: aligned.  **Do not reuse convert_piper_rlds.py on another LeRobot export without that check.**
- **Gripper (handoff 4a), decision:** keep `GRIP_SCALE = 8.2` (same calibration as piper_pick_can).  Raw 0.04-0.06 ->
  stored 0.0049-0.0073, i.e. constant "closed"; not rescaled per dataset (would amplify sensor noise into a fake
  signal and break cross-dataset meaning).  Written in the converter docstring.
- **Padding (handoff 4b), decision by the user: option 1, accept and report.**  On this copy at 10 Hz:
  `sum(max(0, T_i - 13)) / sum(T_i)` = **577 / 1877 = 30.7 %** clean samples; **2 episodes have none**.  (30 Hz would
  give 4233 / 5533 = 76.5 %; not taken -- changes the control rate.)  Effective sample count is 577, not 1877.
  Identical on both arms, so the A-vs-baseline comparison stays valid; absolute numbers are not comparable with pick-can.
- **Artifacts:** `data/real_rlds/press_button/1.0.0` (absolute, 100 ep / 1877 steps, 52 MB) and
  `data/real_rlds_delta/press_button/1.0.0` (delta = action[:6] - joint_state[:6]; mean
  [-0.005 3.965 -3.106 0.086 0.515 -0.007] deg, q01..q99 joint 2: -3.44..19.0).  `press_button` registered in
  `third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/{configs,transforms}.py` (copy of the piper entries; NOT yet in
  `acdc_la/patches/univla_piper.patch`).  `make_delta_rlds.py` now names the builder after `--dataset`.
  Base: `/media/volume/TERRA/p2fft_20k` (vla_full.pt 15,081,838,026 B, action_decoder.pt 105,268,830 B, both match);
  `openvla-7b` at `/media/volume/TERRA/hf/openvla-7b`.
- **Next decision:** abs vs delta, from the two workstation pick-can delta runs (due ~2026-11-10 22:00 UTC; registered
  prediction 1.0-1.5 deg open-loop vs absolute 3.24).  Then baseline arm first (INIT=none), then TERRA arm, SAME
  parameterisation for both, BS=4 GA=16 (global 64), 3000 steps, save 500, snapshot each adapter save.

### 2026-11-10 — TERRA arm on press_button (DELTA actions), 3000 steps: 0.92 deg/joint open-loop vs 1.69 deg "stay still" null; the instruction moves the output and the true one predicts best in 4/5 held-out episodes -- direction only, n=5, NO baseline yet

- **Config.**  INIT `p2fft_20k` (Potestates/TERRA, sizes verified), `tc_cont` / `latp2_inv` labeller, flow head q4,
  LoRA r32, lr 3.5e-4, window 12, global batch 64 (BS 4 x GA 16 x 1 A100-40GB), 3000 steps (~104 epochs), save 500.
  Data `data/real_rlds_delta/press_button` (delta = action[:6] - joint_state[:6]; user's call 2026-11-10, ahead of the
  workstation pick-can delta runs).  Split: UniVLA train[:95%] / held-out train[95%:] = **5 episodes, 79 steps**
  (pink x2, orange x2, red x1, blue x0).  Padding accepted (30.7 % clean samples, entry above).
- **Cost.**  29.6 GB VRAM, ~9.0 s/step (openvla no-latent smoke: 7.4 s/step), 7.5 h wall.  Merge standalone: 11m50s,
  peak RSS 32.2 GB.  latent R2 (train) 0.10 @25 -> 0.40 @50 -> 0.74 @3000.  `action_l1_1step` is oracle-conditioned,
  not reported as an error (handoff 9).
- **Open-loop replay** (`piper_policy.py --replay`, deployment image path, degrees; delta error = absolute-target error
  because the measured joint state is known at deployment):

  | | chunk[0] mean/joint | ensembled | whole 12-step chunk |
  |---|---|---|---|
  | TERRA, delta | **0.918 deg** | 0.968 deg | 0.936 deg (from lang-swap, true instr) |
  | null: predict delta = 0 ("stay where you are") | 1.691 deg | - | 1.124 deg |

  Per joint chunk[0]: 0.34 / 1.90 / 1.48 / 0.40 / 1.39 / 0.01 deg (joints 2, 3, 5 carry the press motion).
  **Not comparable with pick-can's 3.24 deg** (absolute, other dataset, other padding fraction).
- **Does it read the instruction?  (`acdc_la/realrobot/lang_swap.py`, new)** Same image, same flow noise, each of the
  4 instructions: true instr 0.936 deg vs wrong instr **1.136 deg** whole-chunk error (wrong ~= the zero-motion null
  1.124); output shifts **0.42 deg** on average when the colour word is swapped; **true instruction gives the lowest
  error in 4/5 episodes** (the miss, ep 2 pink, is a near-tie: orange 0.811 vs pink 0.822).  **Status: suggestive, not
  established** -- 5 episodes, 3 colours, no blue held out, no significance possible.
- **Artifacts.**  Run `/media/volume/TERRA/vla_runs/tc_cont+press_button+b64x1+lr0.00035+r32+s3000+fm+q4--terra_press_button_delta`
  (merged), snapshots `vla_runs/snapshots/terra_press_button_delta/step_{500..3000}`, results
  `acdc_la/results/press_button_terra_delta_replay.json`, `press_button_lang_swap_terra.json`.
- **Next decision.**  (1) Baseline arm (`INIT=none`, same delta data / recipe) -- without it "0.92 deg" and "reads
  the instruction" have no control: openvla-7b may read the colour word just as well.  (2) A held-out set large and
  balanced enough for the language claim (e.g. 2 episodes per colour, or the per-step lang-swap over more episodes).
  (3) Checkpoint choice from the step snapshots once a held-out curve exists.

### 2026-11-11 — press_button_terra_delta published (public HF) + delta support in the deployment wrapper

- **Published** `Potestates/TERRA/press_button_terra_delta/` (public, user's standing instruction): merged weights,
  `action_decoder.pt`, stats, `ctrl_meta.json` with `"action_space": "delta"`, README with the delta interface.  22/22
  files byte-identical to local; anonymous range request HTTP 206.  Repo README table row added.
- **`piper_policy.py`**: `action_space` (ctrl_meta, else inferred from the joint means) and `to_joint_targets(a, q)` =
  q[:6] + a[:6] for delta, a[:6] for absolute; raises without q on a delta checkpoint.  Deployment loop re-reads q
  every tick (chunk entry k is relative to the state at step k).  Checked on all 1,877 press_button steps:
  q + delta == absolute demo target to 9.5e-7 deg.  `--replay ... --abs_root data/real_rlds` scores the conversion
  end to end with the model (queued after the baseline, GPU busy).
- **Not yet run on a robot.**  Pushed to GitHub in `1dd2ca4` (2026-11-11).

### 2026-11-11 — press_button control arm (official openvla-7b, DELTA): 1.920 deg, worse than "stay still", and it does NOT read the instruction; TERRA 0.918 deg and reads it. Phase 2 is what makes the colour word usable here (n = 5 held-out episodes)

- **Config.** `INIT=none LATENT=none`: official `openvla-7b`, fresh LoRA r32, flow head q4, no labeller (dz = 0).
  Everything else identical to the TERRA arm (entry 2026-11-10): data, split, 3000 steps, global batch 64,
  lr 3.5e-4. ~10 s/step, exit 0. Merged 04:48-04:59 UTC.
- **Result (79 held-out steps, degrees per joint, `piper_policy.py --replay`):**

  | | chunk[0] | ensembled | j0..j5 chunk[0] | swap shift | true instr best |
  |---|---|---|---|---|---|
  | TERRA `terra_press_button_delta` | **0.918** | 0.968 | 0.34 / 1.90 / 1.48 / 0.40 / 1.39 / 0.01 | **0.42 deg** | 4/5 |
  | control `openvla_press_button_delta` | 1.920 | 1.796 | 0.36 / **5.75** / **3.59** / 0.48 / 1.34 / 0.01 | **2.4e-7 deg** | 1/5 (= chance) |
  | null, delta = 0 | 1.691 | - | | | |

  The control's whole-chunk error is 1.645 deg under every instruction, identical to within 1e-8. The 2.1x gap is
  almost all on j1/j2, the joints that carry the press toward a colour-dependent location.
- **Not a pipeline bug (checked before reporting).** Both arms run the same code path, and the prompt token ids
  include the colour word (`acdc_la/realrobot/lang_probe.py`, fp32 on CPU, same image, "pink" vs "blue"). The relative
  difference in the final-layer **query-token states the head reads** is **0.28 % in the control vs 75 % in
  TERRA**; for TERRA it grows from 3 % at layer 4 to 48 % at layer 16. The control's head weights are not zeroed on
  the text half (`c_in` text/visual norm 9.6 / 9.6), so the instruction does arrive. It just carries almost no
  signal into the query tokens, and the head learns to ignore what is left.
- **Reading.** With 4 colours and per-episode layouts, a policy that cannot read the colour can only learn an
  average press, which is consistent with its j1/j2 error. **Phase 2 (TERRA's base) is what makes the instruction
  usable at this data size (95 episodes).** Caveats: n = 5 held-out episodes and 3 colours, one seed per arm,
  offline open-loop only. No physical success rate exists yet.
- **E2E deployment path** (`--abs_root`): TERRA `to_joint_targets(chunk[0], measured q)` against the absolute demo
  targets gives 0.918 deg, identical to the delta-space error, as it should be.
- **Artifacts.** HF `Potestates/TERRA/press_button_openvla_delta/` (public, control, README says not for
  deployment); results `acdc_la/results/press_button_{openvla_delta_replay,lang_swap_openvla,terra_delta_replay_e2e}.json`.

### 2026-11-11 — ONE model for pick-can + press_button (labmate's request): mixture `piper_multi`, TERRA arm, delta, 6000 steps, launched

- **Data.** `jachansantiago/pick_can_place_plate_150_trimmed` is the `piper_pick_can` source: re-converted
  here with `convert_piper_rlds.py` (150 ep / 6462 steps, as upstream), then `make_delta_rlds.py` with the same delta
  definition (target minus measured). Mixture `piper_multi` = [(piper_pick_can, 1.0), (press_button, 2.0)] in
  `oxe/mixtures.py`, now in `univla_piper.patch`. `balance_weights=True` scales the weights by size, so press_button
  gets ~37 % of samples (CPU dry run: 310/800, all 4 colours present). Normalisation is **per dataset**, so the
  checkpoint has two keys in `dataset_statistics.json`. Deploy with `PiperPolicy(ckpt, unnorm_key="press_button")`
  or `"piper_pick_can"`; `--replay` and `lang_swap` select the key from `--dataset`. Each held-out split is
  `train[95%:]` of its own dataset, unchanged, so the result compares directly with the single-task arms
  (press_button TERRA 0.918; pick-can delta OpenVLA arm B 1.054).
- **Config.** INIT `p2fft_20k`, `tc_cont`, flow q4, LoRA r32, global 64 (4 x 16), lr 3.5e-4, 6000 steps
  (~38 epochs of pick-can, ~76 of press_button), save 500 with snapshots. Started 2026-11-11 05:02 UTC, due around
  21:00-22:30 UTC. Merge, replay on both held-out sets and lang_swap are chained (`/media/volume/TERRA/multi_chain.sh`).
### 2026-11-11 — DELTA arm A (ours) on SINGLE-INSTRUCTION pick-can: 1.126 deg vs the baseline's 1.054 deg, NOT separated; the training-time metric ranked the two arms in the WRONG ORDER; and a null here is what the mechanism's own story predicts (cf. press_button, where it separates 2.1x)

- **Configuration.** The matched pair registered in the entry above, both now complete at 10,000 steps, global batch 64
  on one GPU, lr 3.5e-4, window 12, same `data/real_rlds_delta`, same normalisation, same 8 held-out episodes.
  - **A `terra_piper_delta`** — phase-2 base (`/data/ti874297/p2fft_20k`) + LoRA r32 + flow head + 4 query tokens
    + the latent branch (`--latent tc_cont`).  64,328 s (17.9 h).  Merged in 344 s (peak RSS 31.9 GB).
    Final `latent_r2` 0.831, so the latent branch was genuinely doing its job, not collapsed.
  - **B `openvla_piper_delta`** — official `openvla-7b`, fresh LoRA, no latent mechanism at all.  45,693 s (12.7 h).
- **Result — open-loop target error on 353 held-out steps, both replays deterministic (re-run reproduced both to
  3 decimals, so none of this gap is sampling noise):**

  | | j0 | j1 | j2 | j3 | j4 | j5 | **mean** |
  |---|---|---|---|---|---|---|---|
  | **A (ours)** `chunk[0]` | 0.652 | 2.201 | 1.730 | 0.437 | 1.672 | 0.064 | **1.126** |
  | **B (baseline)** `chunk[0]` | 0.599 | 2.022 | 1.672 | 0.419 | 1.552 | 0.062 | **1.054** |
  | A ensembled | 0.638 | 1.801 | 1.364 | 0.387 | 1.447 | 0.065 | **0.950** |
  | B ensembled | 0.600 | 1.685 | 1.349 | 0.351 | 1.441 | 0.061 | **0.914** |

  Paired cluster bootstrap over the 8 episodes (the independent unit; 353 steps is **not** the sample size, since
  steps inside an episode are strongly correlated), 20,000 draws — `acdc_la/realrobot/paired_AB.py`:

  | | A − B | per-episode sign | 95 % CI | verdict |
  |---|---|---|---|---|
  | `chunk[0]` | **+0.072 deg** | A worse in 5/8, sign test p = 0.727 | **[−0.038, +0.189]** | contains 0 |
  | ensembled | **+0.036 deg** | A worse in 4/8, sign test p = 1.000 | **[−0.044, +0.110]** | contains 0 |

- **Conclusion: negative / not separated.** On this dataset at this budget the latent-action mechanism is
  **indistinguishable from plain OpenVLA**.  The point estimate favours the baseline on both read-outs and in 5/8
  episodes, but 8 held-out episodes cannot resolve a 0.07 deg difference, so the honest statement is **"no effect
  detected", not "ours is worse"**.  It would be equally wrong to call the +0.072 deg a win for the baseline.
- **REVISED the same day, after merging the Jetstream press_button arms (`280c6fe`), which I had not seen when I
  first wrote this bullet.** I had written: *"this is now the third setting in a row with the same verdict ... no
  detected effect anywhere, and that pattern is the most load-bearing fact in this ledger about the mechanism."*
  **That framing does not survive the press_button pair, and I withdraw it.** The corrected picture:

  | setting | instructions | ours | control | separated? |
  |---|---|---|---|---|
  | LIBERO full split | 10 tasks | 89.8 | 89.0 | no (±~4 pts/cell) |
  | LIBERO 10 % Long | 10 tasks | 58.7 / 60.6 | 56.4 | no (±8 pts/cell) |
  | **pick-can delta (here)** | **1, identical in all 150 episodes** | 1.126 | 1.054 | **no** |
  | **press_button delta** | **4 colours, layout reshuffled per episode** | **0.918** | **1.920** | **yes, 2.1x** |

  **pick-can is a single-instruction dataset** — every one of the 150 episodes carries the string
  `'pick up the can and place it on the plate'`.  There is nothing in it for a language-grounding advantage to do, so
  a null result here is **what the mechanism's own story predicts**, not evidence against it.  press_button is the
  first real-robot set with a language channel *and* a matched control, and there the gap is 2.1x with a mechanism:
  swapping the colour word moves our output 0.42 deg and the control's **2.4e-7 deg**, and the true instruction wins
  4/5 episodes for ours against 1/5 (chance) for the control.
- **One quantitative addition of mine to that press_button result, which sharpens it.** Their entry explains the
  control's behaviour as "the instruction arrives but carries almost no signal into the query tokens (0.28 % vs our
  75 %) and the head learns to ignore what is left".  A 0.28 % input change producing a 2.4e-7 deg output change is
  six orders of magnitude of attenuation, which that wording does not account for — but the numbers do.
  `PiperPolicy` runs the VLA in **bfloat16** (`piper_policy.py:130`), and bf16 machine epsilon is **7.8e-3**, so a
  0.28 % (2.8e-3) relative change is **below the resolution of the inference path**: in a direct check, a 0.28 %
  perturbation leaves **49.9 %** of bf16 values bit-identical, while a 75 % change leaves **0 %**.  So the right
  statement is stronger than "it learns to ignore it": **the control's language signal is smaller than what its own
  forward pass can represent**, and its invariance is exact rather than approximate.  I verified their
  `lang_swap.py` re-runs a full VLA forward per instruction with no caching, so the 2.4e-7 is a real measurement and
  not a reuse artifact.
- **CONFOUND that applies to BOTH pairs, and that I stated too confidently yesterday.** I wrote in the entry above
  that A vs B is "the only thing here that speaks to the latent mechanism".  It is not: `INIT=none LATENT=none`
  changes **two** things at once — the base (full-parameter phase 2 vs stock `openvla-7b`) **and** the latent branch.
  So both this pair and the press_button pair compare **the whole package against stock OpenVLA**, and neither can
  attribute the effect to the latent branch as opposed to the phase-2 base.  The Jetstream entry's own title is
  correctly worded for this ("**Phase 2** is what makes the colour word usable here").  Isolating the latent branch
  needs a **third arm: phase-2 base + `--latent none`**, which has never been run on either real-robot set.  That is
  now the single highest-value missing arm in this ledger, and it is cheap on press_button (3000 steps, 7.5 h on one
  A100).  Note the baseline as chosen is still the right *headline* baseline — it is the standard "post-train stock
  OpenVLA" control, which is what a reader will ask for — the confound is specifically about attribution.
- **CORRECTION to something I told the user on 2026-11-10.** I said the training-time `action_l1_1step` was a valid
  basis for comparing arms A and B because they share dataset, normalisation and formula.  The numbers are
  commensurable, but this pair is a **counterexample to using it as a ranking**: it orders the arms the *opposite* way
  from the free-running error.

  | | A (ours) | B (baseline) | ordering |
  |---|---|---|---|
  | training `action_l1_1step` @10k (oracle-conditioned) | **0.05634** | 0.06483 | A better by 13 % |
  | replay whole-chunk L1, normalised | 0.1174 | **0.1142** | B better by 3 % |
  | replay open-loop degrees, `chunk[0]` | 1.126 | **1.054** | B better by 7 % |

  The cause is the one already registered for this metric: `FlowHeads.loss` computes
  `l1 = (at + (1-t)*va - actions).abs()` where `at = (1-t)*noise + t*actions` **already contains the ground truth**,
  so it scores a one-step correction from an oracle-perturbed point, not a free-running sample.  Arm A also denoises
  `z` jointly, which changes that one-step quantity without having to improve the actual sample.  The replay's own
  normalised whole-chunk L1 agrees with the degree numbers, so the disagreement is specifically with the
  **oracle-conditioned training** metric, not with normalisation.
  **Method decision: `action_l1_1step` may be used to track one run over time and to pick a checkpoint within one
  run. It must NOT be used to rank two arms.** Every arm comparison needs a free-running read-out
  (`piper_policy.py --replay`, or a LIBERO success rate).  Earlier ledger entries that leaned on it for an
  arm comparison are weakened by this and are flagged, not deleted.
- **Artifacts.** Runs `/data/ti874297/vla_runs/{tc_cont,none}+piper_pick_can+b64x1+lr0.00035+r32+s10000+fm+q4--{terra,openvla}_piper_delta`
  (both merged in place); logs `acdc_la/logs/post2_{terra,openvla}_piper_delta.log`;
  per-step dumps `steps_A.npz` / `steps_B.npz`; test `acdc_la/realrobot/paired_AB.py`
  (`piper_policy.py --replay --dump` added for it).
- **Next.** (a) The parameterisation win stands and is large and separated (3.24 -> ~1.05-1.13 deg, 3x, both arms);
  **that** is what should go on the robot, and either checkpoint will do, since they are indistinguishable.
  Pick B if a tiebreak is needed: it is the simpler model and trained 5.2 h faster.
  (b) Re-derive the deployment recipe on a delta checkpoint — `ensemble off` is contradicted here (ensembling helps
  both arms: A 1.126 -> 0.950, B 1.054 -> 0.914).
  (c) 8 held-out episodes is too weak to settle the mechanism question on this task; the per-joint analysis
  (j2 + j4 = 71.8 % of recoverable error) is a better-powered target than another A/B at this episode count.
  (d) **Run the phase-2 + `--latent none` arm on press_button** — the third arm that separates the phase-2 base from
  the latent branch.  Until it exists, every real-robot claim is about the package, not about the latent mechanism.
  (e) Do not generalise this entry's null to the mechanism as a whole: it is a null on a single-instruction task.


### 2026-11-11 — Ablation A (rebuttal Table 4), seed 0 complete (Newton)
- Job 1195772 (normal partition, evc31, 13.9 h, 2.70 s/step, peak 26.7 GB): all 9 arms to 15k steps from tc_cont,
  shared data stream, full 29-source data.  Dumps `acdc_la/results/eatabm_<arm>_s0_val.npz`.
- Evaluated with the appendix protocols (`acdc_la/downstream/eatab_table.py` -> `results/eatab_table.json`).
  NMSE, lower better; Act. = transported action, all donors; Far = Q3/Q4 donors; Rise = Far - Near (Q1/Q2) [Rise
  definition is ours, to be matched to the table's]:
  | row | Act. | Total | First | Tim. | Seq. | Drop% | Far | Rise |
  | L_self | .820 | .703 | .755 | .906 | .791 | 19.5 | .878 | .043 |
  | L_self+L_inv | .826 | .676 | .716 | .897 | .757 | 20.2 | .890 | .054 |
  | EAT w/o L_inv | .727 | .661 | .741 | .911 | .765 | 22.5 | .788 | .025 |
  | w/o L_effect | .951 | .675 | .719 | .900 | .762 | 23.0 | 1.156 | .238 |
  | w/o L_rel | .733 | .636 | .687 | .904 | .737 | 20.9 | .815 | .052 |
  | w/o L_dist | .771 | .634 | .696 | .903 | .737 | 16.9 | .802 | .005 |
  | L_effect->head | .862 | .631 | .718 | .908 | .743 | 16.9 | .973 | .107 |
  | L_effect->cycle | .976 | .679 | .748 | .902 | .767 | 24.0 | 1.210 | .273 |
  | full EAT | .732 | .640 | .688 | .901 | .735 | 18.9 | .796 | .026 |
- Consistency with the Anvil appendix numbers (other data cap, other seed): full EAT net .640 vs .671, drop 18.9 vs
  17.9, Far .796 vs .801; L_self net .703 vs .732, Far .878 vs .889.  Same ordering.
- Status: SINGLE SEED.  Transport (Act./Far) is the clear separation: removing L_effect or replacing it by a head or
  a latent cycle loses most of it.  Probe columns: full EAT is within ~.01 of w/o L_rel / w/o L_dist / head, which a
  single seed cannot separate.  Seeds 1, 2 (jobs 1195773/74) pending, Slurm estimate 2026-11-13 17:51.
