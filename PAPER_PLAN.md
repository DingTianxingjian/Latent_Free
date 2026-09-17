# Paper plan: ICLR submission, deadline 2026-09-24

Written 2026-09-16 (night), after the user's decision that **the main table comes first and ablations are
deprioritised**.  The ledger (`LEDGER.md`) is the source for every number here; this file only says where each
number goes and what is still missing.  Status: ✅ in hand · ⏳ job running/queued · ❓ decision pending · ✗ not planned.

## Shape

A half-diagnostic paper: controlled diagnostics of latent action models (UniVLA line), then our method, then a
simulation-only positive result.  No real robot.

**Comparison rule (fixed 2026-09-12):** every UniVLA number comes from THEIR recipe trained on OUR capped mixture at
OUR budget (`s2mix_1ep`, 29 OXE sources).  The released weights are a reference row in the probe tables only.
The VLA recipe is identical across arms (openvla-7b + LoRA r32 + action head, 15k x 64, their evaluator,
500 episodes per suite); only the latent labeller changes.

**One-sentence story.**  With the VLA recipe fixed, full-data LIBERO cannot tell two very different latent
action tokenizers apart; controlled measurements show what a latent action should carry (temporally structured
action evidence, robustness to visual nuisance, recoverability from what the policy sees); a transport-consistency
objective that trains the encoder improves those measurements; and the gain is tested where the benchmark has
headroom: post-training on 10 % of the demonstrations.

## Sections

1. Introduction
2. Setting and controlled protocol (the UniVLA pipeline; what is held fixed; OXE probe protocol; LIBERO protocol)
3. Diagnostics
   - 3.1 The benchmark does not measure the latent (T2)
   - 3.2 What latent actions encode: action content vs nuisance (T3, F4)
   - 3.3 Spatial granularity: patch-level evidence is the most nuisance-sensitive (F3)
   - 3.4 Recoverability at deployment (T5)
4. Method: ACDC evidence + structured tokenizer + LATP (F2)
5. Results: main table (T1, F1); LATP attribution (T6)
6. Limitations (below)

## Tables

| id | content | status |
|---|---|---|
| **T1 (main)** | LIBERO success at **10 % demos**, 4 suites x {UniVLA latent, Ours}, avg; per-task in appendix | Spatial UniVLA **84.2 %** ✅; the other 7 cells ⏳ (jobs below) |
| T2 | Benchmark insensitivity: full-data Spatial UniVLA 94.8 vs ACDC-PQ 94.4 (per task); offline chain L1 best latent 0.421 vs no latent 0.433 | ✅ |
| T3 | Latent content on OXE val (net / seg1 / prof / seq / distractor drop): UniVLA released (ref), UniVLA same-data 1 ep and 2 ep, pooled, old ACDC, tc_acdc, Ours | ✅ (`tc_acdc` 0.223 / 0.199 / 0.074 / 0.157 / 31.9 %, below `tc_cont`; ledger 2026-09-17) |
| T4 | Spatial granularity 256 / 64 / 16 / 4 cells | ✅ (ledger 2026-09-16, corrected drop column); may become F3 instead |
| T5 | Deployment recoverability: bits recovered (UniVLA vs ACDC-PQ, n = 1500); frozen-feature latent R2 old ACDC 0.50 vs latp2_inv 0.73 | ✅; the same measurement on the trained main-table VLAs needs a continuous-arm version of `token_bits.py` (after T1 runs) |
| T6 | LATP attribution: tc_cont / +15k reconstruction / +anchored transport / +invariance | ✅ on the tc_cont base; if Ours moves to the tc_acdc base, the +15k control must be rerun there (1 GPU, ~2 h) |
| App. | LATP failure-mode instruments (effect_margin, replay_cos, z_erank, ctx_sens, mmd); per-task demo counts of the 10 % split; per-task success | ✅ / ⏳ |

Known reading of T3 that the text must not overstate: against same-data UniVLA at 2 epochs, `latp2_inv` is better on
net (0.329 vs 0.261) and seg1 (0.283 vs 0.249), **worse on profile** (0.091 vs 0.139), and tied on seq (0.236 vs
0.237) and distractor drop (17.9 vs 18.2 %).  Budgets differ (their 256k steps vs our 65k + 15k).

## Figures

| id | content | status |
|---|---|---|
| F1 | Teaser: per-task Spatial success at 100 % vs 10 % demos (UniVLA), with Ours at 10 % | needs Ours@10 % Spatial ⏳ |
| F2 | Method diagram: 5-frame window -> ACDC evidence [N, A1] on a 4x4 DINO grid -> structured tokenizer -> LATP (anchored effect, MMD, RTC, invariance) -> VLA labels | drawing only |
| F3 | Spatial granularity curve: r2_net clean and under distractor vs cells | ✅ data |
| F4 | What temporal pooling loses: profile R2 pooled 0.017 / ACDC 0.085 / UniVLA same-data 0.125 | ✅ data |
| F5 | Data-efficiency curve (10 / 25 / 100 %) | ✗ unless T1 is done by ~9/20 |

## Experiments (submitted 2026-09-16 night)

VLA training is 4xH100, ~5.6 h; evaluation is 1 GPU, 2-6 h (Long up to 12 h limit).  Evaluations depend only on their
training job.  The libero_10 smoke + load test (20783066 -> 20783067) runs independently as an early warning; the
evaluators are Python modules read at run time, so a failure it finds can be fixed before any evaluation starts.

| cell | train job | eval job |
|---|---|---|
| Spatial, Ours (`latp2_inv`) | 20754114 (re-targeted by override) | 20783070 |
| Object, UniVLA | 20775018 (re-targeted by override) | 20783071 |
| Object, Ours | 20783072 | 20783073 |
| Goal, Ours / UniVLA | 20783074 / 20783076 | 20783075 / 20783077 |
| Long, Ours / UniVLA | 20783078 / 20783080 | 20783079 / 20783081 |
| smoke (libero_10, both arms) -> load test | 20783066 -> 20783067 | |
| LATP on clean ACDC (`latp2_acdc_inv`, from `tc_acdc`) -> probes + z-stats | 20783064 -> 20783065 | |

## Open decision ❓

**Which checkpoint is "Ours".**  `latp2_inv` is built on `tc_cont` (encoder sees F0, 4 x 64 tokens), which is not
the clean ACDC definition (no F0 in the encoder, 8 x 32 structured slots, `tc_acdc`).  Gate 20783065 probes
`latp2_acdc_inv` next to `latp2_inv`.  Proposed rule: if the clean-ACDC version is within ~0.01 on net / seg1 / seq
and not worse on distractor drop, re-target the not-yet-started Ours runs to it (override file) and rerun Spatial;
otherwise keep `latp2_inv` and describe the `tc_cont` architecture in the method section as it is.

## P2, only if T1 is on track by ~9/20

25 % demos on Spatial (both arms, for F5); no-latent control at 10 %; a second evaluation seed per T1 cell;
`tc_cont`@10 % (LATP ablation at the policy level).

## Limitations to state

Simulation only; one training seed per VLA arm (500 episodes, standard error ~1.6 points at 84 %); all tokenizers
read a shared 4x4 DINO grid (spatial curve is linear-probe evidence, not patch-level modelling); capped OXE mixture;
the continuous interface is a regression head, not flow matching (LAFP / RotVLA); the 10 % split is the first 10 %
of the tfds train split with uneven per-task counts (1-9), identical for both arms; action normalisation statistics
come from the full split for every arm.
