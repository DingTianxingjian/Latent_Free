# Experiment review: preservation, invariance/structure, transport

Reviewed 2026-09-06. This is an audit of existing results, logs, and evaluation code;
no new training or checkpoint evaluation was launched. The project ledger is
[LEDGER.md](LEDGER.md). Source directories are siblings: `latent_free/`,
`claude_survey/`, and `codex_survey/`.

The evidence supports preserving spatial and slow temporal evidence and testing
transport explicitly. It does not yet establish the proposed combination of
structured pooling, algebra, VICReg, and a dynamics-prior swap loss. The most
consequential open question is whether prior + cycle aligns the physical meaning
of a code across states, beyond making it plausible and recoverable.

## 1. Preservation has the strongest replicated evidence

All 30 pooling JSONs and their training/evaluation completion logs are present.
Bridge-big: 4,000/1,000 train/validation episodes, 134,464 training windows;
12,000/5,000 frozen-probe windows; five consecutive frames, four deltas;
32 continuous latent dimensions, 456,720 parameters, 6,000 steps, batch 128,
three training seeds per configuration. Ridge tuning uses inner training episodes.

Recomputed three-seed means from the raw JSONs:

| Pooling | Endpoint net xyz R² | Endpoint profile xyz R² | Sequence net xyz R² | Sequence profile xyz R² |
|---|---:|---:|---:|---:|
| Global | 0.1999 | 0.0798 | 0.2252 | 0.0894 |
| Spatial quadrants | 0.2565 | 0.0782 | 0.2717 | 0.0817 |
| Ordered temporal | 0.2390 | 0.1090 | 0.2203 | 0.1004 |
| Temporal DCT | 0.2173 | 0.1023 | 0.2164 | 0.0993 |
| Learned attention | 0.2075 | 0.0803 | 0.2528 | 0.0917 |

Net is summed action; profile is the mean-removed four-step action sequence.
Their R² values have different targets and denominators. In the separate frozen
linear audit, DC+AC1 improves profile by +0.0391 over DC and +0.0498 over equal-width
DC+shuffled-AC1. Its three repeats sample the same episode split.

Implications:

- Quadrants × (DC+AC1) is a well-motivated architecture, but no completed hybrid
  training or hybrid pair-mining result was found. Neither its quality nor a
  lossless-information claim follows from the single-axis experiments.
- Sequence reconstruction alone did not resolve the trade-off at this architecture,
  loss normalization, and budget. This is not a general impossibility result for
  sequence decoders. Endpoint reconstruction sums normalized deltas, while sequence
  reconstruction averages per-step errors; equal beta does not equate their effective
  reconstruction/KL trade-off.
- Preservation does not imply robustness. Under sequence reconstruction, spatial
  pooling has clean/patch net R² 0.2717/0.2023 (25.5% drop); global has
  0.2252/0.1879 (16.5% drop). Spatial retains more absolute action signal in both
  conditions while suffering a larger relative nuisance penalty.
- Low slot redundancy is not sufficient. Sequence DCT has effective rank 26.15
  and cross-slot R² 0.0113, versus global 6.84 and 0.9944, but lower net R².

Sources: [raw pooling results](../codex_survey/results/),
[pooling implementation](../codex_survey/experiment.py),
[quick temporal contrasts](../codex_survey/results/quick_dct_effects.csv),
[preparation log](../codex_survey/logs/cx_poolprep_20148046.out).

## 2. The completed photometric VICReg screen is negative

The master ledger still described job 20314851 as pending. Its log reaches step
3999, prints the result, and ends with `INTERVENTION SSL DONE` on September 2.
Matched Bridge-622 screen: seed 0, 4,000 steps, batch 96, continuous 8×32 query
latent over absolute V-JEPA frame tokens, endpoint feature reconstruction.

| Arm | Clean xyz R² | Clean cont6 R² | Moving-patch cont6 R² | Photo A/B latent cosine |
|---|---:|---:|---:|---:|
| Baseline | 0.0166 | 0.0097 | 0.0054 | 0.9518 |
| Photo cross reconstruction | 0.0213 | 0.0128 | 0.0103 | 0.9809 |
| Photo cross reconstruction + VICReg | 0.0017 | -0.0012 | -0.0039 | 0.9638 |

This implementation does not produce a useful linearly action-decodable latent.
High paired-view cosine alone is not a success criterion. It does not establish
complete representation collapse or failure of all VICReg implementations: no
held-out latent rank, state probe, or nonlinear action probe was saved here.

The optimization also needs calibration. VICReg is applied directly to all 256
latent coordinates, with outer weight 0.05 and internal weights 25/25/1. At the
last logged batch its weighted contribution is 0.3365, versus reconstruction
0.2883. That is a diagnostic clue, not evidence that this weight caused the result.
Invariant state information can satisfy the view match and variance constraint;
the loss does not specifically select action information.

There is a positive but different screen: moving-patch InfoNCE at 8,000 steps
changes clean xyz R² 0.0166→0.1739, clean cont6 0.0096→0.1007, and patch cont6
0.0057→0.0943. It differs from photo VICReg in nuisance, loss, budget, and training
views. It cannot establish InfoNCE > VICReg. Initialization was not explicitly
seeded in the old IRLA script; treat it as one run per arm.

The photometric feature gate explains why lighting may be a weak intervention:
a separately nested, fixed clean probe loses only 1.6–2.3% net xyz R² under the
photo views, versus 15.1% under the moving patch. Feature-difference cosine alone
had ranked their action damage incorrectly.

Evaluation caveat: `intervention_ssl.py` and `irla.py` call
`probes_torch.ridge_probe_t` independently for each evaluation condition. That
function selects alpha using that condition's evaluation labels. The weights use
clean training rows, but these are not one fixed probe applied to every condition.
The earlier ledger's stronger fixed-probe description is withdrawn. The near-zero
VICReg scores remain a negative screening result even with this optimistic tuning.

Sources: [VICReg CSV](../claude_survey/results/intervention_photo_vic_s0.csv),
[completed log](../claude_survey/logs/cs_intssl_20314851.out),
[screen implementation](../claude_survey/intervention_ssl.py),
[probe selection](probes_torch.py),
[IRLA completed comparison](../claude_survey/logs/cs_irla_19694534.out),
[feature gate with nulls](../claude_survey/results/photo_feature_gate_nulls.json).

## 3. Correct donor supervision works; recovery alone has not

Matched operational arms start from the same `s2_base` and fine-tune for 3,000
steps, batch 12, one seed. Evaluation uses 1,062 held-out endpoints; matching
same/cross-scene action distances leaves 689 endpoints.

| Arm | Cross-scene action-matched transport | Donor reconstruction MSE |
|---|---:|---:|
| Base | 0.5070 | 0.18613 |
| Continued self reconstruction | 0.5114 | 0.18125 |
| Oracle positive | 0.6558 | 0.16883 |
| Oracle positive + ranking | 0.6628 | 0.16879 |
| Oracle ranking only | 0.5900 | 0.17554 |
| Frozen-encoder cycle | 0.4955 | 0.18775 |
| Pre-VQ pseudo pairs | 0.5080 | 0.18124 |
| DINO mutual/margin pseudo pairs | 0.4678 | 0.18998 |
| Direct donor spatial-effect matching | 0.3636 | 0.23033 |

The continued-reconstruction arm is an extra-steps control, not training on random
swaps. Random donor codes are a separate evaluation control. The score is
`(MSE_random − MSE_donor)/(MSE_random − MSE_self)`; raw MSE supports the oracle
improvement, so it is not only a changing denominator. It is not a success rate.

Oracle training uses standardized summed seven-dimensional recorded actions.
Despite descriptions saying “7-DoF”, the evaluation donor selector actually uses
only summed xyz (`A = tab['act'][:, :3]`). Its 0.903 bin agreement and 0.9796 cosine
therefore validate translation matching, not matching rotation, gripper, or temporal
profile. Equal net action can conceal different sequences. Those axes need separate
instruments for the proposed temporal latent.

Cycle was tested more strongly than merely using a jointly trained inverse: its
teacher encoder/codebook is frozen and two donors are decoded on one target.
The completed arm still shows no transport gain at this budget. Independently,
the base already recovers real donor codes at 0.6973 across random scenes versus
0.7799 on self cycles; chance is 0.1584. Recoverability and semantic transport are
different measurements. This does not prove cycle has zero headroom or that
cycle + a new prior must fail.

The donor-feature effect arm also failed. Matching a donor's four-quadrant DINO
change to the destination change is not a validated coordinate system for the
same physical action across states. Treat that mechanism as a hypothesis; the
measured result is simply negative for this implemented loss.

DINO pair selection reached 0.5601 action-bin agreement at 3.66% coverage.
Training on that set did not help. Its -0.041 post-VQ action-R² change is below
the later estimated single-seed resolution and must not be used as independent
proof of degradation. Transport has its own unmeasured seed floor.

The often-quoted “18% scene / 82% state” calculation uses the residual to score 1.
That is a descriptive ratio, not a causal variance decomposition: the oracle
pairing is approximate and the achievable ceiling is unknown.

Sources: [base transport](../claude_survey/results/transport2.csv),
[transport results](../claude_survey/results/),
[training code](../claude_survey/transport_train.py),
[evaluation code](../claude_survey/transport.py),
[cycle log](../claude_survey/logs/tr_cycle_20245783.out),
[cycle audit](../claude_survey/results/cycle2.csv),
[DINO pair audit](../claude_survey/results/transport_dino_audit.json).

## 4. A pretrained flow decoder was already tried

`dit_lam.py` freezes a Wan 5B video generator and trains a 53.3M IDM/conditioning
adapter using flow matching. All three logs finish 4,000 steps and evaluation.
Bridge-622, 16,945 training windows, batch default 16, one initialization per arm;
full invocation metadata and the NCE coefficient are not saved in the result CSVs.

| Variant | Action xyz R² | Action cont6 R² | State PCA-64 prediction R² |
|---|---:|---:|---:|
| Original | 0.0153 | 0.0050 | 0.4830 |
| High noise | 0.0118 | 0.0049 | 0.4727 |
| High noise + NCE | 0.0220 | 0.0103 | 0.5018 |

A pretrained generator alone did not make the inferred latent action-like in this
screen. This was conditioning-adapter training through a frozen generator, not the
new proposed trainable feature-flow decoder with a separate frozen prior and swap.
Its negative result does not settle that new objective.

The old `GENERATOR-NEEDS-Z` test is also insufficient: in the NCE run real/shuffled/
zero flow losses are 0.2223/0.2292/0.9081. Zeroing context changes its distribution.
Moreover, real/shuffled/zero calls advance a dedicated noise/timestep generator;
`torch.manual_seed(0)` inside the loop does not reset that generator. There is no
matched-noise uncertainty estimate for the small real-vs-shuffled gap. Do not claim
sample-specific control from this printout. Future tests must hold target, noise,
timestep, and nuisance residual fixed while changing only donor z.

Sources: [DiT implementation](../claude_survey/dit_lam.py),
[original result](../claude_survey/results/ditlam_bridge.csv),
[high-noise result](../claude_survey/results/ditlam_hiN.csv),
[NCE result](../claude_survey/results/ditlam_hiN_nce.csv),
[NCE completion log](../claude_survey/logs/cs_ditlam_19715193.out).

## 5. Two mathematical issues in the new method

The following are deductions, not experimental results.

First, a dynamics prior and an exact cycle still allow different action conventions
in different states. Consider two locally disconnected state regions with equally
plausible left/right changes. Let a region-specific sign be σ(s)∈{−1,+1}, constant
along the local trajectories used for algebra training. Define

`E(s, Δ) = σ(s)Δ`, and `D_delta(s, z) = σ(s)z`.

Self reconstruction and code recovery are exact. Both output directions have the
same prior probability. Local addition/reversal also hold, and the construction
can ignore photometric nuisance entirely. Yet moving a code between regions of
opposite sign reverses its physical effect. A fixed encoder does not remove this
already learned state convention. Thus prior + cycle is a reasonable transport
hypothesis, not semantic transport supervision with identified targets.

There is also a dataset-support issue:

`p_data(Δ | s,c) = ∫ p_env(Δ | s,a) π_data(a | s,c) da`.

A prior learned from demonstrations includes the demonstrator's action choices.
It can penalize a physically valid counterfactual simply because that action was
rare in that state/task. Independently judge feasibility, donor semantics, and
coverage; do not equate the most likely dataset transition with the correct swap.

Second, algebra must respect the descriptor. For literal DCT coefficients of frame
deltas, reversing the clip gives `Δ_rev[t] = −Δ[T−1−t]`, hence

`c_k(reversed clip) = (−1)^(k+1) c_k(original clip)`.

DC changes sign; AC1 keeps its sign. Imposing `z_reverse = −z` on the entire
literal DC+AC1 descriptor conflicts with this structure. Concatenating two windows
also changes the temporal basis and duration: AC1 of the concatenation is generally
not the sum of the two AC1 coefficients. Define a duration-aware composition and
basis-aware inverse, or reserve an algebraic operator channel separate from the
temporal-profile channel. This formula concerns raw fixed-basis deltas; the old
trained DCT pooler first applies a nonlinear token MLP and is not that raw descriptor.

Fixed image quadrants similarly preserve location evidence without supplying a
state-independent physical coordinate frame. The shared physical effect can move
to another quadrant when the object/robot/camera moves. Reconstruction can use a
state-dependent spatial realization; copying a donor's spatial map is an additional
assumption that needs testing.

The paper positioning in the proposal is supported by the checked primary sources:
[ALAM](https://arxiv.org/abs/2605.10819) places joint latent/action flow matching
downstream of its frozen transition encoder;
[Why Latent Actions Fail §4.4](https://arxiv.org/html/2605.20223v1#S4.SS4) assumes
synchronized endogenous transition tuples across exogenous contexts. Neither gives
a theorem for arbitrary cross-endogenous-state swaps.

## 6. Close the loop to the VLA consumer early

An existing offline ridge policy proxy already exposes a gap. On `s2_base`, with
9,863 labelled training rows and 5,016 test rows, true future-derived z→action
scores 0.3181 xyz R². Predicting z from the current frame and then decoding action
scores 0.0759, versus direct current-frame→action 0.1196. Current-frame→z itself
scores 0.0360. At 1,000 labels, z-route/direct is 0.0623/0.0932.

These are one-split, linear, language-free offline proxies, not rollout results or
a test of the new method. They motivate an early matched-label policy test with
language and appropriate history, alongside the latent probes. Better future-derived
representations need not be easier for a deployable policy to predict.

Source: [policy CSV](../claude_survey/results/policy_eval.csv),
[policy protocol](../claude_survey/policy_eval.py).

## 7. Recommended next experiments, not launched

1. Validate quadrants × (DC+AC1) at 32 total latent dimensions and the existing
   6,000-step/three-seed budget. Compare spatial-only, temporal-only, global, and
   matched-width shuffled-AC1 controls. The old net≈0.27/profile≈0.10 joint gate
   is a useful initial target, not an established optimum. Report rotation/gripper
   separately and absolute nuisance performance beside relative drops.
2. Compare no invariance, cross reconstruction, InfoNCE, and VICReg on that same
   non-degenerate model, with the same intervention, budget, and evaluation. Fix
   the probe protocol first; tune alpha on inner clean training episodes and freeze
   the probe for all nuisance conditions. Measure state/nuisance readout, action
   content, latent variance/rank, and per-loss gradient scale.
3. Before large robot runs, use a controlled environment with available
   counterfactuals kept strictly for evaluation. On one architecture compare
   self-only, cycle-only, prior-only, and prior+cycle. Add oracle-positive training
   as a separately labelled positive control. Check whether the learned prior
   admits rare-but-valid donor actions and whether donor semantics survive swaps.
4. Add with/without algebra after defining the inverse/composition for temporal
   coefficients. Report algebra error, representation content, and transport
   independently. Use multiple destinations per donor and multiple donors per
   destination, with shared noise/residual, and an independent evaluator.
5. Advance to a matched-label VLA learning curve and rollout test only after the
   mechanism gates pass. When actions are used to select pair thresholds or loss
   hyperparameters, disclose this as action-informed development even if gradient
   training itself uses only video.

Additional ledger correction: job 19690719 finished its λ=3 IRLA arm and printed
the exact CSV result before timing out in its subsequent λ=10 arm. The previous
ledger attributed the timeout to λ=3. Its recovered xyz/cont6/patch-cont6 values
are 0.0379/0.0169/0.0147 at 10,000 steps. This is not a matched-budget comparison
to the 8,000-step λ=1 arm.
