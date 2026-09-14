# ACDC-LA evaluation funnel (fixed 2026-09-10)

A pretraining variant climbs this ladder; it is promoted only by the criteria below, and the criteria are not changed
after a result is seen.  Every tier maps to an existing probe in this repo.

| tier | what | instrument | cost |
|---|---|---|---|
| **0 Health** | not collapsed, decoder uses z | `common/curve.py` during training: rec, kl_z, kl_u, effective rank of z, swap sensitivity | free (every 10% of budget) |
| **1 Representation** | does z hold what the claim says | `stage1_preserve.probes` on a val dump: R² net / seg1 / **prof** / seq; patch drop; **N_TC** nuisance leak (global reader, `stage2_disentangle.probes`) | ~2 min GPU + CPU |
| **2 Structural / counterfactual** | does the property exist mechanistically | `stage3_transport.probes`: independent-reader transfer vs decoder-only control, replay, E_correct/E_wrong; time-reversal parity (`stage1_preserve.probes`); (registered, not built: same-net-different-profile pairs, same-action-different-nuisance pairs) | minutes, offline |
| **3 Cheap policy-facing** | is z useful for control | `stage3_transport.recovery` few-label curve (1/5/10/25/100 %); `downstream.libero_offline` action-token accuracy / L1 (oracle, chain) | ~15 min GPU |
| **4 Closed-loop** | task success | `downstream.libero_closedloop` (50 starts/task when it matters); VLA fine-tune with our tokens (not built) | hours to days |

## Promotion criteria (equal budget, same seed, vs the pooled baseline `s1_mean_*` unless stated)

Tier 0 → 1: kl_z > 5 nats, effective rank of z > 32, swap sensitivity > 0.2, rec below the pooled baseline + 0.05.
A run failing any of these at 25% of its budget is killed.

Tier 1 → 2 (Stage I claim, *preserve*): ΔR²_prof ≥ +0.03 **and** ΔR²_seg1 ≥ +0.02 **and** ΔR²_net ≥ −0.02 **and** N_TC ≤ baseline + 0.02.
Read at 25% and 50% of the budget: if the profile gap is ≤ 0 at 25% and not growing at 50%, stop the run.

Tier 1 → 2 (Stage II claim, *nuisance*): N_TC ≤ untreated tokenizer − 0.05 **and** N_TI ≥ 0.05 **and** ΔR²_net ≥ −0.02.
(No Stage-II arm has passed this; kl_u must first leave zero.)

Tier 2 → 3 (Stage III claim, *transport*): independent-reader transfer R² ≥ decoder-only control + 0.05 **and** cosine ≥ control + 0.03 **and** clean R² unchanged (frozen tokenizer).  Passed by LATP on three tokenizers (+0.075 to +0.10).

Tier 2 → 3 (Stage IV claim, *portable core + residual*, review 2026-09-13): on the same clips as the LATP checkpoint it starts from,
(a) leakage: N_TC(z^P) ≤ N_TC(z) − 0.03 with N_TI(r) ≥ N_TC(z^P) (the residual, not the core, reads the patch); (b) transfer: independent-reader
transfer R² and cosine of D(s_j, z^P_i, 0) ≥ the continued-LATP control (same extra decoder steps on the full code) − 0.01; (c) the core keeps the
action: ΔR²_net(z^P vs z) ≥ −0.02.  Downstream (Tier 4 test): the policy predicts z^P only and feeds D(s_t, ẑ^P, 0); latent R² of the
predicted core must exceed the full-code policy's 0.50 and chain L1 must drop, or the split changed nothing the policy can use.

Tier 3 → 4: few-label recovery at 10% labels ≥ baseline, **and** offline chain L1 ≤ baseline − 0.01.  Only the arms that pass go to closed-loop; at most three (pooled baseline, best ACDC, best transport) plus UniVLA.

## Status against the funnel (2026-09-10)

* Stage I (`s1_acdc_1ep` vs `s1_mean_1ep`): Tier 1 **passed** (prof +0.068, seg1 +0.054, net −0.012, N_TC 0.14 vs 0.23 for the augmentation control; baseline N_TC not yet read on the mean tokenizer).
* Stage III (`s3_latp_*`): Tier 2 **passed**; Tier 3 **failed** on LIBERO-spatial through every interface (raw z, decoder hidden, decoder OUTPUT effect: oracle-z L1 0.344 vs 0.346, chain 0.421 vs 0.426); the oracle->predicted-latent gap (+0.08 L1) dominates everything the decoder can do.  The claim that survives is the Tier-2 property; the downstream policy is the bottleneck (09-12).
* Stage II: Tier 1 **failed**.  u collapsed by KL price (20 runs); at a fair price u holds 13 nats the decoder uses for a 1% reconstruction change (u-shuffle 09-12) and still no linearly readable nuisance.  Decision: u = optional residual; the method keeps only z_clean ~ z_patch; the paper reports the KL-price and quantiser findings as diagnostics.
* Full-data runs (`s1_acdc_full`, `s1_mean_full`, `s3_latp_full`) emit the Tier-0/1 curve every 10% of budget; the 25% / 50% stop rules above apply to them.
* Stage IV (`s4_pcrr`, submitted 2026-09-13): portable core z^P (192) + residual r (64) on the frozen `s1_acdc_1ep` code, decoder from `s3_latp_1ep`,
  10k steps; control `s3_latp_1ep_cont` (LATP continued 10k steps on the full code).  Evaluations: Stage-I probes on Z = z^P, Stage-II probes with
  U = r, Stage-III transfer with the residual zeroed, then the LIBERO policy recipe with `--tokenizer s4_pcrr --transport s4_pcrr`.
  Result (job 20627479): (a) N_TC 0.106 -> 0.035 but N_TI(r) -0.001 (dropped, not routed); (b) transfer 0.288 vs control 0.321 **fail**;
  (c) action from the core 0.261 vs 0.324 **fail**, the residual reads the action at 0.114.  **Failed at Tier 2**; downstream (job 20642928): latent R2 0.38 vs 0.46 for the full code, chain L1 0.427 unchanged, closed-loop 7 %: **failed at Tier 4** as well.  Closed.
* VQ gate (`downstream/libero_vqgate.py`, job 20660076, 2026-09-14): post-hoc per-cell / product quantisation of the ACDC code as
  the policy target.  Rule: chain L1 <= continuous - 0.015.  Best 0.425 vs 0.430 (-0.005); oracle loses 0.03-0.04.  **Not passed**;
  a deployment prior needs a jointly learned discrete bottleneck, not started before the deadline.
