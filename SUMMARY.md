# Do visual feature differences encode actions?

> Project-wide live experiment ledger: [`LEDGER.md`](LEDGER.md). This file retains
> the detailed frozen-feature experiments; newer transport, pooling-matrix, DC/AC1, and
> intervention decisions are summarized in the project-level ledger.

**LIBERO-spatial · frozen encoders + linear probes · no model training**
Run 2026-07-22, commit `0c340e0`, seed 0, Anvil `ai` partition (1×H100).

---

## Three takeaways

**1. Most of what a two-frame encoder decodes is already in *one* frame.**
Give a probe both frames, `[f(o_t), f(o_{t+s})]`, and then take away the
transition by pairing f(o_t) with a *random other frame* from the split. That
shuffled control scores the same as the single frame — within **0.016** on every
reported group, 0.022 on any per-axis row — so doubling the feature width buys
nothing on its own. The genuine two-frame contribution is only **+0.04–0.08 R²
at stride 1 and +0.14–0.17 at stride 4**, i.e. **74–92% of a two-frame all-6
score is already single-frame state**, in every feature space tested — now across
**three distinct training objectives** (DINOv2 self-distillation, the Wan
reconstruction VAE, and V-JEPA2 masked prediction), so this is a property of the
scripted-demo data, not of any one encoder. Any LAM evaluation without this
control will overstate what its latents know. *(Revised after Exp 4 — the original version of this claim, "Δf decodes
worse than a single frame", was partly an artifact of the CLS token; see below.)*

**2. A variance-greedy bottleneck keeps the action when the *action* owns the
variance and discards it when a distractor does — the theory's mechanism is real
but conditional.**
Tested against the correct null — project Δf onto its top-k PCs vs a *random*
k-dim subspace vs the *bottom*-k PCs, refit, measure R². **On clean data the
theory looks false:** top-k > random-k > bottom-k in every feature space and
stride (42/42 rows), top beating random **1.3–3.9×** at k ≤ 128, the low-variance
tail nearly empty (bottom-128 R² ≈ 0.01–0.07 of a full 0.19–0.84). Action lives
in the *high*-variance directions, so a top-k bottleneck is the best linear
bottleneck of its width. (The earlier "5–30× *below* chance" was an artifact of
the subspace-energy metric — `w_raw = w_std/x_sd` tilts the measured direction
into the tail; the metric-free refit shows the tail is empty.) **But add a
distractor that owns the variance and it flips:** under a *playing* 64%-pixel
border video, top-8 xyz R² collapses to **≈ 0.001 — below the 0.003 a random
subspace gets** — while the frozen-border control (same occlusion, no motion)
keeps top-k winning **5×**. So variance-greedy compression discards action
exactly when task-irrelevant *motion* dominates the variance. Clean scripted
demos never trigger it (the robot owns the variance); the real world can. The
theory isn't wrong — it is a statement about **who owns the variance**.

**3. Task-irrelevant motion contaminates Δf far more than it contaminates a
frame — and the frozen controls say why.**
In patch space (Exp 2A, stride 4, xyz R²), an 8%-pixel *moving* patch cuts Δf
from **0.839 → 0.617**; freezing the same patch — same occlusion, no motion —
leaves Δf at **0.746**. So of the 0.222 Δf drop, occlusion accounts for 0.093 and
distractor *motion* for the other **0.129**. The single frame barely notices the
motion: it falls only 0.725 → 0.632 under the moving patch, and its frozen-patch
score (0.575) is no higher — so it is *differencing* that turns exogenous motion
into error, not the occlusion. Under the 64%-pixel border the split is dominated
by occlusion instead (that condition deletes 64% of the pixels) — see caveats.

---

## Exp 1 — action decodability from a frozen DINOv2 ViT-B/14 (clean)

| stride | features | trans xyz R² | rot R² | all-6 R² | gripper bal.acc |
|---|---|---|---|---|---|
| 1 | Δf | 0.194 | 0.069 | 0.132 | 0.646 |
| 1 | f(o_t) | 0.551 | 0.263 | 0.407 | 0.890 |
| 4 | Δf | 0.549 | 0.236 | 0.393 | 0.802 |
| 4 | f(o_t) | 0.575 | 0.283 | 0.429 | 0.893 |

Ridge alpha = 100 selected in every row; logistic C ∈ {0.1, 100}.
Stride-4 Δf translation R² = 0.549 lands inside the 0.5–0.8 band the brief
predicted, so the pipeline is behaving. Stride 1 is simply a much harder target:
one-step EEF deltas move the image by very little.

## Exp 4 — was it the CLS token? (follow-up, and it partly was)

Exp 3 suggested Exp 1 was measuring the CLS bottleneck rather than frame
differencing. So: the same protocol over four feature spaces × four inputs.
`pmean` is the mean over patch tokens (same width as CLS, no spatial layout);
`pg4` is a 4×4 spatial pooling of the 16×16 patch grid; `pg4pca768` is `pg4`
projected onto its own top-768 train PCs — **a width-matched control**, so a
gain over CLS cannot be explained by dimensionality.

Translation xyz R², clean, both strides:

| space | dim | s=1 f(o_t) | s=1 Δf | s=4 f(o_t) | s=4 Δf |
|---|---|---|---|---|---|
| CLS | 768 | 0.549 | 0.194 | 0.574 | 0.549 |
| patch-mean | 768 | 0.558 | 0.288 | 0.583 | 0.626 |
| patch 4×4 → top-768 PCs | 768 | 0.656 | **0.508** | 0.676 | 0.780 |
| patch 4×4 | 12288 | 0.704 | **0.612** | 0.725 | **0.839** |

* **The CLS token was hiding most of the signal in Δf.** Stride-1 translation
  goes 0.194 → 0.612, a **3.2×** gain. At *matched width* it is still 0.508, so
  this is about **keeping spatial layout**, not about having more dimensions.
  The decomposition is clean: changing the pooling (CLS → patch-mean) buys
  0.194 → 0.288; keeping *where* things moved buys 0.288 → 0.508.
* **Differencing is only outclassed by the single frame at stride 1.** In patch
  space Δf trails f(o_t) at stride 1 (0.612 vs 0.704) but **overtakes it** at
  stride 4 (0.839 vs 0.725). The original takeaway "Δf is worse than one frame"
  was a stride-1 CLS statement and does not generalise.
* **What does generalise is the state decomposition** (all-6 R², `pair` =
  both frames concatenated, `pair_shuf` = f(o_t) with a random other frame):

| stride | space | single | pair_shuf | pair | capacity gain | transition | single/pair |
|---|---|---|---|---|---|---|---|
| 1 | CLS | 0.413 | 0.418 | 0.458 | +0.005 | +0.041 | 90% |
| 1 | patch 4×4 | 0.508 | 0.506 | 0.555 | −0.003 | +0.050 | 92% |
| 1 | patch→768 | 0.479 | 0.481 | 0.556 | +0.002 | +0.076 | 86% |
| 4 | CLS | 0.436 | 0.442 | 0.580 | +0.006 | +0.138 | 75% |
| 4 | patch 4×4 | 0.524 | 0.521 | 0.677 | −0.003 | +0.156 | 77% |
| 4 | patch→768 | 0.497 | 0.500 | 0.671 | +0.002 | +0.171 | 74% |

`capacity gain` = pair_shuf − single (what the extra width alone buys: nothing —
≤ 0.016 on the reported groups, ≤ 0.022 on any of the 90 per-axis rows, always
far below the transition term). `transition` = pair − pair_shuf (the real
two-frame information). Better features raise *both* the single-frame and the
two-frame score — they do **not** change the 74–92% split. That ratio is the
robust finding, and it is what takeaway 1 now says.

Per-axis the split varies a lot and sensibly: at stride 4, **dz is the most
genuinely two-frame** dimension (62–75% state, transition +0.23…+0.29), while at
stride 1 **drz is ~100% state** (transition ≈ 0.00 — one-step wrist rotation is
essentially invisible in these features, and the probe is reading pose instead).
Quoting the 74–92% band as a per-axis claim would be wrong; across all 90 rows
it spans 56–102%.

**Why is rotation the weakest-decoded axis?** (Exp 4b, forward map.) Everything
above is a *backward* readout (probe action from Δf); the complementary question
is how much feature displacement each unit of action *produces*. Fit the forward
map action → Δf(pg4) on standardised actions and read the per-axis coefficient
norm — the "visual footprint" of one std of that action:

| axis | dx | dy | dz | drx | dry | drz | trans mean | rot mean | trans/rot |
|---|---|---|---|---|---|---|---|---|---|
| s=1 | 0.68 | 0.76 | 1.00 | 0.23 | 0.22 | 0.22 | 0.81 | 0.22 | **3.66×** |
| s=4 | 2.73 | 3.07 | 3.93 | 0.91 | 0.89 | 0.86 | 3.25 | 0.89 | **3.66×** |

A std of rotation moves the patch features **3.66× less** than a std of
translation — and the ratio is *identical* at both strides, so this is not a
stride/aliasing artifact. Rotation is weakly *decoded* because it is weakly
*written* into the pixels-to-features signal in the first place; the probe has
less to read, not (only) a worse readout. (Verified: `results/exp4b_footprint.csv`.)

## Exp 3 — a *real* pretrained LAM, identical protocol

UniVLA stage-1 (`qwbu/univla-latent-action-model`), a VQ-VAE over DINOv2 patch
tokens: 4 code positions × a 16-entry codebook, 512-d.

| stride | latent | trans xyz R² | rot R² | all-6 R² | gripper bal.acc |
|---|---|---|---|---|---|
| 1 | z_q (quantized) | 0.476 | 0.094 | 0.285 | 0.612 |
| 1 | z (pre-VQ) | **0.695** | 0.280 | 0.487 | 0.804 |
| 4 | z_q (quantized) | 0.619 | 0.135 | 0.377 | 0.645 |
| 4 | z (pre-VQ) | **0.804** | 0.332 | 0.568 | 0.811 |

All 16/16 codebook entries are used on LIBERO (no codebook collapse).

Two things follow, and both matter for how we frame this work:

* **Exp 1 is *not* the LAM ceiling the brief assumed.** A real LAM reads DINOv2
  *patch tokens*, not one global CLS vector, so it sees far more than the CLS
  difference does: 0.695 vs 0.194 at stride 1. Exp 1 bounds CLS-difference
  decoding, not LAMs in general. **Exp 4 confirms this and shrinks the gap**:
  against a patch-token difference the LAM's stride-1 lead is 0.695 vs 0.612
  (12288-d) or 0.508 at the LAM's own order of width — not 0.695 vs 0.194. Most
  of the apparent LAM advantage was the CLS bottleneck, not the LAM's training.
* **RETRACTED: VQ does not destroy the action — it removes redundant state.**
  The raw drop is real (z R² 0.695 → 0.476 xyz, 0.280 → 0.094 rot at stride 1 —
  a 31% / **66%** cut), but it is almost all *state the decoder already has from
  o_t*, not transition. The incremental probe (`exp3_incremental.py`) settles it:
  give the probe the single-frame patch feature f(o_t) (xyz R² 0.70) and measure
  what z adds **on top**. z_q keeps **89–96% of emb's genuine increment** in
  every dim and stride — the true test of a latent action:

  | | raw emb→z_q loss | **increment** emb→z_q loss (retained) |
  |---|---|---|
  | s1 xyz | 31% | 11% (**89%**) |
  | s1 rot | **66%** | 11% (**89%**) |
  | s4 xyz | 23% | 4% (**96%**) |
  | s4 rot | 59% | 7% (**93%**) |

  The alarming **66% rotation "loss" is 11%** on the increment — the rest was
  single-frame pose (drz is ~100% state at stride 1). So VQ is an *efficient*
  action code: it sheds the redundant state and keeps almost all of the (small,
  +0.03–0.11 R²) genuine transition. The "VQ costs 32%/66% of the action"
  framing is withdrawn; it was measuring free state removal.
* **And the drop that remains is not a linear-probe artifact.** One-hot encoding
  the 4 code indices (64-d, codebook-geometry-free) decodes action to the *same*
  R² as the 512-d z_q to 3 decimals (xyz 0.476/0.619) — the probe reads
  everything the codes carry. (The small residual increment loss above is the
  only genuine quantization cost, and it is small.)

### Exp 3 state decomposition — how much of the LAM latent is single-frame state?

Takeaway 1's shuffle control, applied to the LAM's *own* latent: encode
(o_t, o_{t+s}) vs (o_t, o_other), probe z → the true action at t. A high shuffled
score means z is re-encoding o_t's state — capacity the decoder does not need,
since it already sees o_t. Three shuffle tiers of rising OOD-ness guard against
"the encoder just broke on a weird input"; the trend is consistent and the
floor stays well above zero, so the effect is real state-encoding, not an
artifact. Ratio = R²(shuffled)/R²(pair) on all-6, stride 1 / stride 4.

| latent | pair R² (cont6) | same-ep | diff-ep | random |
|---|---|---|---|---|
| emb (pre-VQ) | 0.487 / 0.568 | 0.60 / 0.53 | 0.39 / 0.37 | 0.40 / 0.37 |
| z_q (post-VQ) | 0.285 / 0.377 | 0.26 / 0.20 | 0.13 / 0.10 | 0.13 / 0.11 |

* **The continuous latent wastes a lot of capacity on state.** Even with a
  *fully random* second frame, emb still recovers **37–40%** of the pair's action
  R²; with a same-scene frame, **53–60%**. Roughly half of what the continuous
  LAM latent decodes is single-frame state the decoder already has — a large,
  reclaimable budget, and the clearest evidence for a residual/passive-predictor
  design that carries only the transition increment.
* **The VQ bottleneck already squeezes most of it out.** z_q keeps only
  **10–26%** of its (smaller) score under shuffle. A tight discrete code cannot
  afford to store state, so quantization does some of that reclaiming for free —
  which is *why* z_q's absolute action score is lower (0.285 vs 0.487): it drops
  state and some genuine transition together. The residual budget is largest for
  high-capacity continuous latents, smaller against an already-tight code.

## Exp 2 — distractor contamination (probe trained & evaluated within condition)

Δf only; the frozen conditions are controls that hold the occluder fixed within
an episode, so they add no temporal variance (verified: border_static ring
pixel drift = 0.000 across t, robot dynamics bit-identical to clean).

| stride | condition | trans xyz R² | rot R² | all-6 R² | gripper bal.acc |
|---|---|---|---|---|---|
| 1 | clean | 0.194 | 0.069 | 0.132 | 0.646 |
| 1 | patch (frozen) | 0.149 | 0.051 | 0.100 | 0.622 |
| 1 | patch (moving) | 0.088 | 0.029 | 0.058 | 0.595 |
| 1 | border (frozen) | 0.049 | 0.016 | 0.032 | 0.553 |
| 1 | border (playing) | 0.018 | 0.004 | 0.011 | 0.531 |
| 4 | clean | 0.549 | 0.236 | 0.393 | 0.802 |
| 4 | patch (frozen) | 0.451 | 0.183 | 0.317 | 0.762 |
| 4 | patch (moving) | 0.321 | 0.124 | 0.222 | 0.713 |
| 4 | border (frozen) | 0.187 | 0.061 | 0.124 | 0.657 |
| 4 | border (playing) | 0.107 | 0.037 | 0.072 | 0.607 |

**One inversion worth a slide.** Under `border_static` — a frozen, per-episode
random border — Δf **beats** the single frame (all-6 R² 0.124 vs 0.016 at stride
4), and single-frame *rotation* R² goes **negative** (−0.053). The frozen border
is an episode fingerprint: the single-frame probe memorises it in training and
fails on held-out episodes, while differencing cancels it exactly. So frame
differencing is **robust to static nuisance and fragile to moving nuisance**.
(It is also a live demonstration of why the split must be by episode — a
frame-level split would have hidden this failure entirely.)

## Exp 2B — does a top-k (PCA) bottleneck keep the action? (theory test)

The theory predicts a variance-greedy latent bottleneck keeps the largest-
variance directions of Δf and discards the action. Test it with the right null:
compress Δf to k dims **three ways** — its top-k PCs, a **random** k-dim
subspace (5 seeds), and its **bottom**-k PCs — then refit the probe inside each
and read val xyz R². Clean condition; `exp2b_null.py` → `results/exp2b_null.csv`.

| space | s | k | top-k R² (% full) | random-k R² | bottom-k R² | top/rand |
|---|---|---|---|---|---|---|
| CLS 768 | 1 | 32 | 0.037 (19%) | 0.024 | 0.003 | 1.5× |
| CLS 768 | 4 | 128 | 0.358 (65%) | 0.281 | 0.018 | 1.3× |
| patch-mean 768 | 4 | 32 | 0.290 (46%) | 0.156 | 0.005 | 1.9× |
| patch 4×4 12288 | 1 | 32 | 0.165 (27%) | 0.050 | 0.015 | 3.3× |
| patch 4×4 12288 | 4 | 8 | 0.301 (36%) | 0.077 | 0.032 | 3.9× |
| patch 4×4 12288 | 4 | 128 | 0.632 (75%) | 0.438 | 0.073 | 1.4× |

**`top-k > random-k > bottom-k` in all 42 rows — every space, both strides, no
exceptions.** At k ≤ 128 the top principal directions carry **1.3–3.9×** more
decodable action than a random subspace of the same size; the **lowest**-variance
directions carry almost none (bottom-128 R² ≈ 0.01–0.07 against a full 0.19–0.84).
So a variance-greedy top-k bottleneck is the *best* linear bottleneck of its
width for action — not a lossy one. The theory's variance≠action mechanism does
not hold at the decoding level, in **any** feature space here.

### The metric that first said the opposite — and why it was wrong

The original CLS run reported the reverse: subspace energy ||P_k A||²/6 sat at
**0.03–0.24× chance** (k/D), read as "action concentrated in the low-variance
tail, below random." It was even "verified two ways" — the raw- and standardized-
feature action bases agreed (cos ≈ 1 in CLS) and the "% of full" refit was
reproduced. **Both checks were insufficient in the same way: neither compared to
a random-k null.** The energy metric's `w_raw = w_std/x_sd` step amplifies low-
variance coordinates, tilting the measured action *direction* into the tail — so
it answers "does the original weight vector lie in the top-k PCs" (fragile), not
"how much action survives a top-k bottleneck" (the refit, robust). In the
12288-d grid the two bases even diverge (cos ≈ 0.46–0.60; std energy ≈ 1–4.5×
chance vs raw ≈ 0.02–0.2×), which should have been the tell. The null control
settles it: the tail is empty, the head is rich. This is the cleanest example in
the whole study of a conclusion that a single extra control overturned.

### Footnote: the "action % vs variance %" framing is a curve shape, not evidence

An earlier write-up compared *% of full action* to *% of variance* at each k and
called it "lags" (CLS) or "leads" (patch grid, stride 4). That contrast is real
but says nothing about the tail: **any** subspace recovers more action as k grows
(random-128 already recovers 25–56% of full). Whether action% runs above or below
variance% only reflects how fast variance concentrates in the first few PCs
versus how fast action does — the top-vs-random null above is the load-bearing
comparison, and it points the same way everywhere.

*(Distractors, CLS: still make the bottleneck worse — at stride 4 `border`'s
top-8 PCs are 43% of Δf variance, the most of any condition, yet refit to xyz
R² ≈ 0.001, because the distractor is the dominant variance.)*

## Exp 2A on patch tokens — the last CLS-only headline, and the theory test *with* exogenous variance

Two purposes, one job (`exp2a_patch.py`, g4 space, all five conditions).

**Purpose A — the distractor probe table, finally in patch space.** Within each
condition, single / Δf / pair / pair_shuf; xyz R², stride 4:

| condition | single | Δf | pair | pair_shuf | transition |
|---|---|---|---|---|---|
| clean | 0.725 | 0.839 | 0.872 | 0.719 | +0.152 |
| patch (moving) | 0.632 | 0.617 | 0.763 | 0.622 | +0.141 |
| patch (frozen) | 0.575 | 0.746 | 0.751 | 0.566 | +0.185 |
| border (playing) | 0.511 | 0.385 | 0.633 | 0.502 | +0.131 |
| border (frozen) | 0.455 | 0.447 | 0.573 | 0.444 | +0.128 |

* **Takeaway 1 holds in patch space under every distractor**: pair_shuf ≈ single
  (gap ≤ 0.012 at every condition and stride), and the real two-frame `transition`
  = pair − pair_shuf survives (+0.13…+0.19 at stride 4).
* **Δf is more distractor-fragile than the concatenated pair.** Under a playing
  border, Δf falls *below* the single frame (0.385 vs 0.511) — differencing
  amplifies the moving distractor — yet the pair still beats it (0.633), because
  the probe can learn to ignore the distractor's dimensions rather than being
  forced to difference them. A representation that *keeps both frames* is more
  robust to task-irrelevant motion than one that *differences* them.

**Purpose B — the null control per condition** (top-k vs random-k vs bottom-k),
the variance≠action test where an *exogenous* distractor, not just the robot,
owns some variance. Raw xyz R², k=8, stride 4:

| condition | top-8 | random-8 | verdict |
|---|---|---|---|
| clean | 0.301 | 0.077 | top ≫ rand (3.9×) |
| patch (moving, 8% px) | 0.044 | 0.021 | top > rand (2.1×) |
| patch (frozen) | 0.258 | 0.057 | top ≫ rand (4.5×) |
| **border (playing, 64% px)** | **0.001** | **0.003** | **top < rand** |
| border (frozen, 64% px) | 0.091 | 0.018 | top ≫ rand (5.1×) |

* **This is takeaway 2's conditional flip, isolated.** Only the *playing* border
  reverses the order (top-k below random-k, and ≈ 0 in absolute terms — the top
  PCs are pure distractor). The frozen border, with the *same* 64% occlusion but
  no motion, keeps top-k winning 5×. So it is exogenous **motion owning the
  variance**, not the deleted pixels, that makes a variance-greedy bottleneck
  discard the action.
* **The moving patch (8% of pixels) is too weak to flip it** — it still gives
  top > rand (2.1×). The effect is threshold-like: the distractor has to actually
  dominate the variance. Clean manipulation data and small distractors don't;
  a screen-filling video does.

## Exp 5 — a *reconstruction*-target space: the Wan2.2 video VAE

DINOv2 is trained by self-distillation. A generative video model's VAE is the
opposite end — trained to reconstruct pixels, so it keeps everything needed to
reproduce appearance, action-relevant or not. We encode every LIBERO frame with
the 48-channel Wan2.2 VAE encoder (`WanVideoVAE38`, VAE latent only, no diffusion
features; each frame as a 1-frame clip → 48×14×14 = 9408-d) and run the identical
protocol. Three predictions, xyz R², vs the DINOv2 patch grid (pg4, same script):

| | Wan single | Wan Δf | Wan pair | Wan pair_shuf | DINO single | DINO Δf |
|---|---|---|---|---|---|---|
| s=1 | 0.572 | 0.556 | 0.641 | 0.559 | 0.704 | 0.612 |
| s=4 | 0.584 | 0.731 | 0.786 | 0.572 | 0.725 | 0.839 |

* **Pred 1 — state confusion holds, and it is encoder-agnostic. ✅** pair_shuf ≈
  single in Wan too (gap −0.012/−0.013), the genuine transition surviving at
  +0.08 (s1) / +0.21 (s4). Takeaway 1 now holds identically across a
  self-distillation encoder *and* a pixel-reconstruction VAE — strong evidence it
  is a property of scripted-demo data, not of any one feature space.
* **Pred 2 — the reconstruction latent is a worse action code per dim. ✅** Wan
  is below DINOv2 on every single-frame readout: xyz 0.572 vs 0.704, rotation
  0.202 vs 0.313 (stride 1). A pixel objective spends capacity on appearance the
  action does not need. (Wan's two-frame Δf/pair stay competitive — 0.731/0.786
  at s4 — so the deficit is concentrated in the state readout and rotation.)
* **Pred 3 — the reconstruction VAE keeps distractor variance, and is more
  susceptible than DINOv2. ✅** On clean Δf the static background cancels in the
  difference, so both encoders keep top-k > random-k (nothing task-irrelevant is
  left to keep). The test needs an *exogenous* distractor, so Exp 2A was re-run
  in Wan space. The controlled comparison — the **8% moving patch, same pixels,
  two encoders, stride 1** — is decisive:

  | encoder | full-Δf R² | top-128 R² (% of full) | top/rand |
  |---|---|---|---|
  | DINOv2 pg4 | 0.343 | 0.123 (**36%**) | 2.5× |
  | Wan VAE | 0.296 | 0.013 (**4%**) | 1.0× |

  The action exists in both (full-Δf ≈ 0.3), but DINOv2's *top variance
  directions* still hold 36% of it while Wan's hold **4%** — Wan's leading PCs
  *are* the moving patch. A pixel-reconstruction objective faithfully encodes
  task-irrelevant motion into its top variance; a semantic (self-distillation)
  objective downweights it. The mechanism is the same as takeaway 2 (playing
  distractors flip top-k below random for both encoders; frozen controls do not),
  but **Wan flips at a lower distractor threshold** — the 8%-patch that DINOv2
  mostly resists already captures Wan's top directions. Under the screen-filling
  playing border both collapse to ≈ 0 (too severe to discriminate).

So across a self-distillation encoder and a reconstruction VAE: state confusion
is **identical** (pred 1), the reconstruction latent is the **weaker** action
code (pred 2), and it is the **more distractor-susceptible** one (pred 3) — the
theory's variance-greedy failure mode bites it sooner. Caveat: Wan is 9408-d vs
DINOv2's 12288-d, a modest width handicap, though the gaps exceed what width
alone explains.

### Exp 5b — the video VAE used *as a video model*: temporal fusion beats differencing

Every number above (and every DINOv2/V-JEPA2 readout) differences *independent*
single-frame latents — the encoder never sees two timesteps at once. That is the
wrong way to use a video VAE. Wan's encoder has a 4× **causal temporal** path:
`iter_ = 1 + (T−1)//4`, so a 5-frame window returns two latents — `z0`, a
fresh-cache encode of `o_t` alone, and `z1`, which fuses `{o_{t+1..t+4}}` through
the temporal convs with causal context from `o_t`. (A 2-frame clip returns
`iter_=1` and silently drops the second frame — a trap; the 4-frame grid is the
real temporal unit.) We encode a `(4m+1)`-frame window (m latents of motion),
target the net action over its `4m` steps, and compare the VAE's **fused** latent
against the naive alternatives at matched width. Contract check passed: `z0`
equals the standalone single-frame latent (smoke rel 0.043; in the table `z0` =
`single` to 3 decimals), so the comparison is honest and `z1`'s content is purely
the temporal integration.

xyz R² (net translation over the window):

| m | span | single | diff (9408) | pair (indep, wide) | **video (fused, wide)** | **zlast (fused, 9408)** | pair_shuf |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.584 | 0.731 | 0.786 (18816) | **0.865** (18816) | **0.858** | 0.573 |
| 2 | 8 | 0.614 | 0.818 | 0.852 (28224) | **0.903** (28224) | **0.869** | 0.596 |

* **The learned temporal fusion strictly dominates — and it is not width.** The
  single fused motion latent `zlast` (9408-d, *same width as `diff` and `single`*)
  beats independent-latent concatenation `pair`, which is 2–3× wider: 0.858 vs
  0.786 (m=1), 0.869 vs 0.852 (m=2). It beats equal-width differencing by +0.127
  / +0.051. `pair_shuf ≈ single ≈ z0` (0.573/0.596 vs 0.584/0.614) confirms extra
  frames add nothing *unless the encoder fuses them* — width itself is inert,
  exactly as in the DINOv2 decomposition. The gain is the causal temporal convs
  integrating the 4-frame motion into a code no linear combination of endpoint
  latents reproduces.
* **— and it is not frame count either (Exp A, the frame-count null).** `zlast`
  saw *five* frames while `diff` and `pair` saw two, so the +0.127 could have been
  "more frames," not fusion. The null concatenates `k` **independent** single-frame
  latents evenly spanning the window (`multi_k`, encoded alone — no fusion) and its
  width-matched top-9408 PCA. Handing the probe exactly the five frames `zlast` saw
  (`multi_5` / `multi_5_pca9408`) reaches only **0.754 / 0.751** — closing just
  **18%** of the `diff`→`zlast` gap; `zlast` still leads by **+0.104** at matched
  frames *and* matched width. At m=2, all nine frames (`multi_9_pca9408`) reach
  0.826, closing **15%**; `zlast` leads by +0.044. Tellingly, extra independent
  frames *hurt* the concat probe (`multi_5` 0.754 < `multi_3` 0.771 at m=1): they
  add redundant **state**, not motion, whereas fusion helps monotonically and helps
  rotation most (`zlast` rot 0.394 vs `multi_5` 0.166, *below* even `diff`'s 0.301).
  So the Wan advantage is genuine learned temporal comparison — **Pillar 1 survives
  its hardest null.** (Verified: `results/exp5c_framecount_null.csv`.)
* **Rotation, the weakest axis everywhere, benefits most in relative terms.**
  `zlast` rot 0.394 vs `diff` 0.301 (m=1) and 0.416 vs 0.375 (m=2); cont6 0.626 vs
  0.516 and 0.643 vs 0.597. Fusion partly recovers the rotation that differencing
  discards.
* **This reframes pred 2.** Single-frame Wan looked like a *weaker* action code
  than DINOv2 (0.572 vs 0.704) — but that measured frame-*independent* use. Used
  as a video model, Wan's fused window latent (xyz 0.858–0.865 at span 4)
  **matches/edges DINOv2's differenced patch grid** (pg4, 12288-d, 0.839) on
  translation, while DINOv2 still leads on rotation (0.438 vs 0.39). The pixel
  objective was never the liability; differencing independent latents was. This is
  the one place in the whole study where a genuine two-timestep *representation*
  (not a hand-differenced one) clears every naive baseline by a real margin.

(Verified: `results/exp_wan_temporal.csv`; DINOv2 from `results/exp4_patch_matrix.csv`.)

## Exp 6 — a *prediction*-target space: V-JEPA2 (the third encoder objective)

DINOv2 is self-distillation, Wan a reconstruction VAE; V-JEPA2 is the third
objective — masked **prediction** in latent space, trained on video. The
hypothesis (item-3 pred 4) was that a prediction encoder carries motion/location
*redundantly* across tokens, so spatial pooling should be **less** harmful than
for DINOv2. We feed each frame as a 2-frame tubelet `[o_t,o_t]` → 256 tokens ×
1024-d, and pool two ways: `g4` (4×4 grid, 16384-d, keeps layout) and `pmean`
(global mean, 1024-d), the exact analog of DINOv2's `pg4`/`patch-mean`. Clean,
xyz R²:

| feature | VJ g4 | VJ pmean | DINO pg4 | DINO pmean |
|---|---|---|---|---|
| s1 single | 0.645 | 0.526 | 0.704 | 0.558 |
| s1 Δf | 0.362 | 0.176 | 0.612 | 0.288 |
| s1 pair | 0.689 | 0.579 | 0.769 | 0.627 |
| s1 pair_shuf | 0.633 | 0.522 | 0.703 | 0.571 |
| s4 Δf | 0.722 | 0.525 | 0.839 | 0.626 |

* **Pred 4 is refuted — pooling is *equally* harmful, not less.** V-JEPA2's Δf
  loses **51%** of its xyz decodability to mean-pooling at s1 (0.362 → 0.176) and
  **27%** at s4 (0.722 → 0.525); DINOv2 loses **53%** / **25%** (0.612 → 0.288,
  0.839 → 0.626). Nearly identical. This is the *stronger* result: Exp 4's central
  lesson — *keep the spatial layout, most of Δf's action signal lives in where
  things moved* — is **encoder-agnostic**, holding to within 2 points across a
  self-distillation and a prediction objective. The prediction encoder is not
  specially pooling-robust.
* **State confusion holds — now across all three objectives.** `pair_shuf ≈
  single` in V-JEPA2 too (g4 gap −0.012 s1, −0.013 s4; pmean −0.004), the genuine
  transition surviving at +0.056 (s1) / +0.155 (s4). Takeaway 1 is now confirmed
  identically for self-distillation, reconstruction, **and** prediction — strong
  evidence it is a property of scripted-demo data, not of any feature space.
* **Absolute level: V-JEPA2's single-frame Δf is the *weakest* of the three at
  stride 1** (g4 0.362 vs DINOv2 0.612 vs Wan 0.556), catching up by stride 4
  (0.722 ≈ Wan 0.731, still below DINOv2 0.839). A frozen frame-pair encoding of a
  video-prediction model is a mediocre one-step action code; its strength is in
  genuine temporal input — **confirmed in Exp 6b below**, where real clips fed to
  V-JEPA2's temporal path fuse far better than any differencing of frozen frames.
* **Distractor null: intermediate susceptibility, completing the ordering.** Under
  the 8% moving patch, V-JEPA2's top-128 Δf directions retain **16%** of the full
  action at s1 (top/rand 1.8×) and 42% at s4. That places prediction between the
  two extremes: **DINOv2 36% > V-JEPA2 16% > Wan 4%** (s1) — semantic downweights
  distractor variance most, reconstruction encodes it into top PCs most,
  prediction sits in between, exactly as the training objectives predict.

(Verified: `results/exp_vjepa_probes.csv`, `results/exp_vjepa_null.csv`, harvested
from the job log — the run hit its 90-min wall during the `border` null control,
which collapses to ≈ 0 for every encoder and adds nothing; all informative rows
completed. DINOv2 from `results/exp4_patch_matrix.csv`.)

### Exp 6b — V-JEPA2 used *as a video model*: fusion is general, not Wan-specific

Exp 6 fed V-JEPA2 a duplicated-frame tubelet `[o_t,o_t]` — a video model used as an
image model, forfeiting the one thing it is for. Exp 6b feeds **genuine clips**: the
`(4m+1)`-frame window, padded to `4m+2` by duplicating the last frame (tubelet size
2, so all `4m+1` real frames are seen), read out as g4-pooled tokens (4×4 layout
kept). Two temporal readouts, both **16384-d — the same width as `vj_single` and
`vj_diff`**: `vj_fused_last` (last temporal token, the mirror of Wan's `zlast`) and
`vj_fused_tmean` (mean over temporal tokens). Contract check passed exactly: fused
`[o_t,o_t]` reproduces the cached `vj_single` to rel **0.000**.

xyz R² (net translation over the window; g4, 16384-d unless noted):

| m | span | vj_single | vj_diff | vj_pair (32768) | multi_k·pca (matched) | **vj_fused_last** | **vj_fused_tmean** |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 0.664 | 0.722 | 0.806 | 0.743 | **0.896** | **0.906** |
| 2 | 8 | 0.704 | 0.842 | 0.873 | 0.824 | **0.940** | **0.942** |

* **Fusion is real and general to video encoders.** At *matched width* the fused
  latent crushes differencing: 0.896 vs `vj_diff` 0.722 (m=1, **+0.175**) and 0.940
  vs 0.842 (m=2, **+0.099**); it also clears the 2× wider `vj_pair` (0.806 / 0.873).
  This is the Exp-5b pattern reproduced in a second, independently-trained video
  encoder — temporal fusion is a property of **video encoders, not of Wan's causal
  convs.**
* **And it is not frame count (Exp A's null, in V-JEPA2 space).** Concatenating the
  `k` *independent* per-frame latents the clip saw, width-matched to 16384, reaches
  only `multi_5_pca` **0.743** (m=1) / `multi_9_pca` **0.824** (m=2) — the fused
  latent leads by **+0.153 / +0.116** at matched frames *and* width. As in Wan,
  extra independent frames *hurt* the concat probe (`multi_5_pca` 0.743 <
  `multi_3_pca` 0.748; `multi_9` 0.824 < `multi_3` 0.832): they add redundant state,
  and only the encoder's own temporal path turns the window into motion.
* **Rotation gains most, again.** `vj_fused_last` rot 0.494 vs `vj_diff` 0.320 (m=1)
  and 0.634 vs 0.468 (m=2) — the largest relative fusion gain of any axis, echoing
  Wan.
* **Side-by-side with Wan (same spans, matched fused width 16384/9408-d):**

  | | Wan `zlast` xyz | V-JEPA2 `fused` xyz | Wan `zlast` rot | V-JEPA2 `fused` rot |
  |---|---|---|---|---|
  | m=1 (span 4) | 0.858 | **0.896** | 0.394 | **0.494** |
  | m=2 (span 8) | 0.869 | **0.940** | 0.416 | **0.634** |

  V-JEPA2's fusion is **stronger** than Wan's on both translation and rotation. With
  its better distractor robustness (top-128 retains **16%** of the action vs Wan's
  **4%** under the 8% moving patch), V-JEPA2 is the better home for a learned latent
  action.
* **Causal vs bidirectional — the caveat, and why it is benign here.** Wan's temporal
  path is causal (`z0` = the standalone single-frame latent); V-JEPA2's attention is
  bidirectional, so within the clip `o_t`'s tokens have already attended to
  `o_{t+4m}`. But our window **ends at the target endpoint** `o_{t+4m}` — it is
  tail-aligned, so nothing *past* the decoded interval leaks. And a latent *action*
  model is by construction handed both endpoints of the transition it must encode, so
  bidirectional fusion over the observed `(o_t, o_{t+k})` window is exactly its
  information access, not leakage. (Where bidirectionality *would* bite is Pillar 2's
  predictive residual — a bidirectional passive predictor can explain away the motion
  it was meant to leave as residual; a method-design constraint, flagged for the
  reviewer, not an Exp-6b artifact.)

(Verified: `results/exp6b_vjepa_temporal.csv`; Wan from `results/exp_wan_temporal.csv`.)

---

## Caveats, and one bug worth knowing about

* **A bug in Exp 3 initially reversed its conclusion.** UniVLA's
  `VectorQuantizer.forward` returns `(z_q, z, x, indices)` where *both* `z_q` and
  `z` are quantized (`z_q = x + (z − x).detach()` is the straight-through
  estimator, numerically equal to `z`); the continuous latent is `x`, exposed as
  `emb`. Reading `z` made the "pre-VQ" row identical to `z_q` — i.e. *"quantization
  is free"*, the opposite of the truth. Caught because the two rows matched to
  three decimals. Now asserted against, with the quantization error printed.
* **Exp 1 is not a LAM upper bound** — see Exp 3 and Exp 4. The brief's framing
  needs correcting, and takeaway 1 has already been rewritten because of it.
* **Exp 2B conclusion changed twice under new controls, and the caveats trail
  the headline on purpose.** First run (CLS, subspace energy): "action in the
  low-variance tail." Patch re-run (refit): "reverses on the grid." Null control
  (top vs random vs bottom): "action is in the high-variance directions in *every*
  space — the tail claim was a metric artifact." The final version is the null
  control; the first is kept in the 2B section only as a documented cautionary
  tale. Takeaway 2's final reading is that the mechanism is **real but
  conditional**: on clean data the action owns the variance and a top-k
  bottleneck is optimal (top > random > bottom, 42/42 rows), and it flips only
  when an *exogenous* distractor owns the variance — the Exp 2A per-condition null
  isolates that flip to the playing border.
* **Null control used a full GPU SVD** for exact top-k *and* bottom-k ordering;
  the refit is the same sklearn probe as everywhere else. The full-D baselines
  (CLS 0.194/0.549, patch 0.612/0.839) match Exp 1 and Exp 4 to 3 decimals.
  Random-k is 5 seeds; the only row where top-k is not clearly above random by
  2σ is CLS s=1 k=8 (R² ≈ 0.009, all values tiny). pg4's earlier `exp2b_patch`
  run used randomized SVD for the variance figures; those are secondary now.
* **`border` conflates occlusion with motion.** It replaces a 45px ring = **64.2%
  of pixels**, so most of its damage is deleted workspace (frozen −75% vs playing
  −91%). **`patch` is the clean distractor test** (8% of pixels, motion-dominated).
* **alpha/C are selected on val and reported on val**, as the brief specified.
  Mildly optimistic. The selected alpha sat on the top of the specified grid in
  every Exp-1 row, so the grid was widened as a check (alpha to 1e6, C to 1e3,
  `results/exp1_decodability_extgrid.csv`):
  * **every Δf row is unchanged to 3 decimals** — alpha=100 was genuinely
    optimal, not grid-truncated. All headline claims rest on these rows.
  * only the **single-frame** rows move (alpha 100 → 1000), by ≤ 0.015 on a
    reported group (stride-4 rot R² 0.283 → 0.298; xyz drifts the other way by
    0.002) and ≤ 0.022 on any per-dim row (stride-4 drz 0.339 → 0.360). The
    single-vs-Δf gap therefore shifts by at most 0.015 in either direction,
    against a gap of 0.357 (stride 1, xyz) — far too little to affect takeaway 1.
  * Exp 3's pre-VQ rows sit at the *low* edge instead (alpha=0.01), but with
    n=49k ≫ d=512 that is essentially OLS and well-posed.
* Ridge's cholesky solver warns of ill-conditioning at the two smallest alphas;
  irrelevant here since alpha=100 won every Exp-1 row.
* **Exp 4 widened the ridge grid to 1e-4 … 1e5** because its spaces span 768 to
  24576 dims and the right shrinkage is not comparable across them. Four rows
  still selected the *bottom* edge (alpha=1e-4), all of them `pair` on the PCA
  spaces. With n=49k ≫ d=1536 that is essentially OLS and well-posed; and if the
  true optimum were lower still, `pair` would only rise, which *strengthens* the
  `transition` term rather than weakening it. No `diff` or `single` row is at an
  edge.
* **Exp 4's GPU probes were checked against the sklearn path** on the real Exp-1
  features before use: worst metric disagreement 5.5e-04, ridge weight rows
  cosine 1.000000, same alpha selected. Patch features also reproduce the cached
  CLS features to max rel err 7.0e-04.
* Encoder ran in fp16 autocast (LAM in bf16). Not re-checked in fp32.

## What I would do next

1. ~~Redo Exp 1 on patch-token differences~~ — **done, Exp 4.** It mattered:
   stride-1 Δf translation R² 0.194 → 0.612.
2. ~~Re-run Exp 2B on patch tokens~~ and ~~add the random-k / bottom-k null~~ —
   **both done.** The null control was decisive: the "action in the low-variance
   tail" mechanism is dead in every space (top-k > random-k > bottom-k, 42/42).
   ~~Still open: Exp 2A (distractors) on patch tokens~~ — **done** (`exp2a_patch.py`).
   No CLS-only headline remains. It also settled takeaway 2 into its conditional
   form: variance-greedy selection fails only when an exogenous distractor owns
   the variance (playing border), not under clean data or a weak distractor.
3. Probe **UniVLA stage-2** (language-conditioned, explicitly designed to suppress
   task-irrelevant motion) under the same distractors. The checkpoint is already
   downloaded; it needs a T5 embedding per sample, and LIBERO ships the
   instruction string in the hdf5 `problem_info` attr.
4. The bottleneck curve is the real headline — worth extending to a trained
   LAM's own latent width rather than PCA as a proxy.

---

## Reproducibility

Commit `0c340e0` · seed 0 · 80/20 **episode-level** split stratified within each
of the 10 tasks (400 train / 100 val of 500 episodes; 62 250 frames).
Env `/anvil/scratch/x-tding2/envs/latent_free` (venv over `envs/orena`; that env
is unmodified). Data `data/LIBERO/libero_spatial` (HF `yifengzhu-hf/LIBERO-datasets`).

```bash
sbatch --partition=ai --account=cis250950-ai --cpus-per-task=24 \
       --gpus-per-node=1 run_all.sh     # 5 conditions extracted in 1.53 min on an H100
sbatch run_exp3.sh                      # optional LAM probe
sbatch run_exp4.sh                      # patch tokens + state control (7.8 min)
sbatch run_exp4_pca.sh                  # width-matched PCA control (0.9 min)
sbatch run_exp2b_patch.sh               # Exp 2B on patch tokens (CPU work on ai; 17 min)
sbatch run_exp2b_null.sh                # 2B null: top-k vs random-k vs bottom-k (15 min)
sbatch run_statedecomp.sh               # LAM latent state decomposition (item 1; 35 min)
sbatch run_incremental.sh               # what z adds beyond f(o_t): VQ = state removal (item 1b)
python exp3_onehot.py                   # VQ cost is real, not a probe artifact
python exp2_verify.py                   # CLS Part-B verification (superseded by the null)
python make_tables.py                   # regenerates every table above from the CSVs
```

**Two verified data facts** (printed by `data.py` on every run):
`len(actions) == len(agentview_rgb)` for all 500 episodes — no off-by-one in this
release, so `(o_t, o_{t+s})` is explained by `actions[t:t+s]`; and frames are
stored bottom-up (`macros_image_convention='opengl'`) and **must be flipped** —
otherwise every feature is computed on an upside-down image.

Outputs: `results/exp1_decodability.csv`, `exp1_decodability_extgrid.csv`,
`exp2_distractor.csv`, `exp2_pca_action_energy.csv`, `exp2_verify_topk.csv`,
`exp2a_patch_probes.csv`, `exp2a_patch_null.csv`, `exp2b_patch_pca.csv`,
`exp2b_null.csv`, `exp3_lam.csv`, `exp3_statedecomp.csv`, `exp3_incremental.csv`, `exp3_onehot.csv`,
`exp3_onehot.csv`, `exp4_patch_matrix.csv`,
`exp4_state_decomposition.csv`;
`figs/exp1_r2_bars.png`, `exp2_r2_clean_vs_distracted.png`,
`exp2_pca_action_energy.png`, `exp4_patch_matrix.png`,
`example_*.png` (3 frames × 5 conditions × 2 episodes).
