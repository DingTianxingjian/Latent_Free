# Gating experiments before the method (PRLA) can be designed

## Why these two, and why now

The planned method has two pillars:

1. **learned temporal comparison** — let the encoder's own temporal path fuse a window,
   instead of hand-differencing independent per-frame latents;
2. **predictive residual** — define the latent action as what a passive predictor cannot
   explain, instead of an unconditional two-frame compression.

Pillar 1 currently rests entirely on **Exp 5b** (Wan fused `zlast` xyz R² 0.858 vs `diff`
0.731 at matched 9408-d width). That result has a hole and a scope limit, and both must be
closed before any method code is written:

* **Hole (Exp A below):** `z1` fuses `o_{t+1..t+4}` with causal context from `o_t` — it has
  *seen five frames*. `diff` and `pair` see only the two endpoints. So the +0.127 could be
  "learned fusion" or simply "more frames". Untested.
* **Scope limit (Exp B below):** Exp 6 fed V-JEPA2 a duplicated-frame tubelet `[o_t, o_t]`,
  i.e. used a video model as an image model. Whether the fusion advantage is a property of
  video encoders in general — or specific to Wan — is unknown, and it decides *which
  encoder the method is built on*.

Both are cheap. Neither requires training a model. Do not start method work until they land.

Repo conventions already established (keep all of them): LIBERO-spatial, 500 episodes /
62,250 frames, **episode-level 80/20 split**, seed 0, ridge probe with the existing alpha
grid, val-selected alpha, report xyz R² / rot R² / cont6 R² / gripper balanced accuracy.
Reuse `data.py`, `probes_torch.py`, cached features wherever possible.

---

## Exp A — frame-count null for the temporal-fusion result (highest priority, ~30–60 min)

**Question.** Is Wan's fused latent better than differencing because of *learned temporal
fusion*, or merely because it *saw more frames*?

**Design.** Same windows as Exp 5b: a `(4m+1)`-frame window, target = net action over its
`4m` steps, m ∈ {1, 2}. Encode each frame **independently** (fresh cache, single-frame
encode — the same path that produced `z0`), then build these baselines and compare against
the existing fused rows:

| row | construction | width | purpose |
|---|---|---|---|
| `single` | `f(o_t)` | 9408 | existing |
| `diff` | `f(o_{t+4m}) − f(o_t)` | 9408 | existing |
| `pair` | concat endpoints | 18816 | existing |
| **`multi_k`** | **concat of k independent single-frame latents, evenly spaced over the window, k ∈ {3, 5} for m=1; k ∈ {3, 5, 9} for m=2** | 9408·k | **new — the null** |
| **`multi_k_pca9408`** | **`multi_k` projected to its own top-9408 train PCs** | 9408 | **new — width-matched null** |
| `video` / `zlast` | fused | 18816 / 9408 | existing |
| `pair_shuf` | `f(o_t)` + random other frame | 18816 | existing |

For m=1 the 5-frame window means `multi_5` uses *exactly the frames the fused latent saw*.
That is the decisive row.

**Verdict rule (state it explicitly in the output):**
* If `multi_5` (and especially `multi_5_pca9408`) **reaches** `zlast` (within ~0.01 xyz R²)
  → the Exp 5b advantage is frame count, not fusion. Pillar 1 is **not supported**; say so
  plainly.
* If `zlast` still leads at matched frames **and** matched width → fusion is real. Report
  the gap separately for xyz / rot / cont6 (rotation is where fusion helped most: 0.394 vs
  0.301, worth confirming).
* Intermediate → report the fraction of the gap explained by frame count.

**Output.** `results/exp5c_framecount_null.csv`, plus a short verdict paragraph written into
SUMMARY under Exp 5b (amend, don't append — the existing 5b claim must be qualified in place).

---

## Exp B — V-JEPA2 with genuine temporal clips (~2–3 h incl. extraction)

**Question.** Is the fusion advantage general to video encoders, or Wan-specific? And does
V-JEPA2 — which our distractor null ranks as far more robust than Wan (top-128 retains 16%
vs 4% under the 8% moving patch) — also get it? This decides which space the method learns
`z` in.

**Design.** Replace the duplicated-frame tubelet of Exp 6 with **real clips**. Feed
V-JEPA2 a genuine `T`-frame window (match Exp 5b's spans: 5 frames for m=1, 9 for m=2;
if V-JEPA2's tubelet size forces a different T, pick the nearest valid T and **state the
actual span in the output** so the Wan comparison is honest about any mismatch).

Extract two readouts per window and pool spatially with `g4` (4×4 grid, keep layout — never
global-mean; T3 says pooling costs ~half):
* **`vj_fused`** — the encoder's tokens for the real clip, spatially pooled, temporally
  pooled or last-timestep (whichever the architecture makes natural — document the choice);
* **`vj_single`** — same encoder, duplicated-frame tubelet for `o_t` alone (the Exp 6 path,
  as the state baseline);
* **`vj_diff`** — difference of two `vj_single` latents at the window endpoints;
* **`vj_pair`**, **`vj_pair_shuf`** — as usual.

Also run **Exp A's frame-count null in this space** (`multi_k` from independent per-frame
`vj_single` latents), so the same verdict rule applies here.

**Important caveat to check and report — future leakage.** V-JEPA2's temporal attention is
likely **bidirectional**, so the "current" timestep has already seen the window's future.
Wan's temporal path is causal (verified in 5b: `z0` equals the standalone single-frame
latent). This is not a bug but it changes what the number means: a bidirectional encoder's
window latent may encode *anticipation*, not just executed motion. Report it as a caveat,
and if the architecture permits, add one control row using a **tail-aligned window**
(window ends at the target interval) so no post-target frames are visible.

**Verdict rule:**
* `vj_fused` ≫ `vj_diff` at matched width and frames → fusion is general; combined with
  V-JEPA2's better distractor robustness, **the method learns `z` in V-JEPA2 space** and
  Wan is reserved for the generation side only.
* `vj_fused` ≈ `vj_diff` → fusion is Wan-specific (causal convs), and the method faces a
  real trade-off between fusion quality and distractor robustness — flag it, do not paper
  over it.

**Output.** `results/exp6b_vjepa_temporal.csv` + a SUMMARY subsection under Exp 6, with a
side-by-side table against Wan's Exp 5b rows (note any span mismatch explicitly).

---

## Housekeeping to fold into the same commit

Three known inconsistencies in SUMMARY.md — fix while these run:

1. **Takeaway 3 still quotes CLS numbers** (0.194 → 0.088) although Exp 2A now has the patch
   version (0.725 / 0.617 / 0.385 table). Rewrite takeaway 3 from the patch numbers.
2. **A caveat still says "the distractor table (Exp 2A) is still CLS-only"** — stale, the
   patch run exists. Remove it.
3. **Takeaway 2 and its caveat disagree**: the takeaway says the mechanism is real but
   conditional ("who owns the variance"); the caveat says takeaway 2 is "a refutation of the
   mechanism". Unify on the takeaway's wording — the caveat predates the Exp 2A per-condition
   null.

---

## Reporting

One consolidated report at the end with: the two verdicts stated in one sentence each, the
tables, what each verdict implies for the method (which encoder, whether pillar 1 survives),
and any caveat that a reviewer would raise. Do not soften a negative verdict — a null result
on Exp A is more valuable now than after two months of method work.