Figures for the paper.  Regenerate all three with

    ../envs/univla_vla/bin/python acdc_la/figures/make_paper_figures.py

| file | role | who appears |
|---|---|---|
| `representation_comparison` | comparison with other latent-action methods, and where the policy ends up | LAPA, UniVLA, TERRA (+ OpenVLA in the closed-loop panel only) |
| `representation_ablation` | our own tokenizer ablation: what the temporal factorisation buys | mean-pooled vs TERRA.  **No external method**; the earlier cell-wise tokenizer is cut from the paper (`SHOW_CELLWISE = True` restores it) |
| `eat_refinement` | objective ablation: the EAT terms, all from one shared checkpoint | recon-only -> no-invariance -> complete |

OpenVLA has no latent action code, so it cannot be probed and never appears in the probe panels; it enters only the
closed-loop success panel, where it is the no-latent reference.  UniVLA is a single entry at the matched budget
(`s2mix_1ep`); the 2-epoch run is not shown.

Every number lives in the DATA block at the top of the script, tagged `[real]` (with its LEDGER date) or
`[PLACEHOLDER]`.  Placeholder values are drawn like measured ones; set `SHOW_PROVENANCE = True` in the script to hatch every
`[PLACEHOLDER]` value and add a legend entry saying so, which is the safer state for a draft that circulates.
Currently placeholder: the LAPA probe row, and the closed-loop panel of the `with_policy` variant except
UniVLA's Spatial/Object/Goal.  Palette validated with the data-viz validator (light surface, categorical, CVD-safe); the
one slot below 3:1 contrast carries a direct value label on every bar as relief.  PDFs are vector, Type-42 fonts.

`libero_suites` is the compact 2x2 (suite names only, sized for a half-width slot); `libero_suites_strip` is the
full-width row of four, which carries each suite's **shortest** instruction verbatim -- short enough to stay legible
under a quarter-width panel.  All three are built by `acdc_la/figures/make_env_figure.py` from frames in
`figures/libero_frames/` (+ `frames.json` with each task's instruction).  Those frames are the **first
observation of a demonstration**, read straight from the RLDS shards in `data/libero_rlds/` on CPU -- the same
images the policy is trained and evaluated on.  `acdc_la/figures/grab_libero_frames.py` renders equivalent frames
from the simulator instead (`set_init_state` + the evaluator's 10 no-op settle), but MuJoCo needs EGL and this
cluster has no OSMesa, so it only runs on a GPU node via `acdc_la/scripts/run_libero_frames.sh`.

`teaser` is the page-1 figure: (a) two windows with the same endpoint and opposite first dynamics mode, (b) one
latent decoded into a near and a far recipient state.  The large marks are schematic; the two inset charts are
numbers.  The profile NMSE inset is measured (`1 - R^2(profile)`: endpoint-only .983 from `s1_mean_1ep`, ours .909
from `latp2_inv`).  **The Q1/Q4 effect-error inset (.80 -> 1.06 recon-only, .77 -> .86 EAT) is quoted from the
manuscript's Fig. 3(a) and has not been recomputed here** -- no such measurement exists in this repository.

`rollout_compare` is the qualitative comparison (two tasks, UniVLA* above TERRA, red frame on the failure step,
green frame + check on completion).  **It is a layout mock-up**: the frames are demonstration observations pulled
from `data/libero_rlds`, not policy rollouts, and the figure prints a footer saying so.  Both rows of a block use
the same episode, so the "failure" row is simply truncated -- the labels under the red frames are the failure modes
the figure is meant to show, not observed ones.  When real rollouts exist, put their frames in
`figures/rollout_frames/` with the same names, set `MOCK = False` in `acdc_la/figures/make_rollout_figure.py`, and
uncomment the figure block in `paper/experiments.tex`.

`nuisance_examples` renders the two perturbations the project actually measures, on real LIBERO observations, with
the project's own code: the moving-patch distractor (`acdc_la/data/distractor.py`, a 64x64 natural-video crop
travelling the frame border, the intervention behind every "drop" number) and one fixed draw of the invariance
term's brightness/contrast family (`stage3_transport/train_v2.py::_jitter`, brightness 0.80 x contrast 1.20).
It is laid out like `rollout_compare` (left colour bar, rotated row name, block titles) and carries no chart: the
measured drops (endpoint-only 33.6 %, UniVLA 22.9 %, ours 17.9 %) belong in the caption, and
`representation_comparison`(c) already plots them.  Scene-level perturbations (a real light source moved in MuJoCo, an
extra object added to the BDDL scene, as in LIBERO-Plus) are NOT what this figure shows and would need a GPU node
plus scene edits.
