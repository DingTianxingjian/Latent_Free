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

`rollout_compare` is the qualitative comparison: two tasks, UniVLA* (blue, the hue it has in every other figure)
above TERRA (violet), four frames each with their real step numbers, a red frame + cross + boxed-and-magnified
region on the failure step and the same in green on the completed step, a thin time axis per task, and the paper's
own nuisance intervention applied to BOTH rows so the arms see one perturbed scene.  The dark square in the
LIBERO-Object frames is that intervention -- the moving patch -- not a wrist camera; the header names it.

**It is a layout mock-up.**  The frames are demonstration observations from `data/libero_rlds`, not policy
rollouts, and the figure prints a footer saying so.  Both rows of a task come from one clip, so $t=0$ is
pixel-identical, but for the same reason the failure row does not yet *look* different from the success row: only
real rollouts can show that.  The ROI boxes are placed by eye and the labels under the red frames are the failure
modes the figure argues for, not observed ones.  For the real version: one episode and one seed for both rows,
ROIs where the failure actually is, observed failure labels, and `MOCK = False`.

`nuisance_examples` renders the two perturbations the project actually measures, on real LIBERO observations, with
the project's own code: the moving-patch distractor (`acdc_la/data/distractor.py`, a 64x64 natural-video crop
travelling the frame border, the intervention behind every "drop" number) and one fixed draw of the invariance
term's brightness/contrast family (`stage3_transport/train_v2.py::_jitter`, brightness 0.80 x contrast 1.20).
It is laid out like `rollout_compare` (left colour bar, rotated row name, block titles) and carries no chart: the
measured drops (endpoint-only 33.6 %, UniVLA 22.9 %, ours 17.9 %) belong in the caption, and
`representation_comparison`(c) already plots them.  Scene-level perturbations (a real light source moved in MuJoCo, an
extra object added to the BDDL scene, as in LIBERO-Plus) are NOT what this figure shows and would need a GPU node
plus scene edits.

`teaser` follows the visual language of Jenni et al. (ECCV 2020), SpeedNet (CVPR 2020) and MoCoGAN (CVPR 2018):
data first, representation second.

(a) is **exact by construction**.  One real LIBERO-Spatial demonstration ("pick up the black bowl from table
center", approach phase t = 0..61) resampled under two temporal warps, w(u) = 1-(1-u)^2 and w(u) = u^2, four frames
each, cropped to the arm and the bowl; the dotted timeline under each strip marks which source frames were taken.  First and last frames are the
same pixels, so an encoder that reads only (x_0, x_T) receives identical inputs and must return the same latent --
no measurement is needed for that claim.

(b) uses real scenes and is **schematic in its arrows**.  Donor: the table-centre demonstration; similar recipient:
another demonstration of the same task; shifted recipient: "bowl next to the cookie box", whose bowl sits where
the donor's table centre is empty.  Every cell is one real t = 0 frame; the grey dashed arrow is the donor's effect
at its absolute location and the coloured arrow is where each objective would send it (reconstruction: the same
absolute target, circled on empty table in the shifted scene; EAT: the recipient's bowl).  No "after" frames: the
figure makes no claim about rolled-out behaviour.  The source clip is encoded once into z, which is applied to both recipient rows; a legend under the grid names the two arrow kinds.  In (a), each strip's name sits horizontally above it, and the column to its right plots the two warps themselves as progress-vs-time curves (after the timelines of Pace Prediction and the speediness curves of SpeedNet): seen only at the endpoints (black dots) the two curves coincide, seen at every frame they separate, and the coloured dots on the lower plot are exactly the samples marked on the strips' timelines.  This replaced a latent-space cartoon, whose dots carried no meaning of their own; the curves are the actual w(u), so the panel is exact.  Panel titles start at the x of each panel's first frame.
Frames: `figures/teaser_eps/keep/*.npy`.


The LIBERO-Long perturbation is a **directional** illumination gradient (0.55x to 1.25x across the frame), not the
global brightness/contrast draw the invariance term trains on (`_jitter`, U(0.8,1.2) scalars).  Testing with the
augmentation family itself would put the perturbation inside our training distribution and outside the baseline's,
which is a fair objection; the gradient is spatial and no arm sees it during training.  The moving patch stays --
it is the intervention every probe number in the paper is measured under -- and the figure now names it with a
dashed box, since it otherwise reads as a wrist-camera inset.
