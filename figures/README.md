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
`[PLACEHOLDER]`.  Placeholder values are drawn **hatched**, with a legend entry saying so, so a reader (or an advisor
seeing the mock-up) cannot mistake an expected shape for a measurement.  All of panel (d) of the comparison figure is
currently expected values.  Palette validated with the data-viz validator (light surface, categorical, CVD-safe); the
one slot below 3:1 contrast carries a direct value label on every bar as relief.  PDFs are vector, Type-42 fonts.
