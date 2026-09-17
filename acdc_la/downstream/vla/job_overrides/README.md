Per-job overrides for `finetune_libero_ctrl.py`, keyed by Slurm job id (see the comment at the top of `finetune()`).

| job | submitted as | re-targeted to | why |
|---|---|---|---|
| 20754114 | `tc_cont` @ 10 % spatial (`--tc_tag tc_cont --run_id_note frac10`) | `latp2_inv` @ 10 % spatial | 2026-09-16, user: the method arm gets the earliest 4-GPU slot; ablations deprioritised |
| 20775018 | `latp2_inv` @ 10 % spatial | `univla` @ 10 % **libero_object** | its content moved to 20754114; slot reused for the main-table baseline |
