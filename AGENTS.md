# Workspace research logging rules

For work under `latent_free/` (including `claude_survey/` and `codex_survey/`), the project-level
experiment ledger is `LEDGER.md` at the repository root (moved from Anvil's `/anvil/scratch/x-tding2/SUMMARY.md`
on 2026-09-14; `~/Projects/SUMMARY.md` is a symlink to it).  GitHub `main` is the sync point: commit and push
ledger updates.

After every experiment finishes, fails, is invalidated, or changes a method decision, update
that file before giving the user the final report. Record the configuration/split/budget,
short numerical result and control, conclusion status, artifact paths, and resulting next
decision. Do not leave experimental conclusions or registered methods only in chat, logs,
or a subdirectory README. Do not report a planned method as an empirical result.

Preserve old conclusions and mark them retracted/invalid when a later control overturns
them. Detailed subproject summaries remain valid supporting records; `LEDGER.md`
is the cross-project index and current method ledger.
