"""Merge LIBERO evaluations that were split across task-subset jobs (`LIBERO_TASK_IDS`).

A 500-episode evaluation is one 2-6 h job, which on a busy queue waits days; ten 50-episode jobs backfill.  The split
changes nothing measured: each task keeps its 50 initial states, and a task is always evaluated by exactly one job.

    python -m acdc_la.downstream.vla.eval_merge --suite libero_spatial --note latp2inv_frac10 --kind cont
    python -m acdc_la.downstream.vla.eval_merge --suite libero_object  --note frac10          --kind token

`--kind token` reads UniVLA's text logs (`Task:` / `Success:` lines), `--kind cont` the json of
`run_libero_eval_cont.py`.  Prints the per-task table and writes `merged_<suite>_<note>.json`.
"""
import argparse, glob, json, os, re
from collections import OrderedDict


def from_token_logs(paths, trials):
    per = OrderedDict()
    for p in sorted(paths):
        task = None
        for line in open(p):
            if line.startswith("Task: "): task = line[6:].strip()
            elif line.startswith("Success:") and task is not None:
                s, n = per.get(task, (0, 0))
                per[task] = (s + int("True" in line.split("steps:")[0]), n + 1)
    return per


def from_cont_jsons(paths, trials):
    per = OrderedDict()
    for p in sorted(paths):
        d = json.load(open(p))
        t = int(d.get("trials", trials))
        for task, rate in d["per_task"].items():
            s, n = per.get(task, (0, 0))
            per[task] = (s + round(rate * t), n + t)
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True); ap.add_argument("--note", required=True)
    ap.add_argument("--kind", choices=("cont", "token"), required=True)
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "vla_eval"))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--include", choices=("parts", "all"), default="parts",
                    help="'parts': only the task-subset files (…_t3_4 / --t3_4).  'all' also takes whole-suite runs of the "
                         "same note, which DOUBLE-COUNTS a task if both a split part and a full run cover it.")
    a = ap.parse_args(); d = os.path.abspath(a.dir)
    if a.kind == "cont":
        paths = glob.glob(os.path.join(d, f"cont_{a.suite}_{a.note}_t*.json" if a.include == "parts" else f"cont_{a.suite}_{a.note}*.json"))
        per = from_cont_jsons(paths, a.trials)
    else:
        # the token evaluator names parts EVAL-<suite>-openvla-<date>--t0_1--<note>.txt (task tag BEFORE the note)
        paths = [p for p in glob.glob(os.path.join(d, f"EVAL-{a.suite}-*{a.note}*.txt"))
                 if (a.include == "all" or re.search(r"--t\d+(_\d+)*--", os.path.basename(p)))]
        per = from_token_logs(paths, a.trials)
    assert paths, f"no evaluation files for {a.suite} / {a.note} in {d}"
    dup = {t: n for t, (_, n) in per.items() if n > a.trials}
    assert not dup, f"more episodes than trials for {list(dup)[:2]}: overlapping evaluations, check --include and the file list"
    succ = sum(s for s, _ in per.values()); tot = sum(n for _, n in per.values())
    print(f"[merge] {a.suite} {a.note} ({a.kind}): {len(paths)} files, {len(per)} tasks")
    for i, (task, (s, n)) in enumerate(per.items()):
        print(f"  {i:2d} {task[:58]:58s} {s:3d}/{n:3d} = {s / n:.2f}")
    print(f"[merge] TOTAL {succ}/{tot} = {succ / tot:.4f}" + ("" if tot == 10 * a.trials else f"   INCOMPLETE (expected {10 * a.trials})"))
    out = dict(suite=a.suite, note=a.note, kind=a.kind, files=sorted(os.path.basename(p) for p in paths),
               per_task={t: dict(successes=s, episodes=n) for t, (s, n) in per.items()}, successes=succ, episodes=tot,
               success_rate=succ / tot, complete=bool(tot == 10 * a.trials))
    p = os.path.join(d, f"merged_{a.suite}_{a.note}.json"); json.dump(out, open(p, "w"), indent=1); print(f"[merge] wrote {p}")


if __name__ == "__main__":
    main()
