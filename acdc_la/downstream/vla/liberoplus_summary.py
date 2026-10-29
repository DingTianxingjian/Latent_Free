"""LIBERO-Plus per-category summary (2026-10-28).  Reads the shard JSONs written by run_libero_eval_cont with
LIBERO_TASK_SHARD (results/vla_eval_plus/cont_<suite>_plus_<note>_s<k>of<n>.json, field per_task_id) and the
benchmark's task_classification.json (entry i = task index i, id = i + 1, names checked equal to the suite order),
and prints success per perturbation category and suite, the mean over the seven categories, and coverage.

    python -m acdc_la.downstream.vla.liberoplus_summary --note p2ours_full
"""
import argparse, glob, json, os
from collections import defaultdict

CLS = "/lustre/fs1/home/ti874297/lf/envs/LIBERO-plus/libero/libero/benchmark/task_classification.json"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CATS = ("Camera Viewpoints", "Robot Initial States", "Language Instructions", "Light Conditions", "Background Textures",
        "Sensor Noise", "Objects Layout")
SHORT = dict(zip(CATS, ("Camera", "Robot", "Language", "Light", "Background", "Noise", "Layout")))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--note", required=True)
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "results", "vla_eval_plus"))
    a = ap.parse_args(); cls = json.load(open(CLS))
    res = {}                                                   # (suite, task index) -> success
    for s in SUITES:
        for f in glob.glob(os.path.join(a.dir, f"cont_{s}_plus_{a.note}_s*of*.json")):
            for k, v in json.load(open(f)).get("per_task_id", {}).items():
                res[(s, int(k))] = v
    agg = defaultdict(lambda: [0.0, 0])                        # (suite|ALL, category|ALL) -> [successes, episodes]
    for (s, i), v in res.items():
        c = cls[s][i]["category"]
        for key in ((s, c), ("ALL", c), (s, "ALL"), ("ALL", "ALL")):
            agg[key][0] += v; agg[key][1] += 1
    rate = lambda k: (100 * agg[k][0] / agg[k][1]) if agg[k][1] else float("nan")
    print(f"LIBERO-Plus, note={a.note}: {len(res)} / {sum(len(cls[s]) for s in SUITES)} tasks evaluated (1 trial each)")
    print(f"{'suite':16s}" + "".join(f"{SHORT[c]:>11s}" for c in CATS) + f"{'cat-mean':>10s}{'all':>8s}{'n':>7s}")
    out = {}
    for s in SUITES + ("ALL",):
        row = [rate((s, c)) for c in CATS]; done = [r for r in row if r == r]
        cm = sum(done) / len(done) if done else float("nan")
        print(f"{s:16s}" + "".join(f"{r:11.1f}" for r in row) + f"{cm:10.1f}{rate((s, 'ALL')):8.1f}{agg[(s, 'ALL')][1]:7d}")
        out[s] = dict({SHORT[c]: row[j] for j, c in enumerate(CATS)}, category_mean=cm, all=rate((s, "ALL")), n=agg[(s, "ALL")][1])
    json.dump(out, open(os.path.join(a.dir, f"summary_plus_{a.note}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
