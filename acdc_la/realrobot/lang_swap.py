"""Does the policy read its instruction?  Counterfactual replay on press_button held-out episodes (2026-11-10).

For every held-out step the SAME image and the SAME flow noise (PiperPolicy.reset(episode) seeds it) are run under
each of the 4 instructions.  If the policy ignores language, the 4 predicted chunks are identical.  Reported:
  * err[instr]: whole-chunk mean |pred - ground truth| in degrees (joints 0-5), per instruction given
  * shift:      mean |pred(swapped) - pred(true)| in degrees -- how much the instruction moves the output
  * rank-1:     fraction of episodes whose TRUE instruction gives the lowest whole-episode error
Caveat: train[95%:] is only 5 episodes; this is a direction, not a rate.

  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -m acdc_la.realrobot.lang_swap \
      --checkpoint <merged run dir> --data_root data/real_rlds_delta --out acdc_la/results/press_button_lang_swap.json
"""
import argparse, json, os
import numpy as np

TASKS = ["press the blue button", "press the orange button", "press the pink button", "press the red button"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", required=True); ap.add_argument("--dataset", default="press_button")
    ap.add_argument("--split", default="train[95%:]"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from acdc_la.realrobot.piper_policy import PiperPolicy, _tf
    p = PiperPolicy(a.checkpoint); _tf()
    import tensorflow_datasets as tfds
    ds = tfds.builder_from_directory(os.path.join(a.data_root, a.dataset, "1.0.0")).as_dataset(split=a.split)
    unnorm = lambda ch: np.where(p.norm_mask, 0.5 * (ch + 1) * (p.high - p.low) + p.low, ch)
    rows = []
    for ei, epi in enumerate(ds):
        steps = list(epi["steps"]); imgs = [s["observation"]["image"].numpy() for s in steps]
        acts = np.stack([s["action"].numpy() for s in steps]).astype(np.float64)
        true = steps[0]["language_instruction"].numpy().decode()
        preds = {}
        for instr in TASKS:
            p.reset(episode=ei); out = []
            for t in range(len(steps)):
                out.append(unnorm(p.act_chunk(imgs[t], instr, normalised=True)))
            preds[instr] = out
        err, shift = {}, {}
        for instr in TASKS:
            e = [np.abs(preds[instr][t][:len(acts[t:t + p.W]), :6] - acts[t:t + p.W, :6]).mean() for t in range(len(steps))]
            err[instr] = float(np.mean(e))
            shift[instr] = float(np.mean([np.abs(preds[instr][t][:, :6] - preds[true][t][:, :6]).mean() for t in range(len(steps))]))
        best = min(err, key=err.get)
        rows.append(dict(episode=ei, steps=len(steps), true=true, err=err, shift=shift, best=best))
        print(f"[lang-swap] ep {ei} ({len(steps)} steps) true={true!r} best={best!r}", flush=True)
        for instr in TASKS:
            print(f"    {instr:<26} err {err[instr]:6.3f} deg   shift vs true {shift[instr]:6.3f} deg"
                  + ("   <- true" if instr == true else ""), flush=True)
    true_err = np.mean([r["err"][r["true"]] for r in rows])
    wrong_err = np.mean([np.mean([v for k, v in r["err"].items() if k != r["true"]]) for r in rows])
    wrong_shift = np.mean([np.mean([v for k, v in r["shift"].items() if k != r["true"]]) for r in rows])
    rank1 = np.mean([r["best"] == r["true"] for r in rows])
    summ = dict(episodes=len(rows), true_instr_err_deg=float(true_err), wrong_instr_err_deg=float(wrong_err),
                mean_shift_wrong_vs_true_deg=float(wrong_shift), rank1=float(rank1))
    print(f"\n[lang-swap] {len(rows)} episodes | whole-chunk joint error: TRUE instr {true_err:.3f} deg, "
          f"WRONG instr {wrong_err:.3f} deg | output shift under a wrong instr {wrong_shift:.3f} deg | "
          f"true instr best in {rank1 * 100:.0f} % of episodes")
    if a.out: json.dump(dict(summary=summ, episodes=rows), open(a.out, "w"), indent=1); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
