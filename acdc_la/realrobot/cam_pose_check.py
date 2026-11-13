"""Is the front camera in the same place across the four real Piper tasks? (2026-11-13)

Collaborator report (Anh Dao, 2026-11-13, before a robot session): her own 4-task joint model works on every task
EXCEPT pick-and-place-can, and "the cam position is quite diff" for the can data, which she did not record herself.
This checks that claim on OUR copy of the data instead of taking it on trust.

Method: phase-correlate the TOP `--rows` rows of the native 640x480 front frames -- wall, cable and the robot's
mount plate, i.e. the parts of the scene that do NOT change with the task's objects (those sit on the table, lower
in the frame).  Reported per pair: the integer (dy, dx) that best aligns them, the normalised correlation PEAK
(how translation-like the difference is at all: a re-aimed camera is not a shift, so the peak collapses) and the
mean |difference| after aligning.  Episodes of the SAME task are the control for "how much does it move anyway".

  envs/univla_vla/bin/python acdc_la/realrobot/cam_pose_check.py          # defaults: the committed paper frames
"""
import argparse, glob, itertools, os
import numpy as np

TASKS = {"pick_can": "pick_can_place_plate", "stack_bowls": "stack_bowls",
         "press_button": "press_button", "put_spoon": "put_spoon_in_bowl"}


def top(path, rows):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"), np.float64)[:rows]
    return a - a.mean()


def align(a, b):
    """(dy, dx) moving b onto a, plus the normalised-cross-power peak (1.0 = a pure translation)."""
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    r = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
    iy, ix = np.unravel_index(np.argmax(r), r.shape)
    H, W = r.shape
    return (iy - H if iy > H // 2 else iy, ix - W if ix > W // 2 else ix), float(r.max())


def report(pairs, get):
    for a, b in pairs:
        (dy, dx), peak = align(get(a), get(b))
        bb = np.roll(np.roll(get(b), dy, 0), dx, 1)
        resid = np.abs(get(a) - bb)[20:-20, 20:-20].mean()
        print(f"  {a:34s} vs {b:34s}  dy={dy:+4d} dx={dx:+4d} px   peak={peak:.3f}   residual={resid:5.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="acdc_la/results/paper_frames")
    p.add_argument("--rows", type=int, default=140, help="top rows to use: background only, no task objects")
    p.add_argument("--pattern", default="{prefix}_ep*_front_0of5.png")
    a = p.parse_args()
    cache = {}
    get = lambda f: cache.setdefault(f, top(f, a.rows))
    frames = {k: sorted(glob.glob(os.path.join(a.dir, a.pattern.format(prefix=v)))) for k, v in TASKS.items()}
    for k, v in frames.items():
        print(f"{k:13s} {len(v)} episode(s)")
    for k, v in frames.items():
        if len(v) > 1:
            print(f"\nWITHIN {k} ({len(v)} episodes, same session) -- the control:")
            report(itertools.combinations([os.path.basename(x) for x in v], 2),
                   lambda f: get(os.path.join(a.dir, f)))
    print("\nBETWEEN tasks (first episode of each):")
    first = {k: v[0] for k, v in frames.items() if v}
    report(itertools.combinations(first, 2), lambda k: get(first[k]))


if __name__ == "__main__":
    main()
