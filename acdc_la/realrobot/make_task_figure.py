"""Figure-ready strips from the frames extract_paper_frames.py wrote (2026-11-13).

Two figures, both at the frames' native 640x480 per tile so they survive a print column:

  fig_tasks.png            one row per TASK, time left -> right: what the 4 tasks look like.
  fig_press_button.png     one row per COLOUR of press_button: the instruction and the layout both change,
                           which is the whole reason that task separates a language-using policy from one that
                           cannot read the word (LEDGER.md 2026-11-12).

  envs/univla_vla/bin/python -m acdc_la.realrobot.make_task_figure --frames acdc_la/results/paper_frames
"""
import argparse, json, os

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TITLES = {"pick_can_place_plate": "pick can", "press_button": "press button",
          "stack_bowls": "stack bowls", "put_spoon_in_bowl": "put spoon in bowl"}


def strip(rows, dst, pad=8, lab_h=46, scale=1.0, caption=None):
    """rows: list of (bold label, grey label, [image paths]) -- one row each, time left to right."""
    w, h = Image.open(rows[0][2][0]).size
    w, h = int(w * scale), int(h * scale)
    n = max(len(r[2]) for r in rows)
    fb, fn = ImageFont.truetype(FONT_B, 28), ImageFont.truetype(FONT, 24)
    cap_h = 40 if caption else 0
    im = Image.new("RGB", (n * w + (n + 1) * pad, len(rows) * (h + lab_h) + pad + cap_h), "white")
    d = ImageDraw.Draw(im)
    for i, (bold, grey, fs) in enumerate(rows):
        y = pad + i * (h + lab_h)
        d.text((pad, y), bold, fill=(0, 0, 0), font=fb)
        if grey: d.text((pad + d.textlength(bold, font=fb) + 14, y + 3), grey, fill=(110, 110, 110), font=fn)
        for j, f in enumerate(fs):
            im.paste(Image.open(f).resize((w, h), Image.LANCZOS), (pad + j * (w + pad), y + lab_h))
    if caption:
        d.text((pad, im.height - cap_h + 6), caption, fill=(110, 110, 110), font=fn)
    im.save(dst); print(f"wrote {dst}  {im.size[0]}x{im.size[1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="acdc_la/results/paper_frames")
    ap.add_argument("--cam", default="front", choices=["front", "wrist"])
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 keeps the native 640x480 per tile")
    a = ap.parse_args()
    idx = {}
    for t in TITLES:
        p = os.path.join(a.frames, f"{t}_index.json")
        if os.path.exists(p): idx[t] = json.load(open(p))

    def files(t, ep):
        r = sorted([r for r in idx[t] if r["camera"] == a.cam and r["episode"] == ep], key=lambda r: r["file"])
        return [os.path.join(a.frames, x["file"]) for x in r]

    rows = []
    for t, recs in idx.items():
        ep = sorted({r["episode"] for r in recs})[0]
        instr = next(r["instruction"] for r in recs if r["episode"] == ep)
        rows.append((TITLES[t], f'"{instr}"', files(t, ep)))
    strip(rows, os.path.join(a.frames, f"fig_tasks_{a.cam}.png"), scale=a.scale,
          caption="AgileX Piper, 30 Hz teleoperation subsampled to 10 Hz; frames equally spaced over one episode.")

    if "press_button" in idx:
        pb = []
        for ep in sorted({r["episode"] for r in idx["press_button"]}):
            instr = next(r["instruction"] for r in idx["press_button"] if r["episode"] == ep)
            pb.append((instr.replace("press the ", "").replace(" button", ""), f'"{instr}"', files("press_button", ep)))
        strip(pb, os.path.join(a.frames, f"fig_press_button_{a.cam}.png"), scale=a.scale,
              caption="One episode per colour. The layout is reshuffled every episode, so the colour word is the "
                      "only thing that says which button to press.")


if __name__ == "__main__":
    main()
