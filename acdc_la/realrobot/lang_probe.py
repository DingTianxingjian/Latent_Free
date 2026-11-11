"""Does the instruction reach the query-token states?  Same image, two instructions, relative difference per layer."""
import torch, glob, sys, time, gc
torch.set_num_threads(24)
from acdc_la.realrobot.piper_policy import PiperPolicy
import tensorflow_datasets as tfds
ds = tfds.builder_from_directory("data/real_rlds_delta/press_button/1.0.0").as_dataset(split="train[95%:]")
img = next(iter(next(iter(ds))["steps"]))["observation"]["image"].numpy()
for name in sys.argv[1:]:
    run = glob.glob(f"/media/volume/TERRA/vla_runs/*--{name}")[0]
    t0 = time.time(); p = PiperPolicy(run, device="cpu"); p.vla.float(); print("loaded", name, round(time.time() - t0), flush=True)
    H = {}
    for ins in ["press the pink button", "press the blue button"]:
        pr = p._build_prompt(ins); inputs = p.processor(pr, p.preprocess(img))
        if not H: print(repr(pr), inputs["input_ids"][0].tolist(), flush=True)
        inputs = {k: (v.float() if v.dtype.is_floating_point else v) for k, v in inputs.items()}
        with torch.no_grad(): hs = p.vla(**inputs, output_hidden_states=True).hidden_states
        qm = inputs["input_ids"][0] > 32000
        H[ins] = [h[0, p.n_patches:][qm] for h in hs]; print("fwd", round(time.time() - t0), flush=True)
    a, b = H.values()
    for L in list(range(0, len(a), 4)) + [len(a) - 1]:
        print(f"{name} layer {L:2d}: rel diff {((a[L]-b[L]).norm()/a[L].norm()).item():.2e}", flush=True)
    del p, H, a, b; gc.collect()
print("PROBE DONE")
