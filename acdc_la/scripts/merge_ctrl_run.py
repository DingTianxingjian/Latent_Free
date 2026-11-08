"""Merge a finetune_libero_ctrl LoRA run into a standalone checkpoint, in a clean single process (2026-11-08).

The trainer does this itself at the end of training (finetune_libero_ctrl.py:510-514), and on a large-memory host that
is fine.  On the 62 GB workstation it is not: the merge peaks at ~31 GB while rank 1 is still alive at `dist.barrier()`
holding its tf.data arena, and it was SIGKILLed three times at the tail of the terra_piper run.  So runs on this host
are launched with `--merge False` and merged afterwards with this, which needs 30.7 GB and 3.5 minutes alone.

It reproduces the trainer's merge exactly, including the one thing that is easy to get wrong: the nq `<ACT_i>` query
tokens enter the tokenizer BEFORE `resize_token_embeddings` (finetune_libero_ctrl.py:320-327), so the vocabulary the
adapter was trained against is 32000 + <PAD> + nq, and resizing without them raises a size mismatch.

  envs/univla_vla/bin/python -m acdc_la.scripts.merge_ctrl_run --run <run dir> --init <phase-2 run dir>
  # --adapters defaults to <run root>/adapters/<run basename>, which is where the trainer puts it
"""
import argparse, json, os, resource, sys, time

import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

from ..paths import LF
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

T0 = time.time()


def mark(tag, floor):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    av = int(next(l for l in open("/proc/meminfo") if l.startswith("MemAvailable")).split()[1]) / 1e6
    print(f"[{int(time.time() - T0):5d}s] {tag:<24} peak RSS {rss:5.1f} GB   MemAvailable {av:5.1f} GB", flush=True)
    # this host is shared: stop rather than let the kernel's OOM killer pick a victim
    if av < floor: sys.exit(f"ABORT: only {av:.1f} GB available (floor {floor} GB), refusing to push this host to the OOM killer")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run dir: holds ctrl_meta.json, gets the merged checkpoint")
    ap.add_argument("--adapters", default=None, help="adapter dir (default <run root>/adapters/<run basename>)")
    ap.add_argument("--init", default=None, help="phase-2 run dir whose vla_full.pt the LoRA was trained on")
    ap.add_argument("--vla_path", default=os.path.join(LF, "cache", "vla", "openvla-7b"))
    ap.add_argument("--nq", type=int, default=None, help="query tokens (default: ctrl_meta.json)")
    ap.add_argument("--min_avail_gb", type=float, default=5.0)
    a = ap.parse_args()
    run = a.run.rstrip("/")
    ad = a.adapters or os.path.join(os.path.dirname(run), "adapters", os.path.basename(run))
    assert os.path.exists(os.path.join(ad, "adapter_config.json")), f"no adapter at {ad}"
    meta = json.load(open(os.path.join(run, "ctrl_meta.json")))
    nq = a.nq if a.nq is not None else int(meta.get("query_tokens") or 0)
    cont = meta["latent"] in ("acdc_cont", "none", "tc_cont", "univla_cont")
    full_init = os.path.join(a.init, "vla_full.pt") if a.init else None
    if full_init: assert os.path.exists(full_init), full_init
    print(f"[merge] run {run}\n        adapter {ad}\n        base {full_init or a.vla_path} | latent {meta['latent']} | nq {nq}", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor = AutoProcessor.from_pretrained(a.vla_path, trust_remote_code=True)
    base = AutoModelForVision2Seq.from_pretrained(a.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    mark("base built", a.min_avail_gb)
    if not cont or nq:                                    # same condition and same order as the trainer
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(nq)]})
        base.resize_token_embeddings(len(processor.tokenizer))
        mark(f"resized to {len(processor.tokenizer)}", a.min_avail_gb)
    if full_init:
        base.load_state_dict(torch.load(full_init, map_location="cpu", mmap=True))
        mark("phase-2 base loaded", a.min_avail_gb)
    merged = PeftModel.from_pretrained(base, ad).merge_and_unload()
    mark("merge_and_unload", a.min_avail_gb)
    merged.save_pretrained(run)
    mark("save_pretrained", a.min_avail_gb)
    print(f"[merge] done -> {run} in {int(time.time() - T0)}s", flush=True)


if __name__ == "__main__":
    main()
