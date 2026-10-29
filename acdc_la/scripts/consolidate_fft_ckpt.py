"""Turn a full-parameter phase-2 run's SHARDED checkpoint (fsdp_ckpt/, torch.distributed.checkpoint, written by
finetune_libero_ctrl.py --full_ft) into a run dir that post-training can start from: vla_full.pt (bf16 VLA state
dict) + action_decoder.pt (heads) + ctrl_meta.json / processor / dataset statistics -- the same files --full_ft writes
at its last step, so `--init_from_run <out>` works mid-training (2026-10-27: intermediate check of the UniVLA-scale run).

CPU only, one process (dcp no_dist=True); needs ~35 GB RAM for the fp32 skeleton.  Snapshot the checkpoint first with
hard links (`cp -al fsdp_ckpt <snap>`) so the trainer's next save cannot delete files mid-read.

    python acdc_la/scripts/consolidate_fft_ckpt.py --ckpt <snap dir> --run <training run dir> --out <new run dir>
"""
import argparse, json, os, shutil, sys, time
import torch
import torch.distributed.checkpoint as dcp

LF = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [LF, os.path.join(LF, "third_party", "UniVLA")]
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from acdc_la.downstream.vla.finetune_libero_ctrl import WrappedModel

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="sharded checkpoint dir (a hard-link snapshot of <run>/fsdp_ckpt)")
ap.add_argument("--run", required=True, help="the training run dir (ctrl_meta.json, processor, dataset_statistics.json)")
ap.add_argument("--out", required=True)
ap.add_argument("--step", type=int, required=True, help="the step the snapshot holds (recorded in ctrl_meta.json)")
a = ap.parse_args()
t0 = time.time()
meta = json.load(open(os.path.join(a.run, "ctrl_meta.json")))
AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
processor = AutoProcessor.from_pretrained(a.run, trust_remote_code=True)          # the run's tokenizer has the <ACT_i> placeholders
vla = AutoModelForVision2Seq.from_pretrained(meta["vla_path"], torch_dtype=torch.float32, low_cpu_mem_usage=True, trust_remote_code=True)
vla.resize_token_embeddings(len(processor.tokenizer))
model = WrappedModel(vla, meta["window_size"], meta["n_tokens"], continuous=True, dz=meta["dz"], head=meta["head"],
                     flow_kw=dict(width=meta["flow"]["width"], layers=meta["flow"]["layers"], z_tok=meta["flow"]["z_tok"]),
                     query_tokens=meta["query_tokens"])
sd = {"model": model.state_dict()}
print(f"[consolidate] skeleton: {len(sd['model'])} tensors, loading {a.ckpt} ({int(time.time() - t0)} s)", flush=True)
dcp.load_state_dict(sd, storage_reader=dcp.FileSystemReader(a.ckpt), no_dist=True)
model.load_state_dict(sd["model"])
os.makedirs(a.out, exist_ok=True)
vsd = {k: v.to(torch.bfloat16) for k, v in model.vla.state_dict().items()}
torch.save(vsd, os.path.join(a.out, "vla_full.pt"))
torch.save(model.heads.state_dict(), os.path.join(a.out, "action_decoder.pt"))
for f in os.listdir(a.run):
    if f.endswith(".json") and f != "ctrl_meta.json" or f.endswith(".py") or f.startswith("tokenizer") or f == "dataset_statistics.json":
        shutil.copy2(os.path.join(a.run, f), a.out)
meta.update(step=a.step, consolidated_from=os.path.abspath(a.ckpt)); json.dump(meta, open(os.path.join(a.out, "ctrl_meta.json"), "w"), indent=1)
fin = [k for k, v in vsd.items() if not torch.isfinite(v.float()).all()]
print(f"[consolidate] wrote {a.out}: vla {len(vsd)} tensors (non-finite: {len(fin)}), heads {len(model.heads.state_dict())}, "
      f"step {a.step} ({int(time.time() - t0)} s)", flush=True)
assert not fin, fin[:5]
