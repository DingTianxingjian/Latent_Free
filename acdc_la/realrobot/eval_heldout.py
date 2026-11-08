"""Offline held-out check for the real-robot arm (2026-11-08).

There is no simulator for the Piper data, so a checkpoint cannot be chosen by rollout the way the LIBERO arms are.
What the 95/5 split leaves out -- train[95%:], 8 episodes -- is the only offline signal, so this reports the two
numbers the trainer logs, measured on that split instead of on train batches: the one-step action L1
(`loss_1` of WrappedModel.forward, the flow head's L1, NOT the flow loss) and the latent r2.

It evaluates several snapshots in ONE process: the 7B base and the phase-2 weights are loaded once, and each
snapshot's adapter is attached, measured and detached.  A snapshot is a directory holding adapter_config.json +
adapter_model.safetensors + action_decoder.pt (what `terra_snapshots/step*` holds), because the trainer writes every
save to the SAME adapter dir (finetune_libero_ctrl.py:496) and keeps no history of its own.

Offline L1 is not real-robot success; it ranks checkpoints, it does not predict the robot.

  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -m acdc_la.realrobot.eval_heldout \
      --init /data/ti874297/p2fft_20k --dataset piper_pick_can --data_root data/real_rlds \
      --snapshots /data/ti874297/terra_snapshots/step6000 ... --frames 1024
"""
import argparse, json, os, time
import torch, torchvision.transforms as transforms
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
from torch.utils.data import DataLoader

from ..paths import LF
from ..downstream.vla.finetune_libero_ctrl import WrappedModel, build_labeller
from ..downstream.vla.datasets_ctrl import RLDSBatchTransformCtrl
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.datasets import RLDSDataset


class Cfg:                                        # the handful of fields build_labeller reads
    latent, tc_tag, acdc_tag, acdc_codebook, lam_path = "tc_cont", "latp2_inv", None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="phase-2 run dir holding vla_full.pt (the LoRA's base)")
    ap.add_argument("--snapshots", nargs="+", required=True)
    ap.add_argument("--dataset", default="piper_pick_can")
    ap.add_argument("--data_root", default=os.path.join(LF, "data", "real_rlds"))
    ap.add_argument("--vla_path", default=os.path.join(LF, "cache", "vla", "openvla-7b"))
    ap.add_argument("--window_size", type=int, default=12)
    ap.add_argument("--query_tokens", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    # train=False takes `frames` frames and caches them (rlds/dataset.py:624-626) off a repeated stream, so with a
    # val split of ~325 frames every frame is drawn several times: coverage is effectively the whole split for every
    # snapshot, which is what makes the snapshots comparable.
    ap.add_argument("--frames", type=int, default=1024)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(a.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(a.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    # finetune_libero_ctrl.py:320-327: the nq placeholders enter the tokenizer BEFORE the resize -> vocabulary 32005
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(a.query_tokens)]})
    vla.resize_token_embeddings(len(processor.tokenizer))
    vla.load_state_dict(torch.load(os.path.join(a.init, "vla_full.pt"), map_location="cpu", mmap=True))
    print(f"[{int(time.time()-t0)}s] base + phase-2 weights ready (vocab {len(processor.tokenizer)})", flush=True)

    labeller, n_tokens, _ = build_labeller(Cfg(), 0)
    dz = getattr(labeller, "dz", 0)
    transform = RLDSBatchTransformCtrl(labeller, "tc_cont", processor.tokenizer,
                                       image_transform=processor.image_processor.apply_transform,
                                       image_transform_lam=transforms.ToTensor(), prompt_builder_fn=PurePromptBuilder,
                                       window_size=a.window_size, query_tokens=a.query_tokens)
    ds = RLDSDataset(a.data_root, a.dataset, transform, resize_resolution=tuple(vla.config.image_sizes),
                     shuffle_buffer_size=a.frames, window_size=a.window_size + 2, train=False,
                     image_aug=False, training_phase="post-training")
    collator = PaddedCollatorForActionPrediction_LIBERO(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    dl = DataLoader(ds, batch_size=a.batch_size, sampler=None, collate_fn=collator, num_workers=0)

    rows = []
    for snap in a.snapshots:
        step = json.load(open(os.path.join(snap, "ctrl_meta.json")))["step"] if os.path.exists(os.path.join(snap, "ctrl_meta.json")) else os.path.basename(snap)
        peft = PeftModel.from_pretrained(vla, snap, is_trainable=False)
        model = WrappedModel(peft, a.window_size, n_tokens, continuous=True, dz=dz, head="flow",
                             flow_kw=dict(width=512, layers=4, z_tok=32), query_tokens=a.query_tokens).to(0)
        res = model.heads.load_state_dict(torch.load(os.path.join(snap, "action_decoder.pt"), map_location="cpu"), strict=False)
        assert not res.unexpected_keys, res.unexpected_keys
        model.eval()
        l1s, r2s, n = [], [], 0
        with torch.no_grad():
            for batch in dl:
                for k in ("input_ids", "attention_mask", "labels", "actions", "latent_action_idx"): batch[k] = batch[k].to(0)
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(0)
                _, _, loss_1, _, r2 = model(batch, p_oracle=0.0)       # flow head ignores p_oracle (:180)
                l1s.append(float(loss_1)); r2s.append(float(r2)); n += len(batch["input_ids"])
                if n >= a.frames: break
        row = dict(snapshot=snap, step=step, frames=n, action_l1_1step=sum(l1s) / len(l1s), latent_r2=sum(r2s) / len(r2s))
        rows.append(row); print(f"[{int(time.time()-t0)}s] step {step}: action_l1_1step {row['action_l1_1step']:.4f}  latent_r2 {row['latent_r2']:.4f}  ({n} frames)", flush=True)
        del model
        vla = peft.unload()                                            # strip the LoRA layers, keep the base for the next snapshot
        torch.cuda.empty_cache()

    print("\nstep      action_l1_1step   latent_r2")
    for r in sorted(rows, key=lambda r: r["step"] if isinstance(r["step"], int) else 0):
        print(f"{str(r['step']):<9} {r['action_l1_1step']:>10.4f}   {r['latent_r2']:>9.4f}")
    best = min(rows, key=lambda r: r["action_l1_1step"])
    print(f"\nlowest held-out one-step action L1: step {best['step']} ({best['action_l1_1step']:.4f})")
    if a.out: json.dump(rows, open(a.out, "w"), indent=1); print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
