"""Controlled UniVLA LIBERO fine-tuning: their recipe (vla-scripts/finetune_libero.py, LoRA r32, batch 8 x accum 2 per
GPU, lr 3.5e-4, image aug, latent tokens in the prompt, MAP action decoder over the latent-token states, L1 on a
12-step chunk) with ONE switch, `--latent {univla,acdc}`, that changes the latent-token labels and nothing else.
Both arms start from the same base VLM (`--vla_path`, default openvla-7b), get their `<ACT_i>` vocabulary added and
trained (LoRA modules_to_save = embed_tokens, lm_head), and read the last n_tokens latent states in the decoder.
W&B is replaced by a jsonl log (compute nodes have no network).

  torchrun --standalone --nproc-per-node 4 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent acdc --max_steps 15000
"""
import os, sys, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus, torch, torch.nn as nn, torch.distributed as dist, tqdm
import torchvision.transforms as transforms
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig, AutoImageProcessor

from ...paths import LF, UNIVLA_ROOT, CKPT
sys.path.insert(0, UNIVLA_ROOT)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.datasets import RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.policy.transformer_utils import MAPBlock
from .datasets_ctrl import RLDSBatchTransformCtrl

LATENT_TOKEN_BASE = 32000          # ids > 32000 are latent-action tokens (openvla-7b: <PAD> = 32000, <ACT_i> = 32001 + i)


class ActionDecoder(nn.Module):
    """UniVLA's low-level decoder; `n_tokens` replaces the hard-coded 4."""
    def __init__(self, window_size=12, hidden_dim=512, n_tokens=4):
        super().__init__(); self.n_tokens = n_tokens
        self.latent_action_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden_dim, n_heads=hidden_dim // 64)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, 7 * window_size), nn.Tanh())

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        return self.proj(self.latent_action_pool(latent_action_tokens[:, -self.n_tokens:], init_embed=visual_embed))


class ContHeads(nn.Module):
    """Continuous-latent arm: no latent vocabulary.  The VLM's last hidden state is pooled into a prediction of the
    CONTINUOUS ACDC code, and the action head is conditioned on that code, so the only thing removed relative to the
    token arms is the quantiser.  Mirrors the oracle->predicted curriculum of downstream/libero_policy.py."""
    def __init__(self, dz, window_size=12, hidden=512):
        super().__init__(); self.dz = dz
        self.vis_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.z_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dz)) if dz else None
        self.act = nn.Sequential(nn.Linear(hidden + dz, hidden), nn.GELU(), nn.Linear(hidden, 7 * window_size), nn.Tanh())

    def forward(self, visual, text, z_for_head=None):
        v = self.vis_pool(visual)
        if self.dz == 0: return self.act(v), None            # --latent none: the same head with no latent input at all
        z_pred = self.to_z(self.z_pool(text, init_embed=v))
        z = z_pred if z_for_head is None else z_for_head
        return self.act(torch.cat([v, z], -1)), z_pred


class WrappedModel(nn.Module):
    def __init__(self, vla, window_size=12, n_tokens=4, continuous=False, dz=0):
        super().__init__(); self.vla, self.window_size, self.continuous = vla, window_size, continuous
        if continuous: self.heads = ContHeads(dz, window_size)
        else: self.action_decoder = ActionDecoder(window_size, n_tokens=n_tokens)

    def forward(self, batch, p_oracle=0.0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.vla(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"],
                           labels=None if self.continuous else batch["labels"], output_hidden_states=True)
        n_patches = self.vla.vision_backbone.featurizer.patch_embed.num_patches
        visual_embed = out.hidden_states[-1][:, :n_patches].to(torch.float); latent_tokens = out.hidden_states[-1][:, n_patches:]
        if self.continuous and self.heads.dz == 0:                      # no-latent control: action head only
            pred = self.heads(visual_embed, latent_tokens.to(torch.float))[0].reshape(-1, self.window_size, 7)
            l1 = torch.nn.functional.l1_loss(pred, batch["actions"], reduction="none")
            z0 = torch.zeros((), device=pred.device)
            return out, l1.mean(), l1[:, 0].mean(), z0, z0
        if self.continuous:
            z_star = batch["latent_action_idx"].to(torch.float)
            zh = self.heads(visual_embed, latent_tokens.to(torch.float))[1]
            use_star = (torch.rand(len(z_star), 1, device=z_star.device) < p_oracle).float()
            pred, _ = self.heads(visual_embed, latent_tokens.to(torch.float), z_for_head=use_star * z_star + (1 - use_star) * zh)
            pred = pred.reshape(-1, self.window_size, 7)
            l1 = torch.nn.functional.l1_loss(pred, batch["actions"], reduction="none")
            lat = torch.nn.functional.mse_loss(zh, z_star)
            r2 = 1 - lat / z_star.var(0).mean().clamp_min(1e-6)
            return out, l1.mean(), l1[:, 0].mean(), lat, r2
        mask = batch["labels"] > LATENT_TOKEN_BASE
        lat = torch.stack([latent_tokens[i][mask[i]] for i in range(len(latent_tokens))]).to(torch.float)
        pred = self.action_decoder(lat, visual_embed).reshape(-1, self.window_size, 7)
        l1 = torch.nn.functional.l1_loss(pred, batch["actions"], reduction="none")
        return out, l1.mean(), l1[:, 0].mean(), None, None


@dataclass
class Cfg:
    latent: str = "acdc"                                   # univla | acdc | acdc_cont
    lam_z: float = 1.0                                     # weight of the latent regression loss (continuous arm)
    curriculum_frac: float = 0.6                           # oracle -> predicted anneal, as in downstream/libero_policy.py
    vla_path: str = os.path.join(LF, "cache", "vla", "openvla-7b")
    lam_path: str = os.path.join(LF, "claude_survey", "ckpt", "s2mix_1ep.pt")      # univla arm: their Stage-2 recipe on our data
    acdc_tag: str = "s1_acdc_1ep"; acdc_codebook: str = "vq_s1_acdc_1ep_K64.npz"    # acdc arm
    tc_tag: str = "tc_cont"                                # tc_cont / tc_vq arms: the structured tokenizer checkpoint
    data_root_dir: Path = Path(os.path.join(LF, "data", "libero_rlds")); dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path(os.path.join(LF, "acdc_la", "ckpt", "vla_runs")); run_id_note: Optional[str] = None
    batch_size: int = 8; grad_accumulation_steps: int = 2; max_steps: int = 15000; save_steps: int = 5000; learning_rate: float = 3.5e-4
    image_aug: bool = True; shuffle_buffer_size: int = 16000; window_size: int = 12
    lora_rank: int = 32; lora_dropout: float = 0.0
    log_every: int = 25


def build_labeller(cfg, device_id):
    if cfg.latent == "none":
        return None, 0, 0                                                 # no labeller: the arm never sees a latent
    if cfg.latent in ("tc_cont", "tc_vq"):
        from .tcvq_tokens import TCVQLatentTokenizer
        t = TCVQLatentTokenizer(cfg.tc_tag, dev=f"cuda:{device_id}" if isinstance(device_id, int) else device_id)
        return t, t.n_tokens, t.codebook_size
    if cfg.latent == "acdc_cont":
        from .acdc_tokens import ACDCLatentTokenizer
        t = ACDCLatentTokenizer(cfg.acdc_tag, cfg.acdc_codebook, dev=f"cuda:{device_id}" if isinstance(device_id, int) else device_id)
        return t, 0, 0                                                    # no vocabulary: the head regresses t.dz values
    if cfg.latent == "univla":
        from ...paths import TORCH_HUB
        torch.hub.set_dir(TORCH_HUB)                                     # offline DINO for their LAM (compute nodes have no network)
        from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
        lam = ControllableDINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128, num_latents=16, patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
        ck = torch.load(cfg.lam_path, map_location="cpu", weights_only=False); sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {k.replace("lam.", "", 1): v for k, v in sd.items()}
        res = lam.load_state_dict(sd, strict=False)          # rala2 does not save the frozen DINO encoder (built from the hub cache) and adds lang_proj
        assert all(k.startswith("dino_encoder.") for k in res.missing_keys) and all(k.startswith("lang_proj.") for k in res.unexpected_keys), (res.missing_keys[:5], res.unexpected_keys[:5])
        lam = lam.to(device_id).eval(); return lam, 4, 16          # their class exposes .device as a read-only property
    from .acdc_tokens import ACDCLatentTokenizer
    t = ACDCLatentTokenizer(cfg.acdc_tag, cfg.acdc_codebook, dev=f"cuda:{device_id}" if isinstance(device_id, int) else device_id)
    return t, t.n_tokens, t.codebook_size


@draccus.wrap()
def finetune(cfg: Cfg) -> None:
    assert torch.cuda.is_available(); assert cfg.latent in ("univla", "acdc", "acdc_cont", "none", "tc_cont", "tc_vq")
    state = PartialState(); torch.cuda.set_device(device_id := state.local_process_index); torch.cuda.empty_cache()
    exp_id = f"{cfg.latent}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}x{state.num_processes}+lr{cfg.learning_rate}+r{cfg.lora_rank}+s{cfg.max_steps}" + (f"--{cfg.run_id_note}" if cfg.run_id_note else "")
    run_dir = cfg.run_root_dir / exp_id; adapter_dir = cfg.run_root_dir / "adapters" / exp_id; os.makedirs(run_dir, exist_ok=True); os.makedirs(adapter_dir, exist_ok=True)
    if state.is_main_process: print(f"[ctrl] {exp_id} -> {run_dir}", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    time.sleep(int(os.environ.get("CTRL_STAGGER_SEC", "0")) * state.local_process_index)   # small-RAM hosts: stagger the 7B loads across ranks
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

    labeller, n_tokens, codebook_size = build_labeller(cfg, device_id)
    cont = cfg.latent in ("acdc_cont", "none", "tc_cont"); dz = getattr(labeller, "dz", 0) if cfg.latent in ("acdc_cont", "tc_cont") else 0
    if not cont:
        # the latent vocabulary: <ACT_0..K-1> appended after <PAD> (32000) -> ids 32001..; embeddings + lm_head rows are trained
        assert len(processor.tokenizer) == LATENT_TOKEN_BASE + 1, len(processor.tokenizer)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(codebook_size)]})
        vla.resize_token_embeddings(len(processor.tokenizer))
    vla = vla.to(device_id)
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout, target_modules="all-linear",
                      init_lora_weights="gaussian", modules_to_save=None if cont else ["embed_tokens", "lm_head"])
    vla = get_peft_model(vla, lora); vla.print_trainable_parameters()
    model = WrappedModel(vla, cfg.window_size, n_tokens, continuous=cont, dz=dz).to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(cfg.max_steps * 0.8), gamma=0.1)

    transform = RLDSBatchTransformCtrl(labeller, cfg.latent, processor.tokenizer, image_transform=processor.image_processor.apply_transform,
                                       image_transform_lam=transforms.ToTensor(), prompt_builder_fn=PurePromptBuilder, window_size=cfg.window_size)
    ds = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, transform, resize_resolution=tuple(model.module.vla.config.image_sizes), shuffle_buffer_size=cfg.shuffle_buffer_size,
                     image_aug=cfg.image_aug, window_size=cfg.window_size + 2, training_phase="post-training")
    if state.is_main_process: save_dataset_statistics(ds.dataset_statistics, run_dir); processor.save_pretrained(run_dir)
    collator = PaddedCollatorForActionPrediction_LIBERO(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    n_patches = model.module.vla.vision_backbone.featurizer.patch_embed.num_patches
    log = open(run_dir / "train_log.jsonl", "a") if state.is_main_process else None; t0 = time.time(); acc_buf, l1_buf, loss_buf = [], [], []

    model.train(); opt.zero_grad()
    for bi, batch in enumerate(dl):
        for k in ("input_ids", "attention_mask", "labels", "actions", "latent_action_idx"): batch[k] = batch[k].to(device_id)
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
        p_or = max(0.0, 1.0 - (bi // cfg.grad_accumulation_steps) / (cfg.curriculum_frac * cfg.max_steps)) if model.module.continuous else 0.0
        out, act_loss, loss_1, lat_loss, lat_r2 = model(batch, p_oracle=p_or)
        loss = act_loss + (cfg.lam_z * lat_loss if model.module.continuous else out.loss)
        (loss / cfg.grad_accumulation_steps).backward()
        with torch.no_grad():
            if model.module.continuous: acc_buf.append(float(lat_r2)); loss_buf.append(float(lat_loss))
            else:
                preds = out.logits[:, n_patches:-1].argmax(2); gt = batch["labels"][:, 1:]; m = gt > LATENT_TOKEN_BASE
                acc_buf.append(float(((preds == gt) & m).sum() / m.sum())); loss_buf.append(float(out.loss))
            l1_buf.append(float(act_loss))
        step = bi // cfg.grad_accumulation_steps
        if (bi + 1) % cfg.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step()
            if state.is_main_process and (step + 1) % cfg.log_every == 0:
                key = "latent_r2" if model.module.continuous else "token_acc"
                rec = {"step": step + 1, key: sum(acc_buf) / len(acc_buf), "latent_loss" if model.module.continuous else "token_loss": sum(loss_buf) / len(loss_buf),
                       "action_l1": sum(l1_buf) / len(l1_buf), "p_oracle": p_or, "lr": opt.param_groups[0]["lr"], "seconds": int(time.time() - t0)}
                print(f"[ctrl:{cfg.latent}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                acc_buf, l1_buf, loss_buf = [], [], []
            if (step + 1) % cfg.save_steps == 0 or step + 1 == cfg.max_steps:
                if state.is_main_process:
                    model.module.vla.save_pretrained(adapter_dir)
                    torch.save((model.module.heads if cont else model.module.action_decoder).state_dict(), run_dir / "action_decoder.pt")
                    json.dump(dict(latent=cfg.latent, n_tokens=n_tokens, codebook_size=codebook_size, dz=dz, window_size=cfg.window_size, step=step + 1, vla_path=cfg.vla_path,
                                   lam_path=cfg.lam_path, acdc_tag=cfg.acdc_tag, acdc_codebook=cfg.acdc_codebook), open(run_dir / "ctrl_meta.json", "w"), indent=1)
                    print(f"[ctrl] saved adapter + decoder at step {step + 1}", flush=True)
                dist.barrier()
            if step + 1 >= cfg.max_steps: break
    if state.is_main_process:                                       # merge LoRA into a standalone checkpoint for the evaluator
        base = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        if not cont: base.resize_token_embeddings(len(processor.tokenizer))
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload(); merged.save_pretrained(run_dir)
        print(f"[ctrl-done] merged checkpoint at {run_dir} ({int(time.time() - t0)}s)", flush=True)
    dist.barrier()


if __name__ == "__main__":
    finetune()
