"""Controlled UniVLA LIBERO fine-tuning: their recipe (vla-scripts/finetune_libero.py, LoRA r32, batch 8 x accum 2 per
GPU, lr 3.5e-4, image aug, latent tokens in the prompt, MAP action decoder over the latent-token states, L1 on a
12-step chunk) with ONE switch, `--latent {univla,acdc}`, that changes the latent-token labels and nothing else.
Both arms start from the same base VLM (`--vla_path`, default openvla-7b), get their `<ACT_i>` vocabulary added and
trained (LoRA modules_to_save = embed_tokens, lm_head), and read the last n_tokens latent states in the decoder.
W&B is replaced by a jsonl log (compute nodes have no network).

  torchrun --standalone --nproc-per-node 4 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent acdc --max_steps 15000
"""
import os, sys, json, time, math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist, tqdm
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
    token arms is the quantiser.  Mirrors the oracle->predicted curriculum of downstream/libero_policy.py.

    LANGUAGE PATH (found 2026-09-18): the VLM input is [BOS, image, text] under a causal mask, so `visual` (image
    positions) never sees the instruction.  With text_direct=False the action head reads [visual summary, z] and the
    instruction reaches it ONLY through the regressed z (for dz = 0, not at all).  text_direct=True (`--head reg_text`)
    adds a pooled summary of the text positions to the action head's input and changes nothing else: the control that
    isolates that bottleneck from the regression-vs-flow change."""
    def __init__(self, dz, window_size=12, hidden=512, text_direct=False):
        super().__init__(); self.dz, self.text_direct = dz, text_direct
        self.vis_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.z_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.t_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64) if text_direct else None
        self.to_z = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dz)) if dz else None
        self.act = nn.Sequential(nn.Linear(hidden * (2 if text_direct else 1) + dz, hidden), nn.GELU(), nn.Linear(hidden, 7 * window_size), nn.Tanh())

    def forward(self, visual, text, z_for_head=None):
        v = self.vis_pool(visual)
        u = [v, self.t_pool(text, init_embed=v)] if self.text_direct else [v]
        if self.dz == 0: return self.act(torch.cat(u, -1)), None     # --latent none: the same head with no latent input
        z_pred = self.to_z(self.z_pool(text, init_embed=v))
        z = z_pred if z_for_head is None else z_for_head
        return self.act(torch.cat(u + [z], -1)), z_pred


class FlowHeads(nn.Module):
    """Continuous arm with a FLOW-MATCHING head (2026-09-18): the action chunk and the continuous latent are generated
    jointly from the VLM's visual and prompt states, as RotVLA's head "jointly denoises latent and robot actions".

    Why it replaces ContHeads: ContHeads regresses the chunk with an L1/MSE target, the behaviour-cloning setting LAFP
    names as the cause of the policy-side failure ("collapse inherently multimodal action distributions into unimodal
    ones").  Generating z and the actions together also avoids LAFP's other named pitfall, a separately trained action
    decoder misaligned with a stochastic latent policy.  Rectified flow: x_t = (1-t) x0 + t x1, x0 ~ N(0, I), target
    velocity x1 - x0; sampling is `steps` Euler steps from noise.  Conditioning is the same MAP pooling of visual and
    prompt states that ContHeads and UniVLA's ActionDecoder use; the denoiser is ~12M parameters, the size of theirs.
    dz == 0 is the no-latent control through the identical head."""
    def __init__(self, dz, window_size=12, hidden=512, width=512, layers=4, z_tok=32):
        super().__init__(); self.dz, self.W, self.z_tok = dz, window_size, z_tok
        assert dz % z_tok == 0, (dz, z_tok)
        self.nz = dz // z_tok
        self.vis_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.txt_pool = MAPBlock(n_latents=1, vis_dim=4096, embed_dim=hidden, n_heads=hidden // 64)
        self.c_in = nn.Linear(2 * hidden, width)
        self.t_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.a_in, self.a_out = nn.Linear(7, width), nn.Linear(width, 7)
        self.z_in, self.z_out = (nn.Linear(z_tok, width), nn.Linear(width, z_tok)) if dz else (None, None)
        self.pos = nn.Parameter(torch.randn(1, 1 + window_size + self.nz, width) * 0.02)
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(width, width // 64, 4 * width, 0.0, batch_first=True,
                                                                       norm_first=True), layers)
        self.norm = nn.LayerNorm(width)

    def cond(self, visual, text):
        v = self.vis_pool(visual)
        return self.c_in(torch.cat([v, self.txt_pool(text, init_embed=v)], -1))

    def _temb(self, t, width):
        f = torch.exp(-math.log(10000.0) * torch.arange(width // 2, device=t.device) / (width // 2))
        a = 1000.0 * t[:, None] * f[None]
        return self.t_mlp(torch.cat([a.sin(), a.cos()], -1))

    def velocity(self, xa, xz, t, c):
        h = [c[:, None], self.a_in(xa)] + ([self.z_in(xz)] if self.dz else [])
        h = torch.cat(h, 1) + self.pos + self._temb(t, c.shape[-1])[:, None]
        h = self.norm(self.blocks(h))
        return self.a_out(h[:, 1:1 + self.W]), (self.z_out(h[:, 1 + self.W:]) if self.dz else None)

    def loss(self, visual, text, actions, z_star=None):
        """-> (action flow loss, latent flow loss, one-step action L1, one-step latent R2); the last two are diagnostics."""
        c = self.cond(visual, text); B = actions.shape[0]
        t = torch.rand(B, device=actions.device)
        a0 = torch.randn_like(actions); at = (1 - t[:, None, None]) * a0 + t[:, None, None] * actions
        z1 = z_star.reshape(B, self.nz, self.z_tok) if self.dz else None
        z0 = torch.randn_like(z1) if self.dz else None
        zt = (1 - t[:, None, None]) * z0 + t[:, None, None] * z1 if self.dz else None
        va, vz = self.velocity(at, zt, t, c)
        la = F.mse_loss(va, actions - a0)
        with torch.no_grad():
            l1 = (at + (1 - t[:, None, None]) * va - actions).abs().mean()           # x1 estimated in one step
        if not self.dz:
            z0_ = torch.zeros((), device=actions.device); return la, z0_, l1, z0_
        lz = F.mse_loss(vz, z1 - z0)
        with torch.no_grad():
            zh = (zt + (1 - t[:, None, None]) * vz).reshape(B, -1)
            r2 = 1 - F.mse_loss(zh, z_star) / z_star.var(0).mean().clamp_min(1e-6)
        return la, lz, l1, r2

    @torch.no_grad()
    def sample(self, visual, text, steps=10, generator=None):
        c = self.cond(visual, text); B, dev = c.shape[0], c.device
        xa = torch.randn(B, self.W, 7, device=dev, generator=generator)
        xz = torch.randn(B, self.nz, self.z_tok, device=dev, generator=generator) if self.dz else None
        for k in range(steps):
            t = torch.full((B,), k / steps, device=dev)
            va, vz = self.velocity(xa, xz, t, c)
            xa = xa + va / steps
            if self.dz: xz = xz + vz / steps
        return xa.clamp(-1, 1), (xz.reshape(B, -1) if self.dz else None)


class WrappedModel(nn.Module):
    def __init__(self, vla, window_size=12, n_tokens=4, continuous=False, dz=0, head="reg", flow_kw=None):
        super().__init__(); self.vla, self.window_size, self.continuous = vla, window_size, continuous
        self.flow = continuous and head == "flow"
        if self.flow: self.heads = FlowHeads(dz, window_size, **(flow_kw or {}))
        elif continuous: self.heads = ContHeads(dz, window_size, text_direct=(head == "reg_text"))
        else: self.action_decoder = ActionDecoder(window_size, n_tokens=n_tokens)

    def forward(self, batch, p_oracle=0.0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.vla(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], pixel_values=batch["pixel_values"],
                           labels=None if self.continuous else batch["labels"], output_hidden_states=True)
        n_patches = self.vla.vision_backbone.featurizer.patch_embed.num_patches
        visual_embed = out.hidden_states[-1][:, :n_patches].to(torch.float); latent_tokens = out.hidden_states[-1][:, n_patches:]
        if self.flow:                                                   # flow-matching head: p_oracle has no role
            z_star = batch["latent_action_idx"].to(torch.float) if self.heads.dz else None
            la, lz, l1, r2 = self.heads.loss(visual_embed, latent_tokens.to(torch.float), batch["actions"].to(torch.float), z_star)
            return out, la, l1, lz, r2
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
    head: str = "reg"                                      # continuous arms: reg (ContHeads) | reg_text (+ direct text) | flow (FlowHeads)
    flow_width: int = 512; flow_layers: int = 4; flow_z_tok: int = 32; flow_steps: int = 10   # flow head; steps used at evaluation
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
    if cfg.latent in ("univla", "univla_cont"):
        from ...paths import TORCH_HUB
        torch.hub.set_dir(TORCH_HUB)                                     # offline DINO for their LAM (compute nodes have no network)
        from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
        lam = ControllableDINOLatentActionModel(in_dim=3, model_dim=768, latent_dim=128, num_latents=16, patch_size=14, enc_blocks=12, dec_blocks=12, num_heads=12, dropout=0.)
        ck = torch.load(cfg.lam_path, map_location="cpu", weights_only=False); sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {k.replace("lam.", "", 1): v for k, v in sd.items()}
        res = lam.load_state_dict(sd, strict=False)          # rala2 does not save the frozen DINO encoder (built from the hub cache) and adds lang_proj
        assert all(k.startswith("dino_encoder.") for k in res.missing_keys) and all(k.startswith("lang_proj.") for k in res.unexpected_keys), (res.missing_keys[:5], res.unexpected_keys[:5])
        lam = lam.to(device_id).eval()
        if cfg.latent == "univla_cont":
            # 2026-09-18, suspect 2 (interface, not latent): UniVLA's DEPLOYED latent — the vq_action codebook vectors of its 4
            # tokens (z_q, 4 x 128) — as a continuous target for the same head our latent uses.  Same information as the 4
            # tokens, delivered through our interface instead of their token pathway.  Standardised per dimension by the
            # spread of the 16 codebook entries (uniform over entries: z_q only ever takes those values).
            cb = lam.vq_action.codebook.weight.detach().float()
            lam.zmean, lam.zstd = cb.mean(0).repeat(4).to(device_id), (cb.std(0) + 1e-6).repeat(4).to(device_id)
            lam.dz = 4 * cb.shape[1]
            return lam, 0, 0
        return lam, 4, 16          # their class exposes .device as a read-only property
    from .acdc_tokens import ACDCLatentTokenizer
    t = ACDCLatentTokenizer(cfg.acdc_tag, cfg.acdc_codebook, dev=f"cuda:{device_id}" if isinstance(device_id, int) else device_id)
    return t, t.n_tokens, t.codebook_size


@draccus.wrap()
def finetune(cfg: Cfg) -> None:
    # Per-job re-targeting (2026-09-16).  Slurm freezes a pending job's environment and `scontrol top` is denied to users
    # on Anvil, so changing what an already-queued job trains without giving up its queue age goes through a file named
    # by its job id.  Applied before anything reads cfg; printed, and recorded in ctrl_meta.json.
    ov = Path(__file__).parent / "job_overrides" / f"{os.environ.get('SLURM_JOB_ID', 'none')}.json"
    ov_d = json.load(open(ov)) if ov.exists() else {}
    for k, v in ov_d.items():
        assert hasattr(cfg, k) and not isinstance(getattr(cfg, k), bool), f"override key {k!r} is not a non-bool Cfg field"
        setattr(cfg, k, v if getattr(cfg, k) is None else type(getattr(cfg, k))(v))
    if ov_d: print(f"[ctrl] OVERRIDE {ov}: {ov_d}", flush=True)
    assert torch.cuda.is_available(); assert cfg.latent in ("univla", "univla_cont", "acdc", "acdc_cont", "none", "tc_cont", "tc_vq")
    state = PartialState(); torch.cuda.set_device(device_id := state.local_process_index); torch.cuda.empty_cache()
    exp_id = f"{cfg.latent}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}x{state.num_processes}+lr{cfg.learning_rate}+r{cfg.lora_rank}+s{cfg.max_steps}" + {"flow": "+fm", "reg_text": "+rt"}.get(cfg.head, "") + (f"--{cfg.run_id_note}" if cfg.run_id_note else "")
    run_dir = cfg.run_root_dir / exp_id; adapter_dir = cfg.run_root_dir / "adapters" / exp_id; os.makedirs(run_dir, exist_ok=True); os.makedirs(adapter_dir, exist_ok=True)
    if state.is_main_process: print(f"[ctrl] {exp_id} -> {run_dir}", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    time.sleep(int(os.environ.get("CTRL_STAGGER_SEC", "0")) * state.local_process_index)   # small-RAM hosts: stagger the 7B loads across ranks
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

    labeller, n_tokens, codebook_size = build_labeller(cfg, device_id)
    cont = cfg.latent in ("acdc_cont", "none", "tc_cont", "univla_cont"); dz = getattr(labeller, "dz", 0) if cfg.latent in ("acdc_cont", "tc_cont", "univla_cont") else 0
    if not cont:
        # the latent vocabulary: <ACT_0..K-1> appended after <PAD> (32000) -> ids 32001..; embeddings + lm_head rows are trained
        assert len(processor.tokenizer) == LATENT_TOKEN_BASE + 1, len(processor.tokenizer)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(codebook_size)]})
        vla.resize_token_embeddings(len(processor.tokenizer))
    vla = vla.to(device_id)
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout, target_modules="all-linear",
                      init_lora_weights="gaussian", modules_to_save=None if cont else ["embed_tokens", "lm_head"])
    vla = get_peft_model(vla, lora); vla.print_trainable_parameters()
    assert cfg.head in ("reg", "reg_text", "flow") and (cfg.head == "reg" or cont), (cfg.head, cfg.latent)
    model = WrappedModel(vla, cfg.window_size, n_tokens, continuous=cont, dz=dz, head=cfg.head,
                         flow_kw=dict(width=cfg.flow_width, layers=cfg.flow_layers, z_tok=cfg.flow_z_tok)).to(device_id)
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
            l1_buf.append(float(loss_1 if cfg.head == "flow" else act_loss))   # flow: one-step action L1, not the flow loss
        step = bi // cfg.grad_accumulation_steps
        if (bi + 1) % cfg.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(); sched.step()
            if state.is_main_process and (step + 1) % cfg.log_every == 0:
                key = "latent_r2" if model.module.continuous else "token_acc"
                rec = {"step": step + 1, key: sum(acc_buf) / len(acc_buf), "latent_loss" if model.module.continuous else "token_loss": sum(loss_buf) / len(loss_buf),
                       ("action_l1_1step" if cfg.head == "flow" else "action_l1"): sum(l1_buf) / len(l1_buf), "p_oracle": p_or, "lr": opt.param_groups[0]["lr"], "seconds": int(time.time() - t0)}
                print(f"[ctrl:{cfg.latent}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                acc_buf, l1_buf, loss_buf = [], [], []
            if (step + 1) % cfg.save_steps == 0 or step + 1 == cfg.max_steps:
                if state.is_main_process:
                    model.module.vla.save_pretrained(adapter_dir)
                    torch.save((model.module.heads if cont else model.module.action_decoder).state_dict(), run_dir / "action_decoder.pt")
                    json.dump(dict(latent=cfg.latent, n_tokens=n_tokens, codebook_size=codebook_size, dz=dz, window_size=cfg.window_size, step=step + 1, vla_path=cfg.vla_path,
                                   lam_path=cfg.lam_path, acdc_tag=cfg.acdc_tag, acdc_codebook=cfg.acdc_codebook, tc_tag=cfg.tc_tag, dataset_name=cfg.dataset_name,
                                   train_frac=os.environ.get("LIBERO_TRAIN_FRAC"), override=ov_d, head=cfg.head,
                                   flow=dict(width=cfg.flow_width, layers=cfg.flow_layers, z_tok=cfg.flow_z_tok, steps=cfg.flow_steps)), open(run_dir / "ctrl_meta.json", "w"), indent=1)
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
