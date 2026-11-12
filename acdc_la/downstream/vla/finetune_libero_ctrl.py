"""Controlled UniVLA LIBERO fine-tuning: their recipe (vla-scripts/finetune_libero.py, LoRA r32, batch 8 x accum 2 per
GPU, lr 3.5e-4, image aug, latent tokens in the prompt, MAP action decoder over the latent-token states, L1 on a
12-step chunk) with ONE switch, `--latent {univla,acdc}`, that changes the latent-token labels and nothing else.
Both arms start from the same base VLM (`--vla_path`, default openvla-7b), get their `<ACT_i>` vocabulary added and
trained (LoRA modules_to_save = embed_tokens, lm_head), and read the last n_tokens latent states in the decoder.
W&B is replaced by a jsonl log (compute nodes have no network).

  torchrun --standalone --nproc-per-node 4 -m acdc_la.downstream.vla.finetune_libero_ctrl --latent acdc --max_steps 15000
"""
import os, sys, gc, json, time, math, subprocess
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist, tqdm
import torchvision.transforms as transforms
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
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


from .heads_ctrl import ContHeads, FlowHeads      # moved verbatim 2026-11-09 (importable without the training stack)


class WrappedModel(nn.Module):
    def __init__(self, vla, window_size=12, n_tokens=4, continuous=False, dz=0, head="reg", flow_kw=None, query_tokens=0):
        super().__init__(); self.vla, self.window_size, self.continuous, self.nq = vla, window_size, continuous, query_tokens
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
        if self.continuous and self.nq:
            # QUERY-TOKEN interface (2026-09-18, the OpenVLA-OFT idea): the heads read the states of nq placeholder tokens
            # placed AFTER the instruction (where OpenVLA emits its action tokens and UniVLA its latent tokens), instead of
            # pooling the prompt.  Index j of `latent_tokens` is input token j (j >= 1), the alignment the token arm uses.
            m = batch["input_ids"] > LATENT_TOKEN_BASE
            latent_tokens = torch.stack([latent_tokens[i][m[i]] for i in range(len(latent_tokens))])
            assert latent_tokens.shape[1] == self.nq, latent_tokens.shape
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
    # PHASE-2 (latent-action policy pretraining, 2026-09-20).  UniVLA's second stage trains the policy to predict
    # LATENT actions on the cross-embodiment mixture -- no action head, because those sources have no common action
    # space -- and only then post-trains on the target robot.  Our controlled comparison skipped it, which is why a
    # latent that is a deterministic function of (o_t, l) could not pay for itself downstream (ledger 2026-09-20).
    lam_a: float = 1.0                                     # weight of the action loss; 0 = latent-only pretraining
    init_from_run: Optional[str] = None                    # start from this run dir's adapter + heads (post-training a phase-2 arm)
    resume: bool = False                                   # continue this exp_id from its last save (Slurm walltime chunks)
    merge: bool = True                                     # merge LoRA into a standalone checkpoint at the end
    curriculum_frac: float = 0.6                           # oracle -> predicted anneal, as in downstream/libero_policy.py
    head: str = "reg"                                      # continuous arms: reg (ContHeads) | reg_text (+ direct text) | flow (FlowHeads)
    flow_width: int = 512; flow_layers: int = 4; flow_z_tok: int = 32; flow_steps: int = 10   # flow head; steps used at evaluation
    query_tokens: int = 0                                  # continuous arms: 0 = pool the prompt; n = read n placeholder tokens after it
    vla_path: str = os.path.join(LF, "cache", "vla", "openvla-7b")
    lam_path: str = os.path.join(LF, "claude_survey", "ckpt", "s2mix_1ep.pt")      # univla arm: their Stage-2 recipe on our data
    acdc_tag: str = "s1_acdc_1ep"; acdc_codebook: str = "vq_s1_acdc_1ep_K64.npz"    # acdc arm
    tc_tag: str = "tc_cont"                                # tc_cont / tc_vq arms: the structured tokenizer checkpoint
    data_root_dir: Path = Path(os.path.join(LF, "data", "libero_rlds")); dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path(os.path.join(LF, "acdc_la", "ckpt", "vla_runs")); run_id_note: Optional[str] = None
    batch_size: int = 8; grad_accumulation_steps: int = 2; max_steps: int = 15000; save_steps: int = 5000; learning_rate: float = 3.5e-4
    image_aug: bool = True; shuffle_buffer_size: int = 16000; window_size: int = 12
    lora_rank: int = 32; lora_dropout: float = 0.0
    # FULL-PARAMETER training (2026-10-15, phase 2 at UniVLA's scale: their stage 2 trains the whole 7B with FSDP).
    # No LoRA: every VLA weight trains, fp32 master weights sharded by FSDP over the ranks, bf16 compute, activation
    # checkpointing on the LLaMA layers -- the settings of UniVLA's prismatic/training/strategies/fsdp.py.  Saves the
    # full VLA state dict (bf16) as vla_full.pt next to the heads; --init_from_run reads it back.
    full_ft: bool = False
    univla_optim: bool = False                             # UniVLA's stage-2 optimiser (prismatic/conf/vla.py): weight decay 0, CONSTANT lr (no 0.1 step at 80 %)
    # LR-DECAY BRANCH (2026-10-29, WSD-style): UniVLA's stage 2 keeps the lr constant; a branch run starts from another
    # full_ft run's sharded checkpoint (hard links into its own run dir, then the usual --resume) and decays the lr
    # linearly from learning_rate at step decay_start to 0 at max_steps.  The source run is left untouched.
    branch_from: Optional[str] = None                      # source run dir (full_ft only); its checkpoint must be at step decay_start
    decay_start: int = -1                                  # >= 0: linear lr decay to 0 over [decay_start, max_steps)
    warmup_steps: int = 0                                  # > 0: linear lr warmup from 0 over the first warmup_steps updates (2026-10-29)
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


def batches(dl, every):
    """Batches of `dl`, starting a fresh iterator every `every` batches (0 = one iterator, the original behaviour).
    The RLDS tf.data pipeline grows ~250 MB per 1000 samples inside its iterator (~27 GB per rank by 3.6k steps, which
    killed the 2-GPU run on the 62 GB workstation 2026-09-14); dropping the iterator returns it (measured 2026-09-19:
    17 GB -> 5 GB after 40k samples), so small-RAM hosts set CTRL_REITER_BATCHES."""
    while True:
        it = iter(dl)
        for i, b in enumerate(it):
            yield b
            if every and i + 1 >= every: break
        del it; gc.collect()


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
    exp_id = f"{cfg.latent}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}x{state.num_processes}+lr{cfg.learning_rate}+" + ("fft" if cfg.full_ft else f"r{cfg.lora_rank}") + f"+s{cfg.max_steps}" + {"flow": "+fm", "reg_text": "+rt"}.get(cfg.head, "") + (f"+q{cfg.query_tokens}" if cfg.query_tokens else "") + (f"--{cfg.run_id_note}" if cfg.run_id_note else "")
    run_dir = cfg.run_root_dir / exp_id; adapter_dir = cfg.run_root_dir / "adapters" / exp_id; os.makedirs(run_dir, exist_ok=True); os.makedirs(adapter_dir, exist_ok=True)
    if state.is_main_process: print(f"[ctrl] {exp_id} -> {run_dir}", flush=True)

    AutoConfig.register("openvla", OpenVLAConfig); AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor); AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    time.sleep(int(os.environ.get("CTRL_STAGGER_SEC", "0")) * state.local_process_index)   # small-RAM hosts: stagger the 7B loads across ranks
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.float32 if cfg.full_ft else torch.bfloat16,
                                                low_cpu_mem_usage=True, trust_remote_code=True)

    labeller, n_tokens, codebook_size = build_labeller(cfg, device_id)
    cont = cfg.latent in ("acdc_cont", "none", "tc_cont", "univla_cont"); dz = getattr(labeller, "dz", 0) if cfg.latent in ("acdc_cont", "tc_cont", "univla_cont") else 0
    nq = cfg.query_tokens if cont else 0
    if not cont or nq:
        # the latent vocabulary: <ACT_0..K-1> appended after <PAD> (32000) -> ids 32001..; embeddings + lm_head rows are trained.
        # Continuous arms with query tokens reuse the same ids as nq PLACEHOLDERS: never predicted, their embeddings trained.
        assert len(processor.tokenizer) == LATENT_TOKEN_BASE + 1, len(processor.tokenizer)
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [f"<ACT_{i}>" for i in range(nq or codebook_size)]})
        vla.resize_token_embeddings(len(processor.tokenizer))
    # Post-training a FULL-parameter phase-2 run (2026-10-27): its whole VLA (vla_full.pt) is the BASE and a LoRA goes on
    # top, as UniVLA post-trains its fully pretrained model.  Loaded from --init_from_run on every start, so a --resume
    # (adapter from this run) sits on the same base, and the final merge uses it too.
    full_init = None
    if not cfg.full_ft and cfg.init_from_run and os.path.exists(os.path.join(cfg.init_from_run, "vla_full.pt")):
        full_init = os.path.join(cfg.init_from_run, "vla_full.pt")
        vla.load_state_dict(torch.load(full_init, map_location="cpu", mmap=True))
        if state.is_main_process: print(f"[ctrl] base VLA from {full_init}", flush=True)
    if not cfg.full_ft: vla = vla.to(device_id)
    lora = LoraConfig(r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16), lora_dropout=cfg.lora_dropout, target_modules="all-linear",
                      init_lora_weights="gaussian", modules_to_save=(["embed_tokens"] if nq else None) if cont else ["embed_tokens", "lm_head"])
    # Where this run's weights start: its own last save (--resume, for Slurm walltime chunks), else another run's
    # (--init-from-run <run dir>: how a phase-2 pretrained arm is post-trained).  A run dir holds the heads
    # (action_decoder.pt); its adapter sits next to it under adapters/<exp_id>.
    src_run = str(run_dir) if (cfg.resume and os.path.exists(run_dir / "action_decoder.pt")) else cfg.init_from_run
    src_ad = (str(adapter_dir) if src_run == str(run_dir) else
              os.path.join(cfg.run_root_dir, "adapters", os.path.basename(str(src_run).rstrip("/")))) if src_run else None
    if cfg.full_ft:
        if src_run == str(run_dir) and os.path.isdir(run_dir / "fsdp_ckpt"): pass     # --resume: the sharded checkpoint below restores everything
        elif src_run and os.path.exists(os.path.join(src_run, "vla_full.pt")):
            vla.load_state_dict(torch.load(os.path.join(src_run, "vla_full.pt"), map_location="cpu"))
            if state.is_main_process: print(f"[ctrl] full VLA weights from {src_run}/vla_full.pt", flush=True)
        else: assert not src_run or src_run == str(run_dir), f"--full_ft: no vla_full.pt in {src_run}"
        vla.requires_grad_(True)
        # no KV cache in training: with activation checkpointing the recomputed layer would see its own cached keys
        # twice (attention mask 289 vs keys 578, smoke 927105); UniVLA's FSDP training runs with use_cache=False too
        vla.config.use_cache = False; vla.language_model.config.use_cache = False
    elif src_ad and os.path.exists(os.path.join(src_ad, "adapter_config.json")):
        vla = PeftModel.from_pretrained(vla, src_ad, is_trainable=True)
        if state.is_main_process: print(f"[ctrl] LoRA from {src_ad}", flush=True)
    elif full_init and src_run == cfg.init_from_run:
        vla = get_peft_model(vla, lora)                              # fresh LoRA on the full phase-2 base loaded above
        if state.is_main_process: print(f"[ctrl] fresh LoRA on the full phase-2 base", flush=True)
    else:
        assert not src_run, f"no adapter at {src_ad}"
        vla = get_peft_model(vla, lora)
    if not cfg.full_ft: vla.print_trainable_parameters()
    elif state.is_main_process: print(f"[ctrl] FULL fine-tuning: {sum(p.numel() for p in vla.parameters()) / 1e9:.2f}B trainable VLA parameters", flush=True)
    assert cfg.head in ("reg", "reg_text", "flow") and (cfg.head == "reg" or cont), (cfg.head, cfg.latent)
    model = WrappedModel(vla, cfg.window_size, n_tokens, continuous=cont, dz=dz, head=cfg.head,
                         flow_kw=dict(width=cfg.flow_width, layers=cfg.flow_layers, z_tok=cfg.flow_z_tok), query_tokens=nq)
    model = model if cfg.full_ft else model.to(device_id)
    if src_run:                                                     # the heads travel with the adapter
        hp = os.path.join(src_run, "action_decoder.pt")
        (model.heads if cont else model.action_decoder).load_state_dict(torch.load(hp, map_location="cpu"), strict=False)
        if state.is_main_process: print(f"[ctrl] heads from {hp}", flush=True)
    if cfg.full_ft:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from timm.models.vision_transformer import Block as ViTBlock
        from prismatic.extern.hf.modeling_prismatic import PrismaticProjector
        layers = {LlamaDecoderLayer, ViTBlock, PrismaticProjector}
        # the VLA in bf16 compute over fp32 shards; the small heads stay a plain fp32 root unit (they take fp32 inputs)
        model.vla = FSDP(model.vla, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=layers),
                         mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32),
                         sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device_id, limit_all_gathers=True, use_orig_params=True)
        apply_activation_checkpointing(model.vla, checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                                       check_fn=lambda m: isinstance(m, LlamaDecoderLayer))
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device_id, use_orig_params=True)
        dist.barrier()
        if state.is_main_process: print(f"[ctrl] FSDP ready, {torch.cuda.memory_allocated(device_id) / 2**30:.1f} GiB on GPU 0", flush=True)
    else:
        model = DDP(model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate, **(dict(weight_decay=0.0) if cfg.univla_optim else {}))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(cfg.max_steps * 0.8), gamma=1.0 if cfg.univla_optim else 0.1)
    step0, tstate = 0, run_dir / "trainer_state.pt"                 # --resume: optimiser, schedule and step count
    seed = [bool(cfg.branch_from) and not tstate.exists()]       # rank 0 decides: the others must not race its copy
    if cfg.branch_from: dist.broadcast_object_list(seed, src=0)
    if seed[0]:
        assert cfg.full_ft and cfg.resume and cfg.decay_start >= 0, "--branch_from needs --full_ft, --resume and --decay_start"
        src = Path(cfg.branch_from)
        if state.is_main_process:
            src_step = int(torch.load(src / "trainer_state.pt", map_location="cpu")["step"])
            assert src_step == cfg.decay_start, f"--branch_from {src} is at step {src_step}, not --decay_start {cfg.decay_start}"
            import shutil
            if (run_dir / "fsdp_ckpt").exists(): shutil.rmtree(run_dir / "fsdp_ckpt")
            subprocess.run(["cp", "-al", str(src / "fsdp_ckpt"), str(run_dir / "fsdp_ckpt")], check=True)   # hard links: no copy, source untouched
            shutil.copy2(src / "trainer_state.pt", str(tstate) + ".tmp"); os.replace(str(tstate) + ".tmp", tstate)
            print(f"[ctrl] branch from {src} at step {src_step}", flush=True)
        dist.barrier()
    if cfg.resume and not tstate.exists() and state.is_main_process:
        # exp_id carries max_steps: a chunked run must pass the SAME --max_steps every time, or it resumes nothing.
        print(f"[ctrl] --resume: no {tstate}, starting from scratch", flush=True)
    if cfg.resume and tstate.exists():
        st = torch.load(tstate, map_location="cpu")
        step0 = int(st["step"])
        if cfg.full_ft:
            # per-rank shards of the fp32 weights (+ heads) and AdamW state; needs the SAME number of ranks as the save
            ck = str(run_dir / "fsdp_ckpt")
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                msd = {"model": model.state_dict()}
                dcp.load_state_dict(msd, storage_reader=dcp.FileSystemReader(ck)); model.load_state_dict(msd["model"])
                osd = load_sharded_optimizer_state_dict(model_state_dict=msd["model"], optimizer_key="optim", storage_reader=dcp.FileSystemReader(ck))
                opt.load_state_dict(FSDP.optim_state_dict_to_load(model, opt, osd["optim"]))
            sched.load_state_dict(st["sched"]); del msd, osd
        elif "opt" in st: opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        else:
            # Optimiser state lost (a rebuilt trainer_state.pt holding only the step): the adapter and heads are the
            # checkpoint's, the step count and the StepLR position are restored, AdamW's moments restart from zero.
            lr = cfg.learning_rate * sched.gamma ** (step0 // sched.step_size)
            for g in opt.param_groups: g["lr"] = lr
            sched.last_epoch, sched._step_count, sched._last_lr = step0, step0 + 1, [lr] * len(opt.param_groups)
            if state.is_main_process: print(f"[ctrl] WARNING optimiser state lost ({st.get('opt_lost')}): AdamW moments reset at step {step0}, lr={lr}", flush=True)
        if state.is_main_process: print(f"[ctrl] resumed at step {step0}", flush=True)

    transform = RLDSBatchTransformCtrl(labeller, cfg.latent, processor.tokenizer, image_transform=processor.image_processor.apply_transform,
                                       image_transform_lam=transforms.ToTensor(), prompt_builder_fn=PurePromptBuilder, window_size=cfg.window_size,
                                       query_tokens=nq)
    ds = RLDSDataset(cfg.data_root_dir, cfg.dataset_name, transform, resize_resolution=tuple(model.module.vla.config.image_sizes), shuffle_buffer_size=cfg.shuffle_buffer_size,
                     image_aug=cfg.image_aug, window_size=cfg.window_size + 2, training_phase="post-training")
    if state.is_main_process: save_dataset_statistics(ds.dataset_statistics, run_dir); processor.save_pretrained(run_dir)
    if state.is_main_process and os.environ.get("CTRL_ANCHOR_STATS"):
        # chunk-anchored runs (datasets_ctrl.anchor_chunk): the head's 7-D output is un-normalised with the per-position
        # anchor stats, not with the RLDS stats of the RAW 13-D field -- overwrite the action entry so deployment can't mix them up
        _anc = json.load(open(os.environ["CTRL_ANCHOR_STATS"])); _p = run_dir / "dataset_statistics.json"; _ds = json.load(open(_p))
        for _k in _ds:
            assert _k in _anc, f"{_k} has no anchor stats"
            _ds[_k]["action"] = _anc[_k]
        json.dump(_ds, open(_p, "w"), indent=2)
    collator = PaddedCollatorForActionPrediction_LIBERO(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=None, collate_fn=collator, num_workers=0)
    n_patches = model.module.vla.vision_backbone.featurizer.patch_embed.num_patches
    log = open(run_dir / "train_log.jsonl", "a") if state.is_main_process else None; t0 = time.time(); acc_buf, l1_buf, loss_buf = [], [], []

    model.train(); opt.zero_grad()
    for bi, batch in enumerate(batches(dl, int(os.environ.get("CTRL_REITER_BATCHES", "0")))):
        for k in ("input_ids", "attention_mask", "labels", "actions", "latent_action_idx"): batch[k] = batch[k].to(device_id)
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
        p_or = max(0.0, 1.0 - (step0 + bi // cfg.grad_accumulation_steps) / (cfg.curriculum_frac * cfg.max_steps)) if model.module.continuous else 0.0
        out, act_loss, loss_1, lat_loss, lat_r2 = model(batch, p_oracle=p_or)
        loss = cfg.lam_a * act_loss + (cfg.lam_z * lat_loss if model.module.continuous else out.loss)   # lam_a = 0: phase-2 latent-only
        (loss / cfg.grad_accumulation_steps).backward()
        with torch.no_grad():
            if model.module.continuous: acc_buf.append(float(lat_r2)); loss_buf.append(float(lat_loss))
            else:
                preds = out.logits[:, n_patches:-1].argmax(2); gt = batch["labels"][:, 1:]; m = gt > LATENT_TOKEN_BASE
                acc_buf.append(float(((preds == gt) & m).sum() / m.sum())); loss_buf.append(float(out.loss))
            l1_buf.append(float(loss_1 if cfg.head == "flow" else act_loss))   # flow: one-step action L1, not the flow loss
        step = step0 + bi // cfg.grad_accumulation_steps
        if (bi + 1) % cfg.grad_accumulation_steps == 0:
            if cfg.decay_start >= 0 and step >= cfg.decay_start:     # set before every update; StepLR (gamma 1) keeps it
                for g in opt.param_groups: g["lr"] = cfg.learning_rate * (cfg.max_steps - step) / (cfg.max_steps - cfg.decay_start)
            if 0 < cfg.warmup_steps and step <= cfg.warmup_steps:   # StepLR's chainable form keeps the group lr between its steps
                for g in opt.param_groups: g["lr"] = cfg.learning_rate * min(1.0, (step + 1) / cfg.warmup_steps)
            (model.clip_grad_norm_(1.0) if cfg.full_ft else torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); opt.zero_grad(); sched.step()
            if state.is_main_process and (step + 1) % cfg.log_every == 0:
                key = "latent_r2" if model.module.continuous else "token_acc"
                rec = {"step": step + 1, key: sum(acc_buf) / len(acc_buf), "latent_loss" if model.module.continuous else "token_loss": sum(loss_buf) / len(loss_buf),
                       ("action_l1_1step" if cfg.head == "flow" else "action_l1"): sum(l1_buf) / len(l1_buf), "p_oracle": p_or, "lr": opt.param_groups[0]["lr"], "seconds": int(time.time() - t0)}
                print(f"[ctrl:{cfg.latent}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
                acc_buf, l1_buf, loss_buf = [], [], []
            if (step + 1) % cfg.save_steps == 0 or step + 1 == cfg.max_steps:
                if cfg.full_ft:
                    # (1) every save: sharded fp32 weights + AdamW state, written to a temp dir and swapped in only when complete
                    ck, tmp = run_dir / "fsdp_ckpt", run_dir / "fsdp_ckpt.tmp"
                    if state.is_main_process and tmp.exists(): import shutil; shutil.rmtree(tmp)
                    dist.barrier()
                    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                        dcp.save_state_dict({"model": model.state_dict(), "optim": FSDP.optim_state_dict(model, opt)}, storage_writer=dcp.FileSystemWriter(str(tmp)))
                    dist.barrier()
                    if state.is_main_process:
                        import shutil
                        if ck.exists(): shutil.rmtree(ck)
                        os.replace(tmp, ck)
                        torch.save(dict(sched=sched.state_dict(), step=step + 1), str(tstate) + ".tmp"); os.replace(str(tstate) + ".tmp", tstate)
                    dist.barrier()
                if cfg.full_ft and step + 1 == cfg.max_steps:    # (2) at the end: full bf16 VLA for post-training / evaluation
                    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                        full_sd = model.state_dict()
                    if state.is_main_process:
                        vsd = {k[len("vla."):]: v.to(torch.bfloat16) for k, v in full_sd.items() if k.startswith("vla.")}
                        hsd = {k[len("heads."):]: v for k, v in full_sd.items() if k.startswith("heads.")}
                        torch.save(vsd, run_dir / "vla_full.pt.tmp"); os.replace(run_dir / "vla_full.pt.tmp", run_dir / "vla_full.pt")
                        torch.save(hsd, run_dir / "action_decoder.pt")
                    del full_sd
                if state.is_main_process:
                    if not cfg.full_ft:
                        model.module.vla.save_pretrained(adapter_dir)
                        torch.save((model.module.heads if cont else model.module.action_decoder).state_dict(), run_dir / "action_decoder.pt")
                    json.dump(dict(latent=cfg.latent, n_tokens=n_tokens, codebook_size=codebook_size, dz=dz, window_size=cfg.window_size, step=step + 1, vla_path=cfg.vla_path,
                                   lam_path=cfg.lam_path, acdc_tag=cfg.acdc_tag, acdc_codebook=cfg.acdc_codebook, tc_tag=cfg.tc_tag, dataset_name=cfg.dataset_name,
                                   train_frac=os.environ.get("LIBERO_TRAIN_FRAC"), override=ov_d, head=cfg.head, query_tokens=nq,
                                   flow=dict(width=cfg.flow_width, layers=cfg.flow_layers, z_tok=cfg.flow_z_tok, steps=cfg.flow_steps),
                                   **({"action_space": "delta_anchored"} if os.environ.get("CTRL_ANCHOR_STATS") else {})), open(run_dir / "ctrl_meta.json", "w"), indent=1)
                    # written to a temp file and renamed: a job killed mid-write (1.2 GB) otherwise leaves a truncated
                    # trainer_state.pt that --resume cannot load (arm B of phase 2, Newton, 2026-10-13, step 19000)
                    if not cfg.full_ft:   # full_ft saves its own (sharded) trainer state above
                        torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1), str(tstate) + ".tmp"); os.replace(str(tstate) + ".tmp", tstate)
                    print(f"[ctrl] saved {'full VLA' if cfg.full_ft else 'adapter'} + decoder at step {step + 1}", flush=True)
                dist.barrier()
            if step + 1 >= cfg.max_steps: break
    del batch; gc.collect()                                         # release the data iterator's memory before the merge reloads the 7B base
    if cfg.merge and not cfg.full_ft and state.is_main_process:                         # merge LoRA into a standalone checkpoint for the evaluator
        base = AutoModelForVision2Seq.from_pretrained(cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        if not cont or nq: base.resize_token_embeddings(len(processor.tokenizer))
        if full_init: base.load_state_dict(torch.load(full_init, map_location="cpu", mmap=True))     # the phase-2 base the LoRA was trained on
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload(); merged.save_pretrained(run_dir)
        print(f"[ctrl-done] merged checkpoint at {run_dir} ({int(time.time() - t0)}s)", flush=True)
    dist.barrier()


if __name__ == "__main__":
    finetune()
