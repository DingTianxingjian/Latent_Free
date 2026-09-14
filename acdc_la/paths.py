"""Filesystem constants for the ACDC-LA repo.  Everything else imports from here.

Layout (all under /anvil/scratch/x-tding2):
    latent_free/cache/{bridge_big, oxe/<source>}   frozen OXE caches: frames.npy + index.npz (+ lang_t5.npz, meta.json)
    latent_free/cache/torch/hub                     pre-fetched DINOv2 weights (compute nodes have no network)
    latent_free/data/distractor_video.npy           natural video used by the moving-patch nuisance
    latent_free/acdc_la/results, ckpt, logs         this repo's outputs (results are json + npz; ckpt is gitignored)
    latent_free/claude_survey/ckpt/swaplam_*.pt     legacy checkpoints (rec_src = the Stage-I tokenizer the
                                                    Stage-III result was measured on); loadable via common.model.load_legacy
"""
import os

ACDC = os.path.dirname(os.path.abspath(__file__))
LF = os.path.dirname(ACDC)                                   # latent_free/
SCRATCH = os.path.dirname(LF)                                # /anvil/scratch/x-tding2
CACHE = os.path.join(LF, "cache")
TORCH_HUB = os.path.join(CACHE, "torch", "hub")
DISTRACTOR_VIDEO = os.path.join(LF, "data", "distractor_video.npy")
RESULTS = os.path.join(ACDC, "results")
CKPT = os.path.join(ACDC, "ckpt")
LOGS = os.path.join(ACDC, "logs")
CLAUDE_SURVEY = os.path.join(LF, "claude_survey")            # legacy experiment ground (moved under latent_free 2026-09-09)
LEGACY_CKPT = os.path.join(CLAUDE_SURVEY, "ckpt")
LEGACY_RESULTS = os.path.join(CLAUDE_SURVEY, "results")
PYTHON = os.path.join(SCRATCH, "envs", "latent_free", "bin", "python")
UNIVLA_ROOT = os.path.join(LF, "third_party", "UniVLA")
UNIVLA_CKPT = {k: os.path.join(CACHE, "univla", f"lam-stage-{k}.ckpt") for k in (1, 2)}

IMG = 224            # DINOv2 ViT-B/14 -> 16x16 patches
GRID = 4             # patch grid pooled to 4x4 ("g4")
DINO_DIM = 768
T5_DIM = 768

for _d in (RESULTS, CKPT, LOGS):
    os.makedirs(_d, exist_ok=True)
