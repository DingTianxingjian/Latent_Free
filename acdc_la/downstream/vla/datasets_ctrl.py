"""One RLDS batch transform for both arms of the controlled comparison.  Identical to UniVLA's
`RLDSBatchTransformLIBERO_withHis` (prompt, history trick with p = 0.5, label masking, action chunk) except for the
labeller: `latent="univla"` encodes (frame ro, frame ro+window-1) with their LAM -> 4 tokens; `latent="acdc"` encodes
the T = 5 frames ro, ro+3, ..., ro+12 with ours -> 8 tokens.  The RLDS window is loaded with window_size + 2 frames for
BOTH arms so that the sampling of windows is identical."""
import os, random
from dataclasses import dataclass
from typing import Any, Dict, Type

import torch
from PIL import Image

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


class KeptFrames:
    """The window's frames when the RLDS pipeline kept only CTRL_OBS_KEEP positions (rlds/traj_transforms.py): indexed by
    the ORIGINAL window position, so every labeller reads exactly the frame it read from the full window.  Asking for a
    position that was not kept is an error, never a silent substitute."""
    def __init__(self, imgs, keep):
        assert len(imgs) == len(keep), (len(imgs), keep)
        self.imgs, self.pos = imgs, {k: i for i, k in enumerate(keep)}

    def __getitem__(self, k):
        if k not in self.pos: raise KeyError(f"window frame {k} not in CTRL_OBS_KEEP={sorted(self.pos)}")
        return self.imgs[self.pos[k]]


_KEEP = [int(k) for k in os.environ["CTRL_OBS_KEEP"].split(",")] if os.environ.get("CTRL_OBS_KEEP") else None

# CHUNK-ANCHORED delta actions (2026-11-11, acdc_la/realrobot/make_anchor_rlds.py).  A *_anc dataset stores RAW 13-D
# actions [absolute target (6), gripper (1), measured joints (6)].  The chunk the head learns is
#     action[s+k, :6] - joints[s]          (every entry relative to the joints when the chunk STARTS)
# normalised to [-1, 1] with that dataset's per-position q01/q99 and clipped, as BOUNDS_Q99 would; gripper raw.
_ANCHOR = None
if os.environ.get("CTRL_ANCHOR_STATS"):
    import json, numpy as np
    _ANCHOR = {k: (np.array(v["q01_k"], np.float32), np.array(v["q99_k"], np.float32))
               for k, v in json.load(open(os.environ["CTRL_ANCHOR_STATS"])).items()}


def anchor_chunk(acts, dataset_name):
    """(W, 13) raw window slice -> (W, 7) normalised chunk-anchored actions."""
    import numpy as np
    name = dataset_name.decode() if isinstance(dataset_name, bytes) else str(dataset_name)
    lo, hi = _ANCHOR[name]
    acts = np.asarray(acts, np.float32); assert acts.shape[1] == 13, acts.shape
    rel = acts[:, :6] - acts[0, 7:13]
    n = np.clip(2 * (rel - lo[:, :6]) / np.maximum(hi[:, :6] - lo[:, :6], 1e-8) - 1, -1, 1)
    return np.concatenate([n, acts[:, 6:7]], 1).astype(np.float32)


@dataclass
class RLDSBatchTransformCtrl:
    labeller: Any
    latent: str
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    image_transform_lam: Any
    prompt_builder_fn: Type[PromptBuilder]
    window_size: int = 12
    predict_stop_token: bool = True
    query_tokens: int = 0                                     # continuous arms: placeholder tokens after the instruction

    def tokens(self, imgs, ro):
        if self.latent == "none": return torch.zeros(1)
        if self.latent in ("tc_cont", "tc_vq"):
            stride = self.window_size // (self.labeller.T - 1)
            fr = [imgs[ro + k * stride] for k in range(self.labeller.T)]
            return self.labeller.encode_continuous(fr) if self.latent == "tc_cont" else self.labeller.encode(fr)
        if self.latent == "acdc_cont":
            stride = self.window_size // (self.labeller.T - 1)
            return self.labeller.encode_continuous([imgs[ro + k * stride] for k in range(self.labeller.T)])
        if self.latent in ("univla", "univla_cont"):
            a = self.image_transform_lam(Image.fromarray(imgs[ro])); b = self.image_transform_lam(Image.fromarray(imgs[self.window_size - 1 + ro]))
            out = self.labeller.vq_encode(torch.stack([a, b], 0).unsqueeze(0).to(self.labeller.device))
            if self.latent == "univla": return out["indices"].reshape(-1)
            return (out["z_q"].reshape(-1) - self.labeller.zmean) / self.labeller.zstd      # their deployed code, continuous
        stride = self.window_size // (self.labeller.T - 1)
        return self.labeller.encode([imgs[ro + k * stride] for k in range(self.labeller.T)])

    def _acts(self, rlds_batch, ro):
        a = rlds_batch["action"][ro: self.window_size + ro]
        if a.shape[-1] == 13:
            assert _ANCHOR is not None, "a *_anc dataset needs CTRL_ANCHOR_STATS=<data root>/anchor_stats.json"
            return anchor_chunk(a, rlds_batch["dataset_name"])
        return a

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]; lang = rlds_batch["task"]["language_instruction"].decode().lower()
        imgs = rlds_batch["observation"]["image_primary"]
        if _KEEP is not None: imgs = KeptFrames(imgs, _KEEP)
        ro = random.randint(0, 1)
        pixel_values = self.image_transform(Image.fromarray(imgs[ro]))
        with torch.no_grad():
            idx = self.tokens(imgs, ro); hist = self.tokens(imgs, 0) if ro > 0 else None
        if self.latent in ("acdc_cont", "none", "tc_cont", "univla_cont"):
            # no latent vocabulary: the prompt is the bare instruction and the whole label is ignored; the VLM's hidden
            # state feeds a regression head instead of the LM head (the history trick has no discrete token to carry)
            pb = self.prompt_builder_fn("openvla")
            ph = "".join(f"<ACT_{i}>" for i in range(self.query_tokens))            # query-token interface (empty if 0)
            for turn in ({"from": "human", "value": f"What action should the robot take to {lang}?"}, {"from": "gpt", "value": ph}):
                pb.add_turn(turn["from"], turn["value"])
            input_ids = torch.tensor(self.base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids)
            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=torch.full_like(input_ids, IGNORE_INDEX),
                        actions=self._acts(rlds_batch, ro), latent_action_idx=idx.float().cpu(), dataset_name=dataset_name)
        action_tokens = "".join(f"<ACT_{int(i)}>" for i in idx)
        input_prompt = f"What action should the robot take to {lang}?"
        if hist is not None: input_prompt += " History action " + "".join(f"<ACT_{int(i)}>" for i in hist)
        pb = self.prompt_builder_fn("openvla")
        for turn in ({"from": "human", "value": input_prompt}, {"from": "gpt", "value": action_tokens}): pb.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids); input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(len(idx) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token: labels[-1] = IGNORE_INDEX
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, actions=self._acts(rlds_batch, ro),
                    latent_action_idx=idx.cpu(), dataset_name=dataset_name)
