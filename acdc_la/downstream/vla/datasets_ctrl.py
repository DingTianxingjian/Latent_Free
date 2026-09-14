"""One RLDS batch transform for both arms of the controlled comparison.  Identical to UniVLA's
`RLDSBatchTransformLIBERO_withHis` (prompt, history trick with p = 0.5, label masking, action chunk) except for the
labeller: `latent="univla"` encodes (frame ro, frame ro+window-1) with their LAM -> 4 tokens; `latent="acdc"` encodes
the T = 5 frames ro, ro+3, ..., ro+12 with ours -> 8 tokens.  The RLDS window is loaded with window_size + 2 frames for
BOTH arms so that the sampling of windows is identical."""
import random
from dataclasses import dataclass
from typing import Any, Dict, Type

import torch
from PIL import Image

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100


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

    def tokens(self, imgs, ro):
        if self.latent == "univla":
            a = self.image_transform_lam(Image.fromarray(imgs[ro])); b = self.image_transform_lam(Image.fromarray(imgs[self.window_size - 1 + ro]))
            return self.labeller.vq_encode(torch.stack([a, b], 0).unsqueeze(0).to(self.labeller.device))["indices"].reshape(-1)
        stride = self.window_size // (self.labeller.T - 1)
        return self.labeller.encode([imgs[ro + k * stride] for k in range(self.labeller.T)])

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = rlds_batch["dataset_name"]; lang = rlds_batch["task"]["language_instruction"].decode().lower()
        imgs = rlds_batch["observation"]["image_primary"]
        ro = random.randint(0, 1)
        pixel_values = self.image_transform(Image.fromarray(imgs[ro]))
        with torch.no_grad():
            idx = self.tokens(imgs, ro); hist = self.tokens(imgs, 0) if ro > 0 else None
        action_tokens = "".join(f"<ACT_{int(i)}>" for i in idx)
        input_prompt = f"What action should the robot take to {lang}?"
        if hist is not None: input_prompt += " History action " + "".join(f"<ACT_{int(i)}>" for i in hist)
        pb = self.prompt_builder_fn("openvla")
        for turn in ({"from": "human", "value": input_prompt}, {"from": "gpt", "value": action_tokens}): pb.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids); input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(len(idx) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token: labels[-1] = IGNORE_INDEX
        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels, actions=rlds_batch["action"][ro: self.window_size + ro],
                    latent_action_idx=idx.cpu(), dataset_name=dataset_name)
