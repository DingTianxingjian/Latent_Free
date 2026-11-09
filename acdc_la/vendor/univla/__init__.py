"""Verbatim copies of the few UniVLA files the deployment path needs, so a robot host does not have to clone UniVLA.

Source: https://github.com/OpenDriveLab/UniVLA at commit 0ab9e9dd3074ffa14fdd1a00aa2aa84e4f8e1352, Apache-2.0
(LICENSE in this directory).  None of our patches in acdc_la/patches/ touch these files, so they are byte-identical to
what training imported from third_party/UniVLA:
    configuration_prismatic.py, modeling_prismatic.py, processing_prismatic.py   <- prismatic/extern/hf/
    transformer_utils.py  (MAPBlock)                                            <- prismatic/models/policy/
    base_prompter.py      (PurePromptBuilder)                                   <- prismatic/models/backbones/llm/prompting/
    crop.py               (crop_and_resize only)                                <- experiments/robot/openvla_utils.py
Do not edit them; re-copy from that commit if UniVLA is ever bumped.
"""
