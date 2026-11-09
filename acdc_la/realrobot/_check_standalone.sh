#!/bin/bash
# Fresh-host check of the deployment path.  The training env puts third_party/UniVLA on sys.path through a .pth file,
# so UniVLA is BLOCKED at import time here (meta-path hook): any import of prismatic/experiments raises, proving the
# policy needs only this repo + the HF checkpoint.
cd /lustre/fs1/home/ti874297/lf/latent_free
PY=/lustre/fs1/home/ti874297/lf/envs/univla_vla/bin/python
export TF_CPP_MIN_LOG_LEVEL=2 HF_HUB_OFFLINE=1
nvidia-smi -L || { echo "NO GPU on $(hostname)"; exit 3; }
CK=/lustre/fs1/groups/ytian/ti874297/latent_free/hf_terra/piper_pick_can_10k
$PY - <<PYEOF
import sys, runpy, importlib.abc
sys.path = [p for p in sys.path if "third_party" not in p]; sys.path.insert(0, ".")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("prismatic", "experiments"): raise ImportError(f"BLOCKED UniVLA import: {name}")
sys.meta_path.insert(0, Block())
try:
    import prismatic; print("WARNING prismatic importable")
except ImportError as e: print("ok:", e)
import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
sys.argv = ["piper_policy", "--checkpoint", "$CK", "--replay", "data/real_rlds", "--out", "acdc_la/results/piper_replay_standalone.json"]
runpy.run_module("acdc_la.realrobot.piper_policy", run_name="__main__")
print("ok: standalone replay finished; loaded modules from UniVLA:", [m for m in sys.modules if m.split(".")[0] in ("prismatic", "experiments")])
PYEOF
# training side still imports after the heads move
PYTHONPATH=$PWD:$PWD/third_party/UniVLA $PY -c "import acdc_la.downstream.vla.finetune_libero_ctrl as m, acdc_la.downstream.vla.heads_ctrl as h; assert m.FlowHeads is h.FlowHeads; print('ok: training module imports, heads re-exported')"
