"""Reconstruct exp_vjepa CSVs from the (timed-out) job log -- all printed rows
survive a TIMEOUT even though main() never reached the to_csv() call."""
import glob
import os
import re
import sys

import pandas as pd

import data

log = sys.argv[1] if len(sys.argv) > 1 else sorted(
    glob.glob("logs/lf_vj_*.out"), key=os.path.getmtime)[-1]
txt = open(log).read()

pa = re.compile(
    r"\[vj:A\]\s+(\S+)\s+(\S+)\s+s=(\d+)\s+(\S+)\s+d=\s*(\d+)\s+"
    r"xyz=([-\d.]+)\s+rot=([-\d.]+)\s+cont6=([-\d.]+)")
A = [dict(space=m[0], condition=m[1], stride=int(m[2]), feature=m[3],
          dim=int(m[4]), r2_xyz=float(m[5]), r2_rot=float(m[6]),
          r2_cont6=float(m[7])) for m in pa.findall(txt)]

pb = re.compile(
    r"\[vj:B\]\s+(\S+)\s+s=(\d+)\s+k=\s*(\d+)\s+top=([-\d.]+)\s+"
    r"rand=([-\d.]+)\s+bottom=([-\d.]+)\s+top/rand=([-\d.]+)x")
B = [dict(space="g4", condition=m[0], stride=int(m[1]), k=int(m[2]),
          r2_xyz_top=float(m[3]), r2_xyz_rand_mean=float(m[4]),
          r2_xyz_bottom=float(m[5]), top_over_rand=float(m[6]))
     for m in pb.findall(txt)]

dfa = pd.DataFrame(A)
dfb = pd.DataFrame(B)
dfa.to_csv(os.path.join(data.RESULTS_DIR, "exp_vjepa_probes.csv"), index=False)
dfb.to_csv(os.path.join(data.RESULTS_DIR, "exp_vjepa_null.csv"), index=False)
print(f"harvested from {log}")
print(f"part A rows: {len(dfa)}  conditions: {sorted(dfa.condition.unique())}")
print(f"part B rows: {len(dfb)}  conditions: {sorted(dfb.condition.unique())}")
print("\n=== part A (clean, both strides, both spaces) ===")
c = dfa[dfa.condition == "clean"].sort_values(["stride", "space", "feature"])
print(c[["space", "stride", "feature", "dim", "r2_xyz", "r2_rot", "r2_cont6"]].to_string(index=False))
