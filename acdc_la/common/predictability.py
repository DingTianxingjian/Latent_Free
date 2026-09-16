"""How much of a latent is determined by what the policy sees at DEPLOYMENT?

Every diagnostic in this repo points at the posterior-to-policy gap, but the gap itself was only ever measured
through a trained policy.  This measures it directly and cheaply from an existing validation dump:

    P = R2( z | F_0 )        ridge from the first-frame quadrant features to the code, per source, episode-disjoint

SCOPE (retraction 2026-09-15): this is a WITHIN-FAMILY instrument only -- valid for comparing arms that share a code
space, a training domain and a reader.  It is NOT valid against UniVLA: their deployed code is 4 indices into a
16-entry codebook, not the 2048-d embedding this regresses; the split here is per-source episode-disjoint on OXE, not their
in-domain LIBERO; and the reader has no language, which is decisive for a code whose LIBERO majority floor is 0.469.
Measured directly from their trained VLA, ~79 % of their label IS recovered at deployment.  For cross-method claims
use held-out latent-token cross-entropy in bits against the label entropy instead.

F_0 is exactly the deployment information set minus language (the dumps carry no instruction), so P is a lower
bound on what any observation-conditioned policy can recover.  Reported next to the code's action content
S = R2(action | z), because the quantity that matters downstream is not either alone but how much of the action
content survives prediction:  usable ~ S x P.

    python -m acdc_la.common.predictability s1_acdc_1ep s1_mean_1ep univla_matched   -> results/predictability.json
"""
import sys, json, os
import numpy as np

from ..paths import RESULTS
from . import readers as R

COLS = ["P_z_from_state", "S_action_from_z", "usable", "ref_action_from_state", "dims"]


def per_dump(d):
    Z = d["Z"].astype(np.float64); F0 = d["F0Q"].astype(np.float64); NET = d["NET"]; GE, SI = d["GE"], d["SI"]
    rows = {}
    for s, tr, te in R.source_splits(GE, SI):
        row = dict(n_test=int(te.sum()), dims=Z.shape[1])
        G, _ = R.fit_reader(F0, Z, GE, tr); row["P_z_from_state"] = R.r2(Z[te], G(F0[te]))      # predictability of the code
        G2, _ = R.fit_reader(Z, NET, GE, tr); row["S_action_from_z"] = R.r2(NET[te], G2(Z[te]))  # action content of the code
        G3, _ = R.fit_reader(F0, NET, GE, tr); row["ref_action_from_state"] = R.r2(NET[te], G3(F0[te]))
        row["usable"] = max(0.0, row["P_z_from_state"]) * max(0.0, row["S_action_from_z"])
        rows[d["names"][s]] = row
    return R.aggregate(rows)


def main(tags):
    p = os.path.join(RESULTS, "predictability.json"); res = json.load(open(p)) if os.path.exists(p) else {}
    for t in tags:
        d = R.load_arrays(t)
        if "F0Q" not in d: print(f"[predict] {t}: dump has no F0Q, skipped", flush=True); continue
        res[t] = per_dump(d); json.dump(res, open(p, "w"), indent=1)
        print(f"[predict] {t}: P(z|state) {res[t]['P_z_from_state']:.3f}  S(action|z) {res[t]['S_action_from_z']:.3f}  "
              f"usable {res[t]['usable']:.3f}  (action from state alone {res[t]['ref_action_from_state']:.3f})", flush=True)
    R.table(res, COLS, [t for t in tags if t in res])


if __name__ == "__main__":
    main(sys.argv[1:])
