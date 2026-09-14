"""Render the results CSVs as markdown tables for SUMMARY.md / slides.

Reads whatever exists in results/ and prints paste-ready tables, so no number in
SUMMARY.md is ever transcribed by hand.
"""

import os

import pandas as pd

import data

R = data.RESULTS_DIR
GROUPS = ["xyz", "rot", "cont6", "gripper"]
GLAB = {"xyz": "trans xyz R²", "rot": "rot R²", "cont6": "all-6 R²",
        "gripper": "gripper bal.acc"}
CLAB = {"clean": "clean", "patch": "patch (moving)", "patch_static": "patch (frozen)",
        "border": "border (playing)", "border_static": "border (frozen)"}
FLAB = {"diff": "Δf", "single": "f(o_t)",
        "pair": "[f(o_t), f(o_t+s)]", "pair_shuf": "[f(o_t), f(o_t′)] shuffled",
        "univla_lam_z_q": "UniVLA LAM z_q", "univla_lam_z_prevq": "UniVLA LAM z (pre-VQ)"}
SLAB = {"cls": "CLS 768-d", "pmean": "patch-mean 768-d", "pg4": "patch-4×4 12288-d"}


def pivot(df, index_cols):
    d = df[df.dim_group.isin(GROUPS)].copy()
    p = d.pivot_table(index=index_cols, columns="dim_group", values="value",
                      sort=False)
    return p.reindex(columns=GROUPS)


def md(p, headers):
    cols = [GLAB[c] for c in p.columns]
    out = ["| " + " | ".join(headers + cols) + " |",
           "|" + "|".join(["---"] * (len(headers) + len(cols))) + "|"]
    for idx, row in p.iterrows():
        idx = idx if isinstance(idx, tuple) else (idx,)
        cells = [f"{v:.3f}" if pd.notna(v) else "—" for v in row]
        out.append("| " + " | ".join([str(i) for i in idx] + cells) + " |")
    return "\n".join(out)


def main():
    f1 = os.path.join(R, "exp1_decodability.csv")
    if os.path.exists(f1):
        df = pd.read_csv(f1)
        df["feature"] = df.feature.map(lambda x: FLAB.get(x, x))
        print("### Exp 1 — decodability from a frozen DINOv2 (clean frames)\n")
        print(md(pivot(df, ["stride", "feature"]), ["stride", "features"]))
        alphas = df[df.dim_group == "cont6"][["stride", "feature", "hyperparam"]]
        print("\nselected ridge alpha (by val all-6 R²): " +
              ", ".join(f"{r.stride}/{r.feature}: {r.hyperparam}"
                        for r in alphas.itertuples()))

    f3 = os.path.join(R, "exp3_lam.csv")
    if os.path.exists(f3):
        d3 = pd.read_csv(f3)
        d3["feature"] = d3.feature.map(lambda x: FLAB.get(x, x))
        print("\n### Exp 3 — pretrained UniVLA stage-1 LAM (same protocol)\n")
        print(md(pivot(d3, ["stride", "feature"]), ["stride", "latent"]))
        u = d3[["codebook_used", "latent_dim"]].drop_duplicates()
        print(f"\ncodebook entries used / latent dim: "
              f"{u.codebook_used.iloc[0]}/16, {u.latent_dim.iloc[0]}-d")

    f4 = os.path.join(R, "exp4_patch_matrix.csv")
    if os.path.exists(f4):
        d4 = pd.read_csv(f4)
        d4["space"] = d4.space.map(lambda x: SLAB.get(x, x))
        d4["feature"] = d4.feature.map(lambda x: FLAB.get(x, x))
        print("\n### Exp 4 — CLS vs patch tokens, single vs both frames\n")
        print(md(pivot(d4, ["stride", "space", "feature"]),
                 ["stride", "space", "input"]))
        edge = d4[(d4.dim_group == "cont6") &
                  d4.hyperparam.isin(["alpha=0.0001", "alpha=100000"])]
        if len(edge):
            print(f"\n**{len(edge)} rows selected an alpha at the edge of the "
                  f"1e-4..1e5 grid** — see caveats.")

    f4d = os.path.join(R, "exp4_state_decomposition.csv")
    if os.path.exists(f4d):
        d = pd.read_csv(f4d)
        d = d[d.dim_group.isin(["xyz", "cont6"])]
        print("\n### Exp 4 — how much of a two-frame score is single-frame state\n")
        print("| stride | space | group | single | pair_shuf | pair | "
              "capacity gain | transition | single/pair |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in d.itertuples():
            print(f"| {r.stride} | {SLAB.get(r.space, r.space)} | {r.dim_group} "
                  f"| {r.single:.3f} | {r.pair_shuf:.3f} | {r.pair:.3f} "
                  f"| {r.capacity_gain:+.3f} | {r.transition:+.3f} "
                  f"| {r.frac_state:.0%} |")

    f2 = os.path.join(R, "exp2_distractor.csv")
    if os.path.exists(f2):
        df = pd.read_csv(f2)
        d = df[df.feature == "diff"].copy()
        d["condition"] = d.condition.map(lambda x: CLAB.get(x, x))
        print("\n### Exp 2 — decodability from Δf under distractors\n")
        print(md(pivot(d, ["stride", "condition"]), ["stride", "condition"]))

    f2b = os.path.join(R, "exp2_pca_action_energy.csv")
    if os.path.exists(f2b):
        p = pd.read_csv(f2b)
        p = p[p.k.isin([8, 16, 32])]
        print("\n### Exp 2 — PCA of Δf vs the clean action subspace\n")
        print("| stride | condition | k | var explained | action energy in top-k |")
        print("|---|---|---|---|---|")
        for r in p.itertuples():
            print(f"| {r.stride} | {CLAB.get(r.condition, r.condition)} | {r.k} "
                  f"| {r.cum_var_explained:.3f} | {r.action_energy_in_topk:.3f} |")

    f2bn = os.path.join(R, "exp2b_null.csv")
    if os.path.exists(f2bn):
        p = pd.read_csv(f2bn)
        p = p[p.k.isin([8, 32, 128])]
        s2 = {"cls": "CLS 768", "pmean": "patch-mean 768", "pg4": "patch 4×4 12288"}
        print("\n### Exp 2B null — top-k vs random-k vs bottom-k action recovery "
              "(clean, xyz R²)\n")
        print("| space | s | k | top-k (% full) | random-k | bottom-k | top/rand |")
        print("|---|---|---|---|---|---|---|")
        for r in p.itertuples():
            print(f"| {s2.get(r.space, r.space)} | {r.stride} | {r.k} "
                  f"| {r.r2_xyz_top:.3f} ({r.frac_top*100:.0f}%) "
                  f"| {r.r2_xyz_rand_mean:.3f} | {r.r2_xyz_bottom:.3f} "
                  f"| {r.r2_xyz_top / r.r2_xyz_rand_mean:.1f}× |")

    f2bp = os.path.join(R, "exp2b_patch_pca.csv")
    if os.path.exists(f2bp):
        p = pd.read_csv(f2bp)
        p = p[p.k.isin([8, 32, 128])]
        s2 = {"cls": "CLS 768", "pmean": "patch-mean 768", "pg4": "patch 4×4 12288"}
        print("\n### Exp 2B on patch tokens — top-k PCA bottleneck (clean, % of full)\n")
        print("| space | stride | k | % var kept | % of full action | action vs var |")
        print("|---|---|---|---|---|---|")
        for r in p.itertuples():
            cv, ca = round(r.cum_var_explained * 100), round(r.frac_of_full_xyz * 100)
            rel = "leads" if ca > cv + 1 else ("lags" if ca < cv - 1 else "~tracks")
            print(f"| {s2.get(r.space, r.space)} | {r.stride} | {r.k} "
                  f"| {cv}% | {ca}% | {rel} |")


if __name__ == "__main__":
    main()
