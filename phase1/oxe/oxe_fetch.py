"""Fetch UniVLA's `omni_magic_soup_plus_plus` mixture from the public OXE bucket.

Pulls dataset_info.json + features.json + every shard of every split for each dataset in
DATASETS into data/oxe_raw/<name>/<version>/, so `tfds.builder_from_directory` works on it
exactly as it does for bridge_raw.  Idempotent: a shard whose size already matches the
bucket's Content-Length is skipped, so the script can be re-run after any interruption.

What is NOT here (decisions recorded in ../../../SUMMARY.md):
  ego4d_split_1..4: not in the bucket (licence + separate curated-clip pipeline) -- the one
  real gap.  bridge_oxe: the OXE re-encode is not in this bucket; we hold the RAIL copy.
  (An earlier version also omitted kuka / language_table / fmb / furniture_bench on a wrong
  "3% of samples" estimate that ignored balance_weights; they are included now.)
"""
import json, os, sys, urllib.request, concurrent.futures as cf, subprocess, time
ROOT = "/anvil/scratch/x-tding2/latent_free/data/oxe_raw"
BUCKET = "https://storage.googleapis.com/gresearch/robotics"
# (name, version, UniVLA sampling weight)
DATASETS = [
    ("fractal20220817_data", "0.1.0", 0.5), ("taco_play", "0.1.0", 2.0), ("jaco_play", "0.1.0", 1.0),
    ("berkeley_cable_routing", "0.1.0", 1.0), ("roboturk", "0.1.0", 2.0), ("viola", "0.1.0", 2.0),
    ("berkeley_autolab_ur5", "0.1.0", 2.0), ("toto", "0.1.0", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", "0.1.0", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", "0.1.0", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", "0.1.0", 3.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", "0.1.0", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", "0.1.0", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", "0.1.0", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", "0.1.0", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", "0.1.0", 1.0),
    ("utaustin_mutex", "0.1.0", 1.0), ("berkeley_fanuc_manipulation", "0.1.0", 2.0),
    ("cmu_stretch", "0.1.0", 1.0), ("bc_z", "0.1.0", 0.2), ("dobbe", "0.0.1", 0.2),
    ("berkeley_gnm_recon", "0.1.0", 2.0), ("berkeley_gnm_cory_hall", "0.1.0", 2.0),
    ("berkeley_gnm_sac_son", "0.1.0", 2.0),
    # the four large sources, added 2026-09-06 after Codex's audit: UniVLA samples with
    # balance_weights=True (p_i ~ w_i * num_transitions_i), so a small w on a huge source is
    # not a small share.  Fetched last; the 5000-episode conversion cap applies to them too.
    ("kuka", "0.1.0", 0.1), ("language_table", "0.1.0", 0.1),
    ("furniture_bench_dataset_converted_externally_to_rlds", "0.1.0", 0.1), ("fmb", "0.0.1", 1.0),
    # UniVLA's `bridge_oxe` = the OXE re-encode of Bridge V2 in this bucket under the name `bridge` (357 GB, 1024 train
    # shards).  Added 2026-09-10 (user: train on everything); the RAIL server that held our 97-shard copy is gone.
    ("bridge_oxe", "0.1.0", 1.0, "bridge"),
]

def shard_urls(name, ver, bucket_name=None):
    bname = bucket_name or name
    base = f"{BUCKET}/{bname}/{ver}"
    info = json.load(urllib.request.urlopen(f"{base}/dataset_info.json", timeout=60))
    out, total = [], 0
    for sp in info["splits"]:
        n = len(sp["shardLengths"]); total += int(sp.get("numBytes", 0))
        pat = sp.get("filepathTemplate", "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}")
        for i in range(n):
            fn = (pat.replace("{DATASET}", bname).replace("{SPLIT}", sp["name"])
                     .replace("{FILEFORMAT}", "tfrecord").replace("{SHARD_X_OF_Y}", f"{i:05d}-of-{n:05d}"))
            out.append(fn)
    return base, ["dataset_info.json", "features.json"] + out, total

def fetch_one(base, dest, fn):
    url, path = f"{base}/{fn}", os.path.join(dest, fn)
    for attempt in range(5):
        try:
            if os.path.exists(path):
                want = int(urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60).headers["Content-Length"])
                if os.path.getsize(path) == want: return fn, "skip", want
            r = subprocess.run(["curl", "-sS", "-f", "--retry", "3", "-o", path + ".part", url], capture_output=True)
            if r.returncode == 0:
                os.replace(path + ".part", path); return fn, "ok", os.path.getsize(path)
        except Exception as e:
            err = str(e)
        time.sleep(5 * (attempt + 1))
    return fn, "FAIL", 0

def main(names, workers):
    for entry in DATASETS:
        name, ver, w = entry[:3]; bname = entry[3] if len(entry) > 3 else None
        if names and name not in names: continue
        dest = os.path.join(ROOT, name, ver); os.makedirs(dest, exist_ok=True)
        base, files, total = shard_urls(name, ver, bname)
        t0, got, fails = time.time(), 0, 0
        print(f"[fetch] {name} {ver}: {len(files)} files, {total/1e9:.1f} GB", flush=True)
        with cf.ThreadPoolExecutor(workers) as ex:
            for k, (fn, st, nb) in enumerate(ex.map(lambda f: fetch_one(base, dest, f), files)):
                got += nb; fails += st == "FAIL"
                if st == "FAIL": print(f"[fetch]   FAIL {fn}", flush=True)
                if (k + 1) % 50 == 0 or k + 1 == len(files):
                    dt = time.time() - t0
                    print(f"[fetch]   {name} {k+1}/{len(files)} {got/1e9:.1f} GB {got/1e6/max(dt,1):.0f} MB/s", flush=True)
        open(os.path.join(dest, "FETCH_DONE" if not fails else "FETCH_INCOMPLETE"), "w").write(f"{len(files)-fails}/{len(files)} files {got} bytes\n")
        print(f"[fetch] {name} DONE fails={fails} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); main(a.only, a.workers)
