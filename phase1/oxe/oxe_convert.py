"""Convert one OXE dataset from RLDS (data/oxe_raw/<name>/<ver>) into our cache layout
(cache/oxe/<name>/): frames.npy (N,256,256,3 u8) + index.npz (starts/lengths/is_val/actions/seed)
+ scene_meta.npz (instruction, file_path per episode) + meta.json -- the layout every loader in
claude_survey/ and latent_free/ already reads for bridge / bridge_big.

Fairness decisions, all recorded in meta.json:
  * action = UniVLA's OWN per-dataset standardisation (univla_transforms.py, vendored verbatim),
    so [dxyz, drot, gripper(+1 open)] means the same thing UniVLA's pipeline produced.
  * image  = the `primary` camera UniVLA's configs.py names for this dataset (rgb_static,
    front_rgb, agentview_rgb, wrist_image for dobbe ...), resized to 256x256 bilinear+antialias,
    which is what their LAM's `resolution: 256` does before its random-crop augmentation.
  * episode subset = first --max-eps episodes of a seeded shard-shuffled stream (whole
    episodes, never partial), val = 10% of those episodes by seeded draw.  Same recipe as
    bridge_big (5000 eps).  UniVLA's mixture WEIGHTS are not applied here -- they are a
    training-time sampling choice and live in the training config, not the cache.
Run in envs/bridge (tensorflow + tfds).
"""
import sys, os, json, time, types, argparse, re, shutil
sys.modules["audioop"] = types.ModuleType("audioop")          # py3.10 lacks audioop; tfds audio unused
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import univla_transforms as T

tf.config.set_visible_devices([], "GPU")
RAW  = "/lustre/fs1/home/ti874297/lf/latent_free/data/oxe_raw"
OUT  = os.environ.get("OXE_OUT", "/lustre/fs1/home/ti874297/lf/latent_free/cache/oxe")   # OXE_OUT=.../cache/oxe_full for uncapped conversions
CFG  = "/lustre/fs1/home/ti874297/lf/latent_free/third_party/UniVLA/prismatic/vla/datasets/rlds/oxe/configs.py"
IMG  = 256


def action_description(raw_dir):
    """Documented meaning of steps/action from the RLDS features.json (all sub-fields if the
    action is a dict).  This is the dataset author's statement, which the statistics below
    cannot replace: austin_buds is '6x end effector delta pose' yet every episode moves the
    same way, so a centre-of-workspace statistic misfires on it."""
    try: j = json.load(open(os.path.join(raw_dir, "features.json")))
    except Exception: return ""
    out = []
    def walk(d, path):
        if not isinstance(d, dict): return
        if "description" in d and "/steps/action" in path and "observation" not in path:
            out.append(f"{path.split('/steps/')[-1]}: {d['description']}")
        for k, v in d.items():
            walk(v, path if k in ("featuresDict", "features", "tensor", "sequence", "feature", "content") else path + "/" + k)
    walk(j, "")
    return " | ".join(out)[:600]

# Cases the features.json cannot settle, decided from the dataset's own documentation and
# UniVLA's transform, with the reason.  Add here rather than tuning the statistics.
ACTION_KIND_OVERRIDES = {
    "jaco_play": ("delta", "action.world_vector is a cartesian velocity command (OXE jaco_play card); transform concatenates it"),
    **{n: ("delta", "GNM action is a 2-D waypoint offset (dx, dy) in the robot frame; UniVLA's gnm_dataset_transform zero-pads dims 2..6 -- only xy are measured")
       for n in ("berkeley_gnm_recon", "berkeley_gnm_cory_hall", "berkeley_gnm_sac_son")},
}

def classify_action(desc, abs_over_diff, center_ratio, name=None):
    """'absolute' / 'delta' / 'unknown'.  Override table, then documented semantics, then statistics."""
    if name in ACTION_KIND_OVERRIDES: return ACTION_KIND_OVERRIDES[name][0], "override: " + ACTION_KIND_OVERRIDES[name][1]
    d = desc.lower()
    if any(w in d for w in ("delta", "relative", "velocit", "displacement", "rel_", "change in")): return "delta", "description"
    if any(w in d for w in ("absolute", "target pose", "target position", "position and orientation", "goal pose")): return "absolute", "description"
    if abs_over_diff > 5.0 and center_ratio > 2.0: return "absolute", "statistic"
    if abs_over_diff < 3.0: return "delta", "statistic"
    return "unknown", "statistic"

def univla_cfg(name):
    src = open(CFG).read()
    m = re.search(r'"%s":\s*\{(.*?)\n    \},' % re.escape(name), src, re.S); assert m, name
    blk = m.group(1)
    img = re.search(r'"primary":\s*("([^"]+)"|None)', blk).group(2)
    enc = re.search(r'"action_encoding":\s*ActionEncoding\.(\w+)', blk).group(1)
    return img, enc

def make_prep(img_key, transform):
    """Graph-mode episode -> (frames u8 (T,256,256,3), actions f32 (T,7), instruction, file_path).
    Stacks the step dataset inside tf.data so decode + transform + resize run in parallel
    workers instead of one eager Python loop (124 f/s eager vs ~10x with 16 threads)."""
    def prep(ep):
        steps = ep["steps"]
        # With decoders={"steps": SkipDecoding()} tfds hands the Sequence over as a dict of already
        # stacked (T, ...) tensors; without it, as a nested tf.data.Dataset.  Accept both.
        traj = steps if isinstance(steps, dict) else steps.batch(1 << 20).get_single_element()
        # Steps are read with tfds SkipDecoding (see main): image fields arrive as encoded bytes
        # and nothing else is decoded.  Decode ONLY the primary camera.  This is what the OXE
        # loaders (dlimp) do; decoding every field is what broke kuka (a non-image field whose
        # declared type does not match the data) and it also skips the secondary cameras.
        img = traj["observation"][img_key]
        if img.dtype == tf.string:
            img = tf.map_fn(lambda b: tf.io.decode_image(b, channels=3, expand_animations=False), img,
                            fn_output_signature=tf.TensorSpec([None, None, 3], tf.uint8))
        traj["observation"][img_key] = img
        traj = transform(traj)
        img = traj["observation"][img_key]
        if img.dtype != tf.uint8: img = tf.cast(img, tf.uint8)
        img = tf.image.resize(img, (IMG, IMG), method="bilinear", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        act = tf.cast(traj["action"], tf.float32)
        lang = traj["language_instruction"][0] if "language_instruction" in traj else tf.constant(b"")
        fp = ep["episode_metadata"]["file_path"] if "episode_metadata" in ep and "file_path" in ep["episode_metadata"] else tf.constant(b"")
        return img, act, lang, fp
    return prep

def episode_to_arrays(item):
    img, act, lang, fp = item
    return (img.numpy(), act.numpy(), lang.numpy().decode("utf-8", "ignore"),
            fp.numpy().decode("utf-8", "ignore") if fp.shape.rank == 0 else fp.numpy()[0].decode("utf-8", "ignore"))

def main(a):
    vers = sorted(os.listdir(os.path.join(RAW, a.name))); ver = vers[-1]
    d = os.path.join(RAW, a.name, ver)
    assert os.path.exists(os.path.join(d, "FETCH_DONE")), f"{a.name}: fetch not complete"
    img_key, enc = univla_cfg(a.name)
    transform = T.OXE_STANDARDIZATION_TRANSFORMS[a.name]
    out = os.path.join(OUT, a.name); os.makedirs(out, exist_ok=True)
    b = tfds.builder_from_directory(d)
    split = "train" if "train" in b.info.splits else list(b.info.splits)[0]
    n_avail = b.info.splits[split].num_examples
    rc = tfds.ReadConfig(shuffle_seed=a.seed, try_autocache=False, skip_prefetch=True)
    ds = b.as_dataset(split=split, shuffle_files=True, read_config=rc,
                      decoders={"steps": tfds.decode.SkipDecoding()})
    print(f"[oxe:{a.name}] {ver} split={split} episodes_available={n_avail} taking<= {a.max_eps} "
          f"img={img_key} action_encoding={enc}", flush=True)
    # Newton (2026-10-16): write frames.npy directly -- a fixed-length (128-byte) .npy v1 header holding a placeholder
    # shape, rewritten in place at the end -- instead of frames.raw + a full copy, which doubled peak disk and I/O.
    npy = os.path.join(out, "frames.npy")
    def npy_header(n):
        h = "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d, %d, 3), }" % (n, IMG, IMG)
        h = h.ljust(128 - 10 - 1) + "\n"; assert len(h) == 118, len(h)
        return b"\x93NUMPY\x01\x00" + len(h).to_bytes(2, "little") + h.encode("latin1")
    raw = open(npy + ".part", "wb"); raw.write(npy_header(0))
    lengths, acts, langs, file_paths = [], [], [], []
    t0, nf, bad = time.time(), 0, 0
    ds = ds.take(a.max_eps).map(make_prep(img_key, transform), num_parallel_calls=a.threads, deterministic=True).prefetch(2 * a.threads)
    for k, item in enumerate(ds):
        try:
            img, act, lang, file_path = episode_to_arrays(item)
        except Exception as e:
            bad += 1; print(f"[oxe:{a.name}] episode {k} skipped: {type(e).__name__}: {str(e)[:120]}", flush=True); continue
        assert img.shape[1:] == (IMG, IMG, 3) and img.dtype == np.uint8, img.shape
        assert act.shape == (len(img), 7), (act.shape, len(img))
        raw.write(np.ascontiguousarray(img).tobytes())
        lengths.append(len(img)); acts.append(act); langs.append(lang); file_paths.append(file_path); nf += len(img)
        if (k + 1) % 200 == 0:
            print(f"[oxe:{a.name}] {k+1} eps {nf} frames {time.time()-t0:.0f}s", flush=True)
    raw.seek(0); raw.write(npy_header(nf)); raw.close(); os.replace(npy + ".part", npy)
    lengths = np.asarray(lengths, np.int64); starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    actions = np.concatenate(acts, 0).astype(np.float32); assert len(actions) == nf == int(lengths.sum())
    rs = np.random.RandomState(a.seed); is_val = np.zeros(len(lengths), bool)
    is_val[rs.choice(len(lengths), max(1, len(lengths) // 10), replace=False)] = True
    chk = np.load(npy, mmap_mode="r"); assert chk.shape == (nf, IMG, IMG, 3), chk.shape
    np.savez(os.path.join(out, "index.npz"), starts=starts, lengths=lengths, is_val=is_val, actions=actions, seed=np.int64(a.seed))
    np.savez(os.path.join(out, "scene_meta.npz"), instruction=np.array(langs, dtype=object), file_path=np.array(file_paths, dtype=object),
             lengths=lengths, dataset=np.array([a.name] * len(lengths), dtype=object))
    # Absolute-pose vs delta diagnostic.  UniVLA labels both "EEF_POS" (its LAM never reads
    # actions); our displacement probes must know.  Two statistics on xyz:
    #   abs_over_diff = |a| / |a[t+1]-a[t]|   large for absolute poses, but ALSO large for
    #                   smooth/held velocity commands (jaco_play: 0.08 m/s held for many steps),
    #                   so it cannot decide alone.
    #   center_ratio  = |mean over episodes of per-episode mean| / std over episodes of the
    #                   per-episode mean.  An absolute pose sits near a workspace centre in
    #                   every episode (ucsd_kitchen x ~ 200 mm always) -> large; a delta or
    #                   velocity command averages to a direction that changes sign across
    #                   episodes -> small.  Median over the three axes.
    # Flag = both large.  Heuristic on the command's FORM only: it does not validate units,
    # axis conventions, or that commands equal realised motion.
    num = den = 0.0; ep_means = []
    for s0, l in zip(starts, lengths):
        if l < 3: continue
        seg = actions[s0:s0 + l, :3]; num += np.abs(seg).sum(); den += np.abs(np.diff(seg, axis=0)).sum() * l / (l - 1)
        ep_means.append(seg.mean(0))
    ep_means = np.asarray(ep_means)
    abs_over_diff = round(float(num / max(den, 1e-9)), 3)
    center_ratio = round(float(np.median(np.abs(ep_means.mean(0)) / (ep_means.std(0) + 1e-9))), 3)
    desc = action_description(d)
    action_kind, action_kind_source = classify_action(desc, abs_over_diff, center_ratio, a.name)
    action_looks_absolute = action_kind == "absolute"
    json.dump(dict(name=a.name, version=ver, split=split, episodes_available=int(n_avail), episodes_taken=int(len(lengths)),
                   action_abs_over_diff_xyz=abs_over_diff, action_center_ratio_xyz=center_ratio,
                   action_kind=action_kind, action_kind_source=action_kind_source, action_description=desc, action_looks_absolute=action_looks_absolute,
                   episodes_skipped=bad, frames=int(nf), image_key=img_key, action_encoding=enc, img_size=IMG, seed=a.seed,
                   max_eps=a.max_eps, transform="univla_transforms.OXE_STANDARDIZATION_TRANSFORMS[name] (vendored verbatim)",
                   n_unique_instructions=int(len(set(langs))), val_fraction=float(is_val.mean()),
                   action_abs_mean=[float(x) for x in np.abs(actions).mean(0)], seconds=int(time.time() - t0)),
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"[oxe:{a.name}] DONE eps={len(lengths)} skipped={bad} frames={nf} ({os.path.getsize(npy)/1e9:.1f} GB) "
          f"uniq_instr={len(set(langs))} action={action_kind}({action_kind_source}) act|.|={np.abs(actions).mean(0).round(3).tolist()} {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--max-eps", type=int, default=5000); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=16)
    main(ap.parse_args())
