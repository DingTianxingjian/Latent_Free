"""Convert a SUBSET of Bridge V2 (downloaded TFDS shards) into the same interface
as LIBERO's data.py, so the existing probing protocol transfers unchanged.

Run in the isolated `envs/bridge` (tensorflow + tfds); the main torch env never
imports tensorflow.  We read only the shards present in data/bridge_raw via the
feature spec (builder_from_directory needs no shards to build `features`), so a
12-shard subset works without the full 1024.

Bridge action is 7-d [world_vector(3), rotation_delta(3), gripper(1)] — the SAME
layout as LIBERO [xyz(3), axis-angle(3), gripper(1)] — and image_0 is the primary
256x256x3 camera.  RLDS convention: action[t] applied at observation[t] produces
observation[t+1] (same as our LIBERO loader), so the probing targets transfer.

Outputs (cache/bridge/): frames.npy (N,256,256,3 u8), and index.npz with
starts/lengths/is_val/actions (N,7) — mirrors data.build_index()'s arrays.
"""

import glob
import os
import sys
import types

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.modules["audioop"] = types.ModuleType("audioop")   # orena py3.10 lacks audioop; tfds audio unused

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

RAW = "/lustre/fs1/home/ti874297/lf/latent_free/data/bridge_raw"
OUT = "/lustre/fs1/home/ti874297/lf/latent_free/cache/bridge_big"
MAX_EPISODES = 5000          # subset cap; 12 shards give ~this many
VAL_FRAC = 0.2
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    builder = tfds.builder_from_directory(RAW)
    features = builder.info.features
    shards = sorted(glob.glob(os.path.join(RAW, "bridge_dataset-train.tfrecord-*")))
    print(f"[bridge] {len(shards)} shards; decoding episodes (cap {MAX_EPISODES})", flush=True)

    ds = tf.data.TFRecordDataset(shards)
    frames_list, act_list, lengths = [], [], []
    n_ep = 0
    for raw in ds:
        ep = features.deserialize_example(raw.numpy())
        steps = next(iter(ep["steps"].batch(100000)))       # RLDS steps -> stacked tensors
        img = steps["observation"]["image_0"].numpy()      # (T,256,256,3) u8
        act = steps["action"].numpy().astype(np.float32)   # (T,7)
        T = len(img)
        if T < 6:                                          # need >= span+1 frames
            continue
        frames_list.append(img)
        act_list.append(act)
        lengths.append(T)
        n_ep += 1
        if n_ep % 100 == 0:
            print(f"  {n_ep} episodes, {sum(lengths)} frames", flush=True)
        if n_ep >= MAX_EPISODES:
            break

    lengths = np.array(lengths, np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    frames = np.concatenate(frames_list, 0)                # (N,256,256,3)
    actions = np.concatenate(act_list, 0)                  # (N,7)
    assert len(frames) == int(lengths.sum()) == len(actions)

    # gripper -> {0,1} (Bridge gripper is continuous [0,1]; threshold at 0.5)
    # keep raw in actions[:,6]; the probe binarizes as needed (matches LIBERO handling)

    rng = np.random.RandomState(SEED)
    is_val = np.zeros(len(lengths), bool)
    n_val = int(round(len(lengths) * VAL_FRAC))
    is_val[rng.permutation(len(lengths))[:n_val]] = True

    np.save(os.path.join(OUT, "frames.npy"), frames)
    np.savez(os.path.join(OUT, "index.npz"), starts=starts, lengths=lengths,
             is_val=is_val, actions=actions, seed=SEED)
    print(f"[bridge] wrote {n_ep} episodes, {len(frames)} frames "
          f"({frames.nbytes/2**30:.1f} GiB), {is_val.sum()} val episodes -> {OUT}", flush=True)
    print(f"[bridge] action stats: xyz|.|~{np.abs(actions[:,:3]).mean():.3f} "
          f"rot|.|~{np.abs(actions[:,3:6]).mean():.3f} "
          f"grip range [{actions[:,6].min():.2f},{actions[:,6].max():.2f}]", flush=True)


if __name__ == "__main__":
    main()
