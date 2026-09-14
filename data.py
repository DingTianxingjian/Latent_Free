"""LIBERO libero_spatial data utilities.

Structural facts VERIFIED against the actual hdf5 files (run `python data.py` to
reprint them):

  * Each demo group holds `actions` (T,7) and `obs/agentview_rgb` (T,128,128,3).
  * len(actions) == len(frames) == T exactly, for all 500 episodes -- there is NO
    off-by-one offset in this release.  This is the robomimic convention:
    obs[t] together with actions[t] produces obs[t+1].
    => the transition (o_t -> o_{t+s}) is explained by actions[t : t+s].
  * Images are stored with the hdf5 attr `macros_image_convention == 'opengl'`,
    i.e. bottom-up.  They MUST be flipped vertically to be upright; every loader
    here does that, so nothing downstream should flip again.
  * actions[:, 6] (gripper) takes exactly two values, -1.0 and +1.0.
  * actions[:, 0:3] are EEF translation deltas, actions[:, 3:6] axis-angle
    rotation deltas.  Translation std is ~7x the rotation std.
"""

import os
import glob

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LIBERO_DIR = os.path.join(ROOT, "data", "LIBERO", "libero_spatial")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGS_DIR = os.path.join(ROOT, "figs")

IMG_KEY = "obs/agentview_rgb"
SEED = 0
VAL_FRAC = 0.2
STRIDES = (1, 4)

# action layout
XYZ_DIMS = [0, 1, 2]
ROT_DIMS = [3, 4, 5]
CONT_DIMS = [0, 1, 2, 3, 4, 5]
GRIP_DIM = 6
DIM_GROUPS = {"xyz": XYZ_DIMS, "rot": ROT_DIMS, "cont6": CONT_DIMS}


def task_files():
    """The 10 libero_spatial task hdf5 files, in a fixed sorted order."""
    fs = sorted(glob.glob(os.path.join(LIBERO_DIR, "*.hdf5")))
    if not fs:
        raise FileNotFoundError(f"no hdf5 files under {LIBERO_DIR}")
    return fs


def list_episodes():
    """Enumerate every episode as a dict, in a fixed deterministic order.

    Returns a list of dicts with keys: ep_id, task_id, task, path, demo, length.
    """
    eps = []
    for task_id, path in enumerate(task_files()):
        task = os.path.basename(path).replace("_demo.hdf5", "")
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            for demo in demos:
                g = f["data"][demo]
                T = g["actions"].shape[0]
                # fail loudly if the frame/action alignment assumption breaks
                T_img = g[IMG_KEY].shape[0]
                assert T == T_img, (
                    f"alignment violated in {path}:{demo}: "
                    f"len(actions)={T} != len(frames)={T_img}"
                )
                eps.append(
                    dict(ep_id=len(eps), task_id=task_id, task=task,
                         path=path, demo=demo, length=int(T))
                )
    return eps


def episode_split(eps, seed=SEED, val_frac=VAL_FRAC):
    """80/20 split AT THE EPISODE LEVEL (never by frame -- adjacent frames are
    nearly identical and a frame-level split leaks).

    The split is done within each task so both splits contain all 10 tasks.
    Returns a bool array `is_val` of length len(eps).
    """
    is_val = np.zeros(len(eps), dtype=bool)
    task_ids = np.array([e["task_id"] for e in eps])
    for t in np.unique(task_ids):
        idx = np.where(task_ids == t)[0]
        rng = np.random.RandomState(seed * 1000 + int(t))
        perm = rng.permutation(len(idx))
        n_val = int(round(val_frac * len(idx)))
        is_val[idx[perm[:n_val]]] = True
    return is_val


def load_episode_frames(ep):
    """(T,128,128,3) uint8, flipped to upright.  See module docstring."""
    with h5py.File(ep["path"], "r") as f:
        img = f["data"][ep["demo"]][IMG_KEY][:]
    return np.ascontiguousarray(img[:, ::-1])  # opengl bottom-up -> upright


def load_episode_actions(ep):
    """(T,7) float32."""
    with h5py.File(ep["path"], "r") as f:
        return f["data"][ep["demo"]]["actions"][:].astype(np.float32)


def build_index():
    """Build (and cache) the episode index + the concatenated action array.

    Everything downstream addresses frames by a single global row id; `starts`
    and `lengths` map episodes onto that array.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "index.npz")
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)

    eps = list_episodes()
    is_val = episode_split(eps)
    lengths = np.array([e["length"] for e in eps], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    actions = np.zeros((int(lengths.sum()), 7), dtype=np.float32)
    for e, s in zip(eps, starts):
        actions[s:s + e["length"]] = load_episode_actions(e)

    np.savez(
        path,
        starts=starts,
        lengths=lengths,
        is_val=is_val,
        actions=actions,
        task_id=np.array([e["task_id"] for e in eps]),
        task=np.array([e["task"] for e in eps]),
        demo=np.array([e["demo"] for e in eps]),
        path=np.array([e["path"] for e in eps]),
        seed=SEED,
    )
    return np.load(path, allow_pickle=True)


def episode_pairs(index, e, stride):
    """Pairs for ONE episode -- the single source of truth for how a frame pair
    maps to its target, shared by exp1/exp2 (cached features) and exp3 (a LAM
    that has to re-encode raw frames).

    Returns
        t         : local frame index of the first frame of each pair
        i, j      : global row ids of frame t and frame t+stride
        y_cont    : (n,6) sum of actions over [t, t+stride)  -- the net command
        y_grip    : (n,)  gripper at t+stride-1, as {0,1}
    """
    s, T = int(index["starts"][e]), int(index["lengths"][e])
    actions = index["actions"]
    n = T - stride                          # valid t: 0 .. T-1-stride
    if n <= 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, np.zeros((0, 6), np.float32), z
    t = np.arange(n, dtype=np.int64)
    # cumulative sum trick: sum(actions[s+t : s+t+stride])
    c = np.cumsum(np.vstack([np.zeros((1, 7), np.float64),
                             actions[s:s + T].astype(np.float64)]), axis=0)
    y_cont = (c[t + stride, :6] - c[t, :6]).astype(np.float32)
    y_grip = (actions[s + t + stride - 1, GRIP_DIM] > 0).astype(np.int64)
    return t, s + t, s + t + stride, y_cont, y_grip


def build_pairs(index, stride, split):
    """All pairs (o_t, o_{t+stride}) for every episode in `split`.

    Returns i, j (global row ids), y_cont (n,6), y_grip (n,) in {0,1}.
    """
    assert split in ("train", "val")
    want_val = split == "val"
    is_val = index["is_val"]

    i_all, j_all, yc_all, yg_all = [], [], [], []
    for e in range(len(index["starts"])):
        if bool(is_val[e]) != want_val:
            continue
        _, i, j, yc, yg = episode_pairs(index, e, stride)
        i_all.append(i); j_all.append(j); yc_all.append(yc); yg_all.append(yg)

    return (np.concatenate(i_all), np.concatenate(j_all),
            np.concatenate(yc_all), np.concatenate(yg_all))


def print_dataset_stats():
    eps = list_episodes()
    is_val = episode_split(eps)
    lengths = np.array([e["length"] for e in eps])
    print(f"suite dir           : {LIBERO_DIR}")
    print(f"tasks               : {len(task_files())}")
    print(f"episodes            : {len(eps)}  (train {int((~is_val).sum())} / "
          f"val {int(is_val.sum())}, episode-level split, seed={SEED})")
    print(f"frames              : {int(lengths.sum())}")
    print(f"episode length      : min {lengths.min()} / mean {lengths.mean():.1f} "
          f"/ max {lengths.max()}")
    print(f"frame/action offset : 0  (len(actions) == len(frames), asserted per episode)")

    acts = np.concatenate([load_episode_actions(e) for e in eps])
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    print("action ranges (raw, per step):")
    for k, nm in enumerate(names):
        c = acts[:, k]
        print(f"   {nm:5s} min {c.min():7.3f}  max {c.max():7.3f}  "
              f"mean {c.mean():8.4f}  std {c.std():6.3f}")
    u = np.unique(acts[:, GRIP_DIM])
    print(f"   gripper unique values: {u}  (frac +1 = {(acts[:, 6] > 0).mean():.3f})")

    for stride in STRIDES:
        idx = build_index()
        for split in ("train", "val"):
            i, j, yc, yg = build_pairs(idx, stride, split)
            print(f"stride {stride} {split:5s}: {len(i):6d} pairs   "
                  f"|y_cont| mean {np.abs(yc).mean(axis=0).round(3)}   "
                  f"grip pos frac {yg.mean():.3f}")


if __name__ == "__main__":
    print_dataset_stats()
