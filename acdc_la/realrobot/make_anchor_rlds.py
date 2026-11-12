"""CHUNK-ANCHORED delta actions for the Piper RLDS (2026-11-11).

Why the per-step delta (make_delta_rlds.py) breaks long chunks
--------------------------------------------------------------
There, chunk entry k is trained as  action[s+k] - joint_state[s+k]:  each entry is relative to where the arm is AT STEP
k.  At deployment the robot adds entry k to the joints it measures at step k.  Within one chunk the policy sees no new
image, so if the arm is anywhere other than where the demonstrator's arm was at step k (tracking lag, a different
load, the policy's own earlier error), entry k is added to the wrong base and nothing pulls it back: the error
accumulates over the chunk.  With 1-2 steps per chunk the next image corrects it; with long chunks it drifts.

The fix (pi0 / UMI style)
-------------------------
Every entry of the chunk is relative to the joints measured ONCE, when the chunk is predicted:
    anchored[k] = action[s+k] - joint_state[s],   k = 0 .. W-1
At deployment: targets[k] = q_s + anchored[k], with q_s read once per chunk.  The targets of a chunk no longer depend
on how the arm tracks within it, so the offline open-loop error IS the deployment open-loop error (images aside).

How it is stored
----------------
The anchor depends on the chunk's start s, so it cannot be precomputed per step.  The RLDS action field becomes
13-D RAW = [absolute target (6), gripper (1), measured joints (6)], with the normalisation mask all-false
(ActionEncoding.JOINT_ANCHOR in univla_piper.patch).  datasets_ctrl.RLDSBatchTransformCtrl forms
action[s+k, :6] - action[s, 7:13] per sample and normalises it with the q01/q99 written here
(`anchor_stats.json`, computed over exactly the windows the trainer samples: 14-frame window, offset ro in {0, 1},
indices floored at 0).  The dataset is named <dataset>_anc so it can never be mistaken for the 7-D ones.

  PYTHONPATH=.:third_party/UniVLA envs/univla_vla/bin/python -c "import sys; sys.argv=['x','--dataset','press_button'];
      from acdc_la.realrobot.make_anchor_rlds import main; main()"
"""
import argparse, json, os
import numpy as np

W, WIN = 12, 14          # chunk length, RLDS window (window_size + 2) -- finetune_libero_ctrl.py


def anchored_windows(acts, js):
    """All (W, 6) anchored chunks the trainer can sample from one episode (both offsets ro)."""
    T = len(acts); out = []
    for t in range(T):
        for ro in (0, 1):
            s = t - (WIN - 1) + ro
            idx = np.maximum(np.arange(s, s + W), 0)
            out.append(acts[idx, :6] - js[max(s, 0), :6])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/real_rlds"); ap.add_argument("--out", default="data/real_rlds_anchor")
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf; tf.config.set_visible_devices([], "GPU")
    import tensorflow_datasets as tfds

    name = a.dataset + "_anc"
    src = tfds.builder_from_directory(os.path.join(a.src, a.dataset, "1.0.0"))
    eps = []
    for epi in src.as_dataset(split="train"):
        steps = list(epi["steps"])
        acts = np.stack([s["action"].numpy() for s in steps]).astype(np.float32)
        js = np.stack([s["observation"]["joint_state"].numpy() for s in steps]).astype(np.float32)
        assert np.abs(acts[:, :6]).max() > 20, "src must be the ABSOLUTE RLDS"
        eps.append(dict(idx=int(epi["episode_metadata"]["episode_index"].numpy()),
                        path=epi["episode_metadata"]["file_path"].numpy().decode(),
                        instr=steps[0]["language_instruction"].numpy().decode(),
                        img=[s["observation"]["image"].numpy() for s in steps],
                        wrist=[s["observation"]["wrist_image"].numpy() for s in steps],
                        state=[s["observation"]["state"].numpy() for s in steps],
                        jstate=[s["observation"]["joint_state"].numpy() for s in steps],
                        acts=acts, js=js, action=np.concatenate([acts, js[:, :6]], 1)))
    # stats over the TRAINING episodes only, in the same order TFDS splits them (train[:95%])
    train_idx = {int(e["episode_metadata"]["episode_index"].numpy()) for e in src.as_dataset(split="train[:95%]")}
    ch = np.concatenate([np.stack(anchored_windows(e["acts"], e["js"])) for e in eps if e["idx"] in train_idx])  # (N, W, 6)
    flat = ch.reshape(-1, 6); g = np.concatenate([e["acts"][:, 6] for e in eps])
    st = dict(q01=np.quantile(flat, 0.01, 0).tolist() + [0.0], q99=np.quantile(flat, 0.99, 0).tolist() + [1.0],
              mean=flat.mean(0).tolist() + [float(g.mean())], std=flat.std(0).tolist() + [float(g.std())],
              min=flat.min(0).tolist() + [float(g.min())], max=flat.max(0).tolist() + [float(g.max())],
              mask=[True] * 6 + [False],
              # PER CHUNK POSITION k: entry 0 moves ~1 deg, entry 11 up to ~130 deg (press_button j1).  One range for
              # all k would cost entry 0 most of its resolution, so each k gets its own q01/q99 (gripper raw: 0/1).
              q01_k=[np.quantile(ch[:, k], 0.01, 0).tolist() + [0.0] for k in range(W)],
              q99_k=[np.quantile(ch[:, k], 0.99, 0).tolist() + [1.0] for k in range(W)])
    print(f"[anchor] {name}: {len(eps)} episodes, {len(train_idx)} train, {len(ch)} training windows")
    print(f"[anchor] q01 {np.round(st['q01'][:6], 2)}\n[anchor] q99 {np.round(st['q99'][:6], 2)}")
    print(f"[anchor] |anchored| by chunk position k (deg, mean over joints): "
          + " ".join(f"{np.abs(ch[:, k]).mean():.2f}" for k in range(W)))

    class _AncBase(tfds.core.GeneratorBasedBuilder, skip_registration=True):
        VERSION = tfds.core.Version("1.0.0")
        def _info(self):
            F = tfds.features
            return self.dataset_info_from_configs(features=F.FeaturesDict({
                "steps": F.Dataset({
                    "observation": F.FeaturesDict({
                        "image": F.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "wrist_image": F.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "state": F.Tensor(shape=(14,), dtype=np.float32),
                        "joint_state": F.Tensor(shape=(7,), dtype=np.float32)}),
                    "action": F.Tensor(shape=(13,), dtype=np.float32),
                    "discount": F.Scalar(dtype=np.float32), "reward": F.Scalar(dtype=np.float32),
                    "is_first": F.Scalar(dtype=np.bool_), "is_last": F.Scalar(dtype=np.bool_),
                    "is_terminal": F.Scalar(dtype=np.bool_), "language_instruction": F.Text()}),
                "episode_metadata": F.FeaturesDict({"file_path": F.Text(), "episode_index": F.Scalar(dtype=np.int64)})}))
        def _split_generators(self, dl_manager):
            return {"train": self._generate_examples()}
        def _generate_examples(self):
            for e in eps:
                n = len(e["action"]); steps = []
                for i in range(n):
                    steps.append({"observation": {"image": e["img"][i], "wrist_image": e["wrist"][i],
                                                  "state": e["state"][i], "joint_state": e["jstate"][i]},
                                  "action": e["action"][i], "discount": 1.0, "reward": float(i == n - 1),
                                  "is_first": i == 0, "is_last": i == n - 1, "is_terminal": i == n - 1,
                                  "language_instruction": e["instr"]})
                yield f"ep{e['idx']}", {"steps": steps, "episode_metadata": {"file_path": e["path"], "episode_index": e["idx"]}}

    Builder = type("".join(w.capitalize() for w in name.split("_")), (_AncBase,), {})
    assert Builder.name == name, (Builder.name, name)
    b = Builder(data_dir=a.out); b.download_and_prepare()
    # same keys, same count -> TFDS puts the same episodes in train[95%:]; check it, the comparison depends on it
    ho = lambda bb: [int(e["episode_metadata"]["episode_index"].numpy()) for e in bb.as_dataset(split="train[95%:]")]
    assert ho(b) == ho(src), (ho(b), ho(src))
    sp = os.path.join(a.out, "anchor_stats.json")
    allst = json.load(open(sp)) if os.path.exists(sp) else {}
    allst[name] = st; json.dump(allst, open(sp, "w"), indent=1)
    print(f"[anchor] wrote {b.data_dir}; held-out episodes {ho(b)} (same as {a.dataset}); stats -> {sp}")


if __name__ == "__main__":
    main()
