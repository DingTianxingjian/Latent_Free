"""Log a real-robot run so it can be analysed offline (2026-11-12).

Three lines in the deployment loop:

    from acdc_la.realrobot.deploy_logger import DeployLogger
    log = DeployLogger("robot_logs", checkpoint=CKPT, instruction="press the red button", chunk_exec=1)
    ...
    t0 = time.time(); chunk = p.act_chunk(frame, instr); t_inf = time.time() - t0
    for k in range(n):
        q = read_joint_state_deg()                       # the 6 measured joints (+ gripper if you have it)
        target = p.to_joint_targets(chunk[k], q)         # (or p.chunk_targets for anchored checkpoints)
        log.step(frame=frame, q=q, chunk=chunk, k=k, target=target, t_infer=t_inf if k == 0 else None)
        send_joint_targets(target); ...
    ...
    log.end(success=True, note="touched the red button, a bit to the LEFT of centre")   # your own judgement

Each episode is written to robot_logs/<timestamp>_<instruction>/: steps.npz (time, q, target, chunk, k, t_infer),
meta.json (checkpoint, instruction, chunk_exec, success, note) and frames/NNNN.jpg (the RAW camera frame, as given
to act_chunk).  Send the whole robot_logs/ directory, e.g.
    hf upload <you>/terra-robot-logs robot_logs --repo-type dataset --private
What gets checked from it: camera shift vs the training images (a fixed offset direction), the arm's tracking lag vs
the demonstrations, the real loop rate and inference latency, and where along the episode the error starts.
"""
import json, os, time
import numpy as np


class DeployLogger:
    def __init__(self, root, checkpoint="", instruction="", chunk_exec=1, save_every=1, **extra):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, f"{stamp}_{instruction.replace(' ', '_')[:40]}")
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        self.meta = dict(checkpoint=str(checkpoint), instruction=instruction, chunk_exec=int(chunk_exec),
                         start=time.time(), **extra)
        self.rows, self.n, self.save_every = [], 0, max(1, int(save_every))

    def step(self, frame, q, chunk=None, k=0, target=None, t_infer=None):
        """One control tick.  frame: HxWx3 RGB uint8 (what act_chunk got); q: measured joints (deg);
        chunk: the policy's (W, 7) chunk this tick executes from; k: index into it; target: what was SENT."""
        if self.n % self.save_every == 0 and frame is not None:
            from PIL import Image
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(os.path.join(self.dir, "frames", f"{self.n:04d}.jpg"), quality=95)
        self.rows.append(dict(t=time.time(), q=np.asarray(q, np.float64).ravel(),
                              target=None if target is None else np.asarray(target, np.float64).ravel(),
                              chunk=None if chunk is None else np.asarray(chunk, np.float64), k=int(k),
                              t_infer=np.nan if t_infer is None else float(t_infer)))
        self.n += 1

    def end(self, success=None, note=""):
        """Call once per episode with YOUR judgement: success (pressed the right button?) and a free-text note,
        e.g. which way it was off-centre."""
        R = self.rows
        pad = lambda xs, d: np.stack([x if x is not None else np.full(d, np.nan) for x in xs]) if xs else np.zeros((0, d))
        dq = len(R[0]["q"]) if R else 6
        dt = len(next((r["target"] for r in R if r["target"] is not None), np.zeros(6)))
        ch = next((r["chunk"] for r in R if r["chunk"] is not None), None)
        np.savez_compressed(os.path.join(self.dir, "steps.npz"),
                            t=np.array([r["t"] for r in R]), q=pad([r["q"] for r in R], dq),
                            target=pad([r["target"] for r in R], dt), k=np.array([r["k"] for r in R]),
                            t_infer=np.array([r["t_infer"] for r in R]),
                            chunk=pad([r["chunk"] for r in R], ch.shape) if ch is not None else np.zeros(0))
        self.meta.update(success=success, note=note, steps=len(R), end=time.time())
        json.dump(self.meta, open(os.path.join(self.dir, "meta.json"), "w"), indent=1)
        print(f"[deploy-log] {len(R)} steps -> {self.dir}")
        return self.dir
