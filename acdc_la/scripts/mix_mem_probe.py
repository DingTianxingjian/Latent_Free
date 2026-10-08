"""How much host RAM does ONE rank's acdc_oxe29 pipeline need, as a function of the shuffle buffer and the
trajectory thread counts?  Data only: no model, no GPU.  Both DDP ranks pay this, on top of their 7B model, on a
62 GB host -- the 2026-10-08 smoke was SIGKILLed 80 s after the (cached) statistics step.

    python acdc_la/scripts/mix_mem_probe.py <shuffle_buffer> <threads> <n_samples>
"""
import os, resource, sys, threading, time

import prismatic.vla.datasets.datasets as DS
from acdc_la.paths import LF, IMG

SHUF, THREADS, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
_real = DS.make_interleaved_dataset
def patched(**kw):
    kw["shuffle_buffer_size"] = SHUF
    kw["traj_transform_threads"] = THREADS
    kw["traj_read_threads"] = THREADS
    return _real(**kw)
DS.make_interleaved_dataset = patched

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0      # GiB, high-water mark

def avail():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1048576.0


FLOOR = float(os.environ.get("PROBE_ABORT_AVAIL_GB", "8"))   # this host is shared: never drive it to the OOM killer


def watchdog():
    """The shuffle buffer fills BEFORE the first element is yielded, so the in-loop check cannot see that fill:
    at shuf=16000 the host went from 54 GB to 1 GB available in two minutes with nothing yielded yet."""
    while True:
        a = avail()
        if a < FLOOR:
            print(f"[probe] WATCHDOG abort: avail={a:.1f}G < floor {FLOOR}G, peak_rss={rss():.1f}G", flush=True)
            os._exit(3)
        time.sleep(2)


threading.Thread(target=watchdog, daemon=True).start()

ds = DS.RLDSDataset(os.path.join(LF, "data", "oxe_raw"), "acdc_oxe29", lambda b: b,
                    resize_resolution=(IMG, IMG), shuffle_buffer_size=SHUF, image_aug=True,
                    window_size=14, training_phase="post-training")
print(f"[probe] built shuf={SHUF} threads={THREADS} rss={rss():.1f}G avail={avail():.1f}G", flush=True)
t0 = time.time()
for i, b in enumerate(ds):
    if (i + 1) % 200 == 0:
        print(f"[probe] shuf={SHUF} threads={THREADS} n={i+1} rss_peak={rss():.1f}G avail={avail():.1f}G "
              f"{(i+1)/(time.time()-t0):.0f} samp/s", flush=True)
    if avail() < FLOOR:
        print(f"[probe] ABORT shuf={SHUF} threads={THREADS} n={i+1} rss_peak={rss():.1f}G avail={avail():.1f}G "
              f"< floor {FLOOR}G", flush=True)
        break
    if i + 1 >= N:
        break
print(f"[probe] DONE shuf={SHUF} threads={THREADS} peak_rss={rss():.1f}G avail={avail():.1f}G", flush=True)
