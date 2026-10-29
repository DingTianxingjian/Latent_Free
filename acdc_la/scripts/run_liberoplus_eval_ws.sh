#!/bin/bash
# LIBERO-Plus on the UCF workstation (2026-10-29): run_liberoplus_eval_portable.sh plus the two things this host needs
# and Newton does not.  There is no sudo here, so ImageMagick (which LIBERO-plus needs: env_wrapper.py imports Wand at
# module level for the Sensor Noise motion blur) comes from a conda prefix, and its libraries have to win the soname
# lookup against the ones the python wheels bundle:
#
#   MAGICK_HOME  the conda prefix       conda create -p $MAGICK -c conda-forge imagemagick   (ImageMagick 7, apt has 6;
#                                       the blur angle is random per call, so the major version is not a result change)
#   LD_PRELOAD   glib + libjpeg         cv2 drags in the system glib 2.80 while conda's libgobject needs 2.90
#                                       ("undefined symbol: g_string_copy"), and torchvision's bundled libjpeg.so.8
#                                       has no 12-bit symbols while conda's libtiff wants them ("jpeg12_write_raw_data")
#
#   RUN=/data/ti874297/liberoplus/<post-trained run dir> SUITE=libero_10 NOTE=p2ours_full \
#     bash acdc_la/scripts/run_liberoplus_eval_ws.sh
#
# Defaults: all 8 shards over both GPUs, i.e. the same strided shards Newton ran for the other three suites.  Rerun the
# same command after an interruption (finished shards are skipped, unfinished ones resume task by task).
set -u -o pipefail
cd "$(dirname "$0")/../.."
M=${MAGICK:-/data/ti874297/liberoplus/magick}
export MAGICK_HOME=$M
export LD_PRELOAD="$M/lib/libglib-2.0.so.0 $M/lib/libjpeg.so.8"
PY=${PY:-$PWD/envs/univla_vla/bin/python} \
LIBERO_PLUS=${LIBERO_PLUS:-/data/ti874297/liberoplus/LIBERO-plus} \
RUN=${RUN:?set RUN=<post-trained run dir>} SUITE=${SUITE:-libero_10} NOTE=${NOTE:?set NOTE} \
NSHARD=${NSHARD:-8} SHARDS="${SHARDS:-0 1 2 3 4 5 6 7}" GPUS="${GPUS:-0 1}" \
  bash acdc_la/scripts/run_liberoplus_eval_portable.sh
