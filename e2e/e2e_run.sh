#!/bin/bash
# End-to-end comparison runs on one GPU, one after another: train each <arm>:<seed> with e2e_train.py, then evaluate
# its snapshots with the REFERENCE tree (e2e_eval.py). ref = targets/reference (tag upstream-snapshot), opt = this
# repository's tree.
#   ASSETS=<assets dir> [ITERS=20000] [SNAPS=4000,...] [RUNS=e2e/runs] e2e_run.sh <gpu index> <arm>:<seed> [...]
set -uo pipefail
GPU=$1; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
A=${ASSETS:?set ASSETS (env/ and data/lafan1/, see README)}
PY=$A/env/bin/python
NPZ=$A/data/lafan1
ITERS=${ITERS:-20000}
SNAPS=${SNAPS:-4000,8000,12000,16000,20000}
RUNS=$ROOT/${RUNS:-e2e/runs}
declare -A TREE=([ref]=$ROOT/targets/reference [opt]=$ROOT)
E=$ROOT/e2e
export CUDA_VISIBLE_DEVICES=$GPU PYTHONNOUSERSITE=1 MUJOCO_GL=egl WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH=$A/env/bin:$PATH
unset PYTHONPATH
for job in "$@"; do
  ARM=${job%:*}; SEED=${job#*:}
  OUT=$RUNS/${ARM}_s$SEED
  mkdir -p "$OUT"
  export TMPDIR=${SCRATCH:-/tmp}/faster-supermimic-$ARM-s$SEED; mkdir -p "$TMPDIR"
  export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor TRITON_CACHE_DIR=$TMPDIR/triton WARP_CACHE_PATH=$TMPDIR/warp \
    CUDA_CACHE_PATH=$TMPDIR/cuda XDG_CACHE_HOME=$TMPDIR/xdg MPLCONFIGDIR=$TMPDIR/mpl
  echo "[$(date '+%F %T')] train $ARM seed $SEED on GPU $GPU, $ITERS iterations"
  "$PY" $E/e2e_train.py --tree ${TREE[$ARM]} --npz-dir $NPZ --seed $SEED --out "$OUT" --iters $ITERS \
    --snap-its $SNAPS > "$OUT/train.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] train exit $rc: $(tail -n 1 "$OUT/train.log")"
  if [ $rc != 0 ]; then echo "[$(date '+%F %T')] FAILED $ARM s$SEED"; continue; fi
  for it in ${SNAPS//,/ }; do
    echo "[$(date '+%F %T')] eval $OUT/snap/it$it with the reference tree"
    "$PY" $E/e2e_eval.py --tree ${TREE[ref]} --npz-dir $NPZ --ckpt-dir "$OUT/snap/it$it" \
      --out "$OUT/benchmark_it$it.json" > "$OUT/eval_it$it.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] eval exit $rc: $(grep -E 'clips \| failed' "$OUT/eval_it$it.log" || tail -n 1 "$OUT/eval_it$it.log")"
  done
  echo "[$(date '+%F %T')] DONE $ARM s$SEED"
done
