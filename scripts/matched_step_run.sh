#!/bin/bash
# The inference side of the matched-step experiment (prereg/matched-step.md).
#
# Transcribed from the script that actually ran on an RTX 4090 pod on 2026-11-02,
# including the one fix applied to it mid-session. It is kept here because
# prereg/matched-step-result.md claims the run is reproducible, and the pod it ran
# on has since been terminated. Paths are the pod's; adjust W, and the three
# scripts it calls are scripts/padview.py, scripts/matched_step_score.py and
# src/inkrepro/metrics.py.
#
# Why a padded view: infer.json records that the atlas inferred on a zero-copy
# padded view of each render (1200x6540 -> 1216x6592, padview.py). Skipping that
# produces maps at the unpadded size, which cannot be paired against the published
# ones without putting a pipeline artefact into the result.
#
# Why these exact flags: they are the atlas's own recipe, from infer.json's
# "recipe" field. --num-workers 2 and the pinned BLAS threads matter because
# OpenBLAS otherwise spawns 64 threads per DataLoader worker and the workers die.
set -uo pipefail
export PATH=/root/.local/bin:$PATH
export VIRTUAL_ENV=/workspace/v314
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/workspace/v314/bin/python
R=https://huggingface.co/datasets/rodriguescarson/eligible-scroll-atlas-renders/resolve/main
OUT=/workspace/results.csv
LOG=/workspace/batch.log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $LOG; }

log "BATCH START 40 surfaces"
tail -n +2 /workspace/matched_step_sample.csv | while IFS=, read -r SCROLL MESH COV IOU; do
  [ -z "$MESH" ] && continue
  if [ -f $OUT ] && grep -q "^$SCROLL,$MESH," $OUT; then log "SKIP $SCROLL/$MESH (already scored)"; continue; fi
  T0=$SECONDS
  W=/workspace/b/$SCROLL/$MESH; rm -rf $W; mkdir -p $W
  if ! curl -sS -L -f -o $W/sv.tar "$R/$SCROLL/$MESH/surface-volumes.tar"; then log "FAIL download $SCROLL/$MESH"; rm -rf $W; continue; fi
  tar xf $W/sv.tar -C $W && rm -f $W/sv.tar
  SRC=$(ls -d $W/surface-volumes/*.zarr 2>/dev/null | head -1)
  [ -z "$SRC" ] && { log "FAIL no zarr $SCROLL/$MESH"; rm -rf $W; continue; }
  # The published maps are named with the render's volume id, which only the zarr knows.
  VID=$(basename "$SRC" | sed "s/.*volume-\([0-9]*\)\.zarr/\1/")
  F=$W/infolder; mkdir -p $F/$MESH
  $PY /workspace/padview.py "$SRC" $F/$MESH/$MESH.zarr >> $LOG 2>&1 || { log "FAIL padview $SCROLL/$MESH"; rm -rf $W; continue; }
  cd /workspace/villa/vesuvius
  $PY -m vesuvius.ink_detection.inference.infer --folder $F \
    --checkpoint-path /workspace/ck/s43_020k.pth --direction both --overlap 0.5 \
    --blend-mode hann --batch-size 16 --num-workers 2 --no-compile \
    --output-prefix atlas > $W/infer.out 2>&1 || { log "FAIL infer $SCROLL/$MESH"; tail -3 $W/infer.out >> $LOG; rm -rf $W; continue; }
  FWD=$(find $F/$MESH/preds -name "*_forward_*.tif" | head -1)
  REV=$(find $F/$MESH/preds -name "*_reverse_*.tif" | head -1)
  [ -z "$FWD" ] || [ -z "$REV" ] && { log "FAIL no preds $SCROLL/$MESH"; rm -rf $W; continue; }
  mkdir -p /workspace/maps/$SCROLL/$MESH
  cp "$FWD" /workspace/maps/$SCROLL/$MESH/s43_020k.tif
  cp "$REV" /workspace/maps/$SCROLL/$MESH/s43_020k_reverse.tif
  S=/workspace/maps/$SCROLL/$MESH
  P=$R/$SCROLL/$MESH/ink-detection/$SCROLL-$MESH-$VID-ink9um
  ok=1
  curl -sS -L -f -o $S/s42_020k.tif         "$P-s42_020k.tif"         || ok=0
  curl -sS -L -f -o $S/s42_020k_reverse.tif "$P-s42_020k_reverse.tif" || ok=0
  curl -sS -L -f -o $S/s43_060k.tif         "$P-s43_060k.tif"         || ok=0
  curl -sS -L -f -o $S/s43_060k_reverse.tif "$P-s43_060k_reverse.tif" || ok=0
  curl -sS -L -f -o $S/valid_mask.tif       "$R/$SCROLL/$MESH/ink-detection/valid_mask.tif" || ok=0
  [ $ok -eq 0 ] && { log "FAIL published maps $SCROLL/$MESH"; rm -rf $W; continue; }
  $PY /workspace/score.py $S $SCROLL $MESH $OUT >> $LOG 2>&1 || { log "FAIL score $SCROLL/$MESH"; rm -rf $W; continue; }
  rm -rf $W
  log "OK $SCROLL/$MESH  $((SECONDS-T0))s  rows=$(grep -c "^$SCROLL,$MESH," $OUT)"
done
log "BATCH DONE surfaces_scored=$(tail -n +2 $OUT 2>/dev/null | cut -d, -f1,2 | sort -u | wc -l) rows=$(tail -n +2 $OUT 2>/dev/null | wc -l)"
