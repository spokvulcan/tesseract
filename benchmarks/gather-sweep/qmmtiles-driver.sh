#!/bin/bash
# qmmtiles ABBA driver.
# Per candidate config: 8 launches in the order
#   stock, cand, cand, stock, stock, cand, cand, stock   (= 4 ABBA pairs)
# Each launch walks all shapes (bitwise dump + timed graph), so every shape
# sees that exact interleave. After each candidate block, the candidate's
# dumps are cmp'd byte-for-byte against stock's (bitwise gate) and removed.
#
# Usage: qmmtiles-driver.sh <logfile> [cfg ...]
#   cfg format: bm,bk,bn   e.g. 64,32,64
set -u
BIN=/tmp/gather-sweep/.build/release/gather-sweep
DUMPS=/tmp/gather-sweep/qmmtiles-dumps
LOG=${1:-/tmp/gather-sweep/qmmtiles-run.log}
shift || true
CFGS=("$@")
if [ ${#CFGS[@]} -eq 0 ]; then CFGS=("64,32,64" "64,64,64" "32,32,64" "16,32,32"); fi
mkdir -p "$DUMPS"

launch() { # $1 = cfg ("stock" or "bm,bk,bn")
  local cfg="$1" rc
  if [ "$cfg" = stock ]; then
    env -u MLX_QMM_TILES "$BIN" qmmtiles >>"$LOG" 2>&1
  else
    MLX_QMM_TILES="$cfg" "$BIN" qmmtiles >>"$LOG" 2>&1
  fi
  rc=$?
  echo "QTLAUNCH cfg=$cfg rc=$rc ts=$(date +%H:%M:%S)" >>"$LOG"
  return $rc
}

for cand in "${CFGS[@]}"; do
  slug=${cand//,/x}
  echo "QTBLOCK cand=$cand start=$(date +%H:%M:%S)" >>"$LOG"
  launch stock   || true
  launch "$cand" || true
  launch "$cand" || true
  launch stock   || true
  launch stock   || true
  launch "$cand" || true
  launch "$cand" || true
  launch stock   || true
  echo "QTBLOCK cand=$cand end=$(date +%H:%M:%S)" >>"$LOG"
  # bitwise gate for this candidate (stock dumps are reference; kept)
  for f in "$DUMPS"/*__"$slug".bin; do
    [ -e "$f" ] || continue
    shape=$(basename "$f" | sed "s/__${slug}\.bin//")
    stock="$DUMPS/${shape}__stock.bin"
    if [ ! -e "$stock" ]; then
      echo "QTGATE $cand $shape NOREF" >>"$LOG"
    elif cmp -s "$stock" "$f"; then
      echo "QTGATE $cand $shape IDENT" >>"$LOG"
    else
      echo "QTGATE $cand $shape DIFF" >>"$LOG"
    fi
    rm -f "$f"
  done
done
echo "QTDONE $(date +%H:%M:%S)" >>"$LOG"
