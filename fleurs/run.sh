#!/usr/bin/env bash
# FLEURS example search (README.md, section "FLEURS example"). Run from the repository root.
# Usage: bash fleurs/run.sh STAGE OUT [GPU]
#   prep   CPU  FLEURS en_us dev+test joined to FLoRes-200 devtest articles -> OUT/prep/
#   terms  GPU  glossary terms per line, Qwen2.5-7B-Instruct on vLLM, greedy, raw outputs -> OUT/terms_raw.jsonl
#   items  CPU  terms, glossary blocks, items and the three arms -> OUT/items/ (sentences.jsonl in the schema of data/)
#   align  GPU  word times of the gold transcript, faster-whisper-large-v3, decision sentences -> OUT/align.jsonl
#   stream GPU  Qwen2.5-Omni-7B, LA-2 on 1.0 s prefixes, instructed prompt, arms correct/irrelevant/wrong
#               (stglossary.stream through fleurs/stream3.py) -> OUT/stream.jsonl (resumable)
#   scan   CPU  the pre-registered criterion -> OUT/scan.json
#   all    every stage in order; a stage whose output exists is skipped (stream resumes)
# Environment:
#   FLEURS_CORPUS  the directory fleurs/fetch_fleurs.sh filled (default corpus/fleurs)
#   MODELS         directory holding Qwen2.5-7B-Instruct, faster-whisper-large-v3 and Qwen2.5-Omni-7B (default models)
#   PYTHON         interpreter for prep, items, stream and scan (default python3; stream needs torch + transformers)
#   PYTHON_VLLM    interpreter with vLLM, for terms (default $PYTHON)
#   PYTHON_WHISPER interpreter with faster-whisper, for align (default $PYTHON)
#   MIN_DUR        shortest recording used, in seconds (default 8.0)
# GPU stages take the GPU from the 3rd argument or CUDA_VISIBLE_DEVICES. A checkout with modified tracked files is
# refused, because every decoded row records the commit. OUT/STAGE.commit is appended to on every start.
set -euo pipefail
STAGE=$1 OUT=$(realpath -m "$2") GPU=${3:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORPUS=$(realpath -m "${FLEURS_CORPUS:-$ROOT/corpus/fleurs}")
MODELS=$(realpath -m "${MODELS:-$ROOT/models}")
PY=${PYTHON:-python3} PY_VLLM=${PYTHON_VLLM:-${PYTHON:-python3}} PY_WHISPER=${PYTHON_WHISPER:-${PYTHON:-python3}}
MIN_DUR=${MIN_DUR:-8.0}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
model() { [ -d "$MODELS/$1" ] || { echo "no model directory $MODELS/$1 (see README.md, FLEURS example)" >&2; exit 1; }
  echo "$MODELS/$1"; }
[ -n "$GPU" ] && export CUDA_VISIBLE_DEVICES=$GPU
gpu() { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "stage $1 needs a GPU (3rd argument or CUDA_VISIBLE_DEVICES)" >&2; exit 2; }; }
[ -f "$CORPUS/SOURCE.json" ] || { echo "no corpus at $CORPUS (fleurs/fetch_fleurs.sh)" >&2; exit 2; }
mkdir -p "$OUT"
DIRTY=$(git -C "$ROOT" status --porcelain --untracked-files=no | wc -l)
{ echo "== start $STAGE $(date -Is)"; git -C "$ROOT" rev-parse HEAD; echo "dirty_files=$DIRTY"
  echo "gpus=${CUDA_VISIBLE_DEVICES:-unset} min_dur=$MIN_DUR corpus=$CORPUS"; } >> "$OUT/$STAGE.commit"
[ "$DIRTY" -eq 0 ] || { echo "the checkout has $DIRTY modified tracked files; a run starts only from a clean commit" >&2; exit 2; }
L=$OUT/prep/lines.jsonl T=$OUT/terms_raw.jsonl S=$OUT/items/sentences.jsonl A=$OUT/align.jsonl
need() { [ -e "$1" ] || { echo "missing $1: run stage $2 first" >&2; exit 2; }; }
cd "$ROOT"
prep() { [ -e "$L" ] && { echo "$L exists; prep skipped"; return; }
  "$PY" -m fleurs.prep --corpus "$CORPUS" --out "$OUT/prep"; }
terms() { [ -e "$T" ] && { echo "$T exists; terms skipped"; return; }; need "$L" prep; gpu terms
  local M; M=$(model Qwen2.5-7B-Instruct); echo "terms model $M" >> "$OUT/$STAGE.commit"
  "$PY_VLLM" -m fleurs.terms --lines "$L" --model "$M" --out "$T"; }
items() { [ -e "$S" ] && { echo "$S exists; items skipped"; return; }; need "$T" terms
  "$PY" -m fleurs.items --lines "$L" --terms "$T" --out "$OUT/items" --min-dur "$MIN_DUR"; }
align() { [ -e "$A" ] && { echo "$A exists; align skipped"; return; }; need "$S" items; gpu align
  local W; W=$(model faster-whisper-large-v3); echo "align model $W" >> "$OUT/$STAGE.commit"
  "$PY_WHISPER" -m stglossary.align --sentences "$S" --corpus "$CORPUS" --subset first --model "$W" --out "$A"; }
stream() { need "$A" align; gpu stream
  local O7; O7=$(model Qwen2.5-Omni-7B); echo "stream model $O7; items $(sha256sum "$S")" >> "$OUT/$STAGE.commit"
  "$PY" -m fleurs.stream3 decode --mode speech --family omni25 --model "$O7" \
      --sentences "$S" --corpus "$CORPUS" --align "$A" --chunk 1.0 --prompt instructed --subset first \
      --out "$OUT/stream.jsonl"; }
scan() { need "$OUT/stream.jsonl" stream
  "$PY" -m fleurs.scan --stream "$OUT/stream.jsonl" --align "$A" --sentences "$S" --out "$OUT/scan.json"; }
case $STAGE in
  prep|terms|items|align|stream|scan) $STAGE ;;
  all) prep; terms; items; align; stream; scan ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "stage $STAGE finished $(date -Is)"
