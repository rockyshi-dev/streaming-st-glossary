#!/usr/bin/env bash
# Fetch FLEURS en_us dev+test (audio + transcripts), the cmn_hans_cn transcripts, and the FLoRes-200 tarball (whose
# metadata_devtest.tsv gives each sentence's source article) into DIR (default: corpus/fleurs).
# Files of 8 MB or more are fetched in 8 parallel byte ranges, one resume loop per part with a fresh curl each pass;
# then a size check, a sha256 check against the values below, an atomic rename and DIR/SOURCE.json. For the two LFS
# files the expected sha256 also equals the hub's LFS oid.
# Then the audio tarballs and the FLoRes tarball are extracted in place, and the directory is made read-only.
# Usage: bash fleurs/fetch_fleurs.sh [DIR]
#   HF_ENDPOINT (default https://huggingface.co) may point at a mirror that serves the same resolve/ paths.
set -uo pipefail
DIR=$(realpath -m "${1:-corpus/fleurs}")
REV=70bb2e84b976b7e960aa89f1c648e09c59f894dd
HF=${HF_ENDPOINT:-https://huggingface.co}/datasets/google/fleurs/resolve/$REV/data
FLORES=https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
N=8
# local path | URL | bytes | expected sha256 | hub LFS oid (- for non-LFS)
FILES=(
"en_us/dev.tsv|$HF/en_us/dev.tsv|213065|9d57ee7e91e9d4c92edb39f6bbea668ef8dc2a3ff96eb510d5580b2ad05d17ec|-"
"en_us/test.tsv|$HF/en_us/test.tsv|367864|74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca|-"
"cmn_hans_cn/dev.tsv|$HF/cmn_hans_cn/dev.tsv|205248|6b4efd804b543048feb278db06f3b58b5ea171cdd4ba072e328ad630ca25384b|-"
"cmn_hans_cn/test.tsv|$HF/cmn_hans_cn/test.tsv|491487|5734461648f816181d7dab5fc79204b18c4b9bc2cd5138225b25c72d18385d21|-"
"en_us/audio/dev.tar.gz|$HF/en_us/audio/dev.tar.gz|171250900|2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c|2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c"
"en_us/audio/test.tar.gz|$HF/en_us/audio/test.tar.gz|289851356|d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356|d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356"
"flores200/flores200_dataset.tar.gz|$FLORES|25585843|b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6|-"
)
mkdir -p "$DIR/.parts" && cd "$DIR" || exit 1
exec 9>"$DIR/.fetch.lock"; flock -n 9 || { echo "another fetch holds the lock"; exit 0; }

fetch_part() {  # URL SIZE CHUNK i TAG
  local url=$1 size=$2 chunk=$3 i=$4 tag=$5 start end want f have code
  start=$(( i * chunk )); end=$(( start + chunk - 1 )); [ "$end" -ge "$size" ] && end=$(( size - 1 ))
  want=$(( end - start + 1 )); f=.parts/$tag.part$i
  [ "$want" -le 0 ] && { : > "$f"; return 0; }
  for try in $(seq 1 400); do
    have=0; [ -f "$f" ] && have=$(stat -c %s "$f")
    [ "$have" -eq "$want" ] && return 0
    [ "$have" -gt "$want" ] && { echo "$tag part$i overlong, restarting"; rm -f "$f"; have=0; }
    code=$(curl -s -L --connect-timeout 20 --speed-limit 2048 --speed-time 60 \
                -r $(( start + have ))-"$end" -o "$f.chunk" -w '%{http_code}' "$url")
    if [ "$code" = "206" ] && [ -s "$f.chunk" ]; then cat "$f.chunk" >> "$f"; fi
    rm -f "$f.chunk"; sleep 2
  done
  echo "$tag part$i gave up"; return 1
}

fetch_whole() {  # URL PATH.tmp: small files, fetched whole (a mirror may serve non-LFS files without byte ranges)
  local url=$1 f=$2 code
  for try in $(seq 1 50); do
    code=$(curl -s -L --connect-timeout 20 --speed-limit 2048 --speed-time 60 -o "$f" -w '%{http_code}' "$url")
    [ "$code" = "200" ] && return 0
    rm -f "$f"; sleep 2
  done
  echo "$f gave up"; return 1
}

JSON=()
for spec in "${FILES[@]}"; do
  IFS='|' read -r path url size want_sha lfs <<< "$spec"
  tag=$(echo "$path" | tr / _)
  if [ -f "$path" ] && [ "$(stat -c %s "$path")" = "$size" ]; then
    echo "$path already present"; sum=$(sha256sum "$path" | cut -d' ' -f1)
  else
    mkdir -p "$(dirname "$path")"
    if [ "$size" -lt 8388608 ]; then
      fetch_whole "$url" "$path.tmp" || { echo "FAILED: $path"; exit 1; }
    else
      chunk=$(( (size + N - 1) / N )); pids=()
      for i in $(seq 0 $(( N - 1 ))); do fetch_part "$url" "$size" "$chunk" "$i" "$tag" & pids+=($!); done
      ok=1; for p in "${pids[@]}"; do wait "$p" || ok=0; done
      [ "$ok" = 1 ] || { echo "FAILED: $path a part did not complete"; exit 1; }
      : > "$path.tmp"; for i in $(seq 0 $(( N - 1 ))); do cat ".parts/$tag.part$i" >> "$path.tmp"; done
    fi
    got=$(stat -c %s "$path.tmp"); [ "$got" = "$size" ] || { echo "FAILED: $path size $got != $size"; exit 1; }
    sum=$(sha256sum "$path.tmp" | cut -d' ' -f1)
    [ "$sum" = "$want_sha" ] || { echo "FAILED: $path sha256 $sum != expected $want_sha"; exit 1; }
    [ "$lfs" = "-" ] || [ "$sum" = "$lfs" ] || { echo "FAILED: $path sha256 $sum != LFS oid $lfs"; exit 1; }
    mv "$path.tmp" "$path" && rm -f .parts/"$tag".part*
  fi
  [ "$sum" = "$want_sha" ] || { echo "FAILED: $path on disk sha256 $sum != expected $want_sha"; exit 1; }
  echo "accepted $path $sum"
  JSON+=("{\"path\": \"$path\", \"url\": \"$url\", \"bytes\": $size, \"sha256\": \"$sum\", \"sha256_expected\": \"$want_sha\", \"hub_lfs_oid\": \"$lfs\"}")
done
rmdir .parts 2>/dev/null

# extract (idempotent): en_us/audio/{dev,test}/*.wav and flores200/flores200_dataset/
for s in dev test; do
  [ -d en_us/audio/$s ] || { mkdir -p en_us/audio/.x-$s && tar xzf en_us/audio/$s.tar.gz -C en_us/audio/.x-$s \
    && mv en_us/audio/.x-$s/$s en_us/audio/$s && rmdir en_us/audio/.x-$s || { echo "FAILED: extract $s"; exit 1; }; }
  echo "en_us/audio/$s: $(ls en_us/audio/$s | wc -l) files"
done
[ -d flores200/flores200_dataset ] || { tar xzf flores200/flores200_dataset.tar.gz -C flores200 || { echo "FAILED: extract flores"; exit 1; }; }

{ echo "{\"repo\": \"google/fleurs\", \"revision\": \"$REV\", \"via\": \"${HF_ENDPOINT:-https://huggingface.co}\", \"license\": \"CC BY 4.0\","
  echo " \"flores\": {\"url\": \"$FLORES\", \"license\": \"CC BY-SA 4.0\", \"note\": \"FLoRes-200; metadata_devtest.tsv gives each line's article URL\"},"
  echo " \"accepted_by\": \"sha256 == the expected value in fetch_fleurs.sh (== hub LFS oid for LFS files)\","
  echo " \"fetched\": \"$(date -u +%FT%TZ)\", \"files\": ["
  for i in "${!JSON[@]}"; do sep=,; [ "$i" -eq $(( ${#JSON[@]} - 1 )) ] && sep=; echo "  ${JSON[$i]}$sep"; done
  echo "]}"; } > SOURCE.json
python3 -c "import json; json.load(open('SOURCE.json'))" || { echo "FAILED: SOURCE.json not valid"; exit 1; }
chmod -R a-w en_us cmn_hans_cn flores200 SOURCE.json
echo "DONE $(date -Is)"
