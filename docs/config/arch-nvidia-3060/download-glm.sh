#!/usr/bin/env bash
#
# Download the GLM-5.2 3-bit (UD-IQ3_XXS) GGUF shards from Hugging Face into the
# model dir this node's start-glm.sh expects. ~282 GB across 7 shards.
#
# No huggingface-cli required — uses curl (to list files) + wget -c (resumable).
# The repo is PUBLIC, so NO HF_TOKEN is needed; set HF_TOKEN only if you hit
# rate limits or the repo becomes gated.
#
#   ./download-glm.sh                 # download / resume into the default dir
#   HF_TOKEN=hf_xxx ./download-glm.sh # optional auth
#
# Safe to re-run: wget -c resumes partial files and skips complete ones.
# Run it in the background and watch with ./glm-progress.sh:
#   nohup ./download-glm.sh > /mnt/storage1/models/GLM-5.2/download.log 2>&1 &
set -euo pipefail

REPO="${REPO:-unsloth/GLM-5.2-GGUF}"
PAT="${PAT:-UD-IQ3_XXS}"
DEST="${DEST:-/mnt/storage1/models/GLM-5.2}"
TOKEN="${HF_TOKEN:-}"

mkdir -p "$DEST"
CURL_AUTH=(); WGET_AUTH=()
if [ -n "$TOKEN" ]; then
  CURL_AUTH=(-H "Authorization: Bearer ${TOKEN}")
  WGET_AUTH=(--header "Authorization: Bearer ${TOKEN}")
fi

echo "Listing ${PAT} shards in ${REPO} …"
# Pull the recursive file tree and emit "path<TAB>size" for matching shards.
LIST="$(curl -fsS -m 60 ${CURL_AUTH[@]+"${CURL_AUTH[@]}"} \
  "https://huggingface.co/api/models/${REPO}/tree/main?recursive=true" \
  | python3 -c '
import sys, json
data = json.load(sys.stdin)
pat = sys.argv[1]
for f in data:
    if isinstance(f, dict) and f.get("type")=="file" and pat in f.get("path",""):
        sz = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        print(f["path"] + "\t" + str(sz))
' "$PAT")"

if [ -z "$LIST" ]; then
  echo "ERROR: no files matching '${PAT}' in ${REPO} (gated repo? try HF_TOKEN)" >&2
  exit 1
fi

N="$(printf '%s\n' "$LIST" | wc -l)"
TOTAL_GB="$(printf '%s\n' "$LIST" | awk -F'\t' '{s+=$2} END{printf "%.1f", s/1e9}')"
echo "Found ${N} shard(s), ${TOTAL_GB} GB total. Destination: ${DEST}"
echo

i=0
while IFS=$'\t' read -r path size; do
  i=$((i+1))
  fname="$(basename "$path")"
  dest="${DEST}/${fname}"
  url="https://huggingface.co/${REPO}/resolve/main/${path}?download=true"
  have=0; [ -f "$dest" ] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
  if [ "$have" = "$size" ]; then
    echo "[$i/$N] $fname — already complete ($(awk "BEGIN{printf \"%.1f\", $size/1e9}") GB), skipping"
    continue
  fi
  echo "[$i/$N] $fname — $(awk "BEGIN{printf \"%.1f\", $size/1e9}") GB (have $(awk "BEGIN{printf \"%.1f\", $have/1e9}") GB), downloading…"
  wget -c ${WGET_AUTH[@]+"${WGET_AUTH[@]}"} --progress=bar:force:noscroll -O "$dest" "$url"
done <<< "$LIST"

echo
echo "Verifying sizes…"
ok=1
while IFS=$'\t' read -r path size; do
  fname="$(basename "$path")"; dest="${DEST}/${fname}"
  have=0; [ -f "$dest" ] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
  if [ "$have" = "$size" ]; then echo "  OK   $fname"; else echo "  BAD  $fname (have $have, want $size) — re-run to resume"; ok=0; fi
done <<< "$LIST"

[ "$ok" = 1 ] && echo "All shards complete. Start the server: ./start-glm.sh" || { echo "Some shards incomplete — re-run this script."; exit 1; }
