#!/usr/bin/env bash
#
# Snapshot of the GLM-5.2 download progress: per-shard and overall % of the
# expected ~282 GB. Reads expected sizes from the HF API, compares to files on
# disk. Run any time (the download itself is download-glm.sh).
#
#   ./glm-progress.sh
#   watch -n10 ./glm-progress.sh     # live refresh every 10s
set -euo pipefail

REPO="${REPO:-unsloth/GLM-5.2-GGUF}"
PAT="${PAT:-UD-IQ3_XXS}"
DEST="${DEST:-/mnt/storage1/models/GLM-5.2}"

LIST="$(curl -fsS -m 30 "https://huggingface.co/api/models/${REPO}/tree/main?recursive=true" \
  | python3 -c '
import sys, json
data = json.load(sys.stdin); pat = sys.argv[1]
for f in data:
    if isinstance(f, dict) and f.get("type")=="file" and pat in f.get("path",""):
        sz = f.get("size") or (f.get("lfs") or {}).get("size") or 0
        print(f["path"] + "\t" + str(sz))
' "$PAT")"

want_tot=0; have_tot=0; done_n=0; n=0
echo "GLM-5.2 ${PAT} — ${DEST}"
printf "%-44s %8s %8s %6s\n" "shard" "have" "want" "%"
while IFS=$'\t' read -r path size; do
  n=$((n+1)); fname="$(basename "$path")"; dest="${DEST}/${fname}"
  have=0; [ -f "$dest" ] && have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
  want_tot=$((want_tot+size)); have_tot=$((have_tot+have))
  pct="$(awk "BEGIN{printf \"%.0f\", ($size>0)?100*$have/$size:0}")"
  [ "$have" = "$size" ] && done_n=$((done_n+1))
  printf "%-44s %7.1fG %7.1fG %5s%%\n" "$fname" \
    "$(awk "BEGIN{print $have/1e9}")" "$(awk "BEGIN{print $size/1e9}")" "$pct"
done <<< "$LIST"

echo "----------------------------------------------------------------------"
printf "TOTAL: %.1f / %.1f GB  (%.1f%%)   shards complete: %d/%d\n" \
  "$(awk "BEGIN{print $have_tot/1e9}")" "$(awk "BEGIN{print $want_tot/1e9}")" \
  "$(awk "BEGIN{print ($want_tot>0)?100*$have_tot/$want_tot:0}")" "$done_n" "$n"
df -h "$DEST" | awk 'NR==2{printf "disk: %s free on %s\n", $4, $6}'
