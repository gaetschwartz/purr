#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ] || [ ! -d "$1" ]; then
  echo "Usage: drop-ggml-build.sh <cargo target directory>" >&2
  exit 2
fi
target="$1"

# a fresh .fingerprint makes cargo replay build/<package>-<hash>/output without running the script,
# so both directories go together
count=0
total=0
while IFS= read -r -d '' directory; do
  kib="$(du -sk "$directory" | cut -f1)"
  echo "removing $directory ($((kib / 1024)) MiB)"
  rm -rf "$directory"
  count=$((count + 1))
  total=$((total + kib))
done < <(
  find "$target" -mindepth 3 -maxdepth 5 -type d \
    \( -path '*/build/whisper-rs-sys-*' -o -path '*/.fingerprint/whisper-rs-sys-*' \) \
    -prune -print0
)

if [ "$count" -eq 0 ]; then
  echo "no whisper-rs-sys build or fingerprint directory under $target"
else
  echo "removed $count directories, $((total / 1024)) MiB uncompressed"
fi
