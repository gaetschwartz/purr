#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
target="$work/target"
failures=0

make() { mkdir -p "$(dirname "$1")" && echo data > "$1"; }

for profile in release r x86_64-unknown-linux-gnu/release; do
  make "$target/$profile/build/whisper-rs-sys-aaaa/out/build/libggml.a"
  make "$target/$profile/build/whisper-rs-sys-aaaa/output"
  make "$target/$profile/build/whisper-rs-sys-bbbb/build_script_build"
  make "$target/$profile/.fingerprint/whisper-rs-sys-aaaa/run-build-script-build-script-build"
  make "$target/$profile/.fingerprint/whisper-rs-sys-bbbb/build-script-build-script-build"
  make "$target/$profile/.fingerprint/whisper-rs-sys-cccc/lib-whisper_rs_sys"
  make "$target/$profile/build/whisper-rs-0000/output"
  make "$target/$profile/.fingerprint/whisper-rs-0000/lib-whisper_rs"
  make "$target/$profile/build/ring-1111/output"
  make "$target/$profile/deps/libwhisper_rs_sys-cccc.rlib"
done
make "$target/release/build/whisper-rs-sys-dddd/out/whisper-rs-sys-eeee/file"

check() {
  if [ "$2" = "$3" ]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; failures=$((failures + 1)); fi
}

output="$("$here/drop-ggml-build.sh" "$target")"
echo "$output" | sed 's/^/     /'

left="$(cd "$target" && find . -type d \( -name 'whisper-rs-sys-*' \) | wc -l | tr -d ' ')"
check "every whisper-rs-sys build and fingerprint directory is gone" "$left" "0"

kept="$(cd "$target" && find . -type f | grep -c -E 'whisper-rs-0000|ring-1111|libwhisper_rs_sys-cccc' | tr -d ' ')"
check "other crates and the compiled rlibs stay" "$kept" "12"

case "$output" in *"removed 16 directories"*) echo "ok   the summary counts the directories" ;; *) echo "FAIL summary: $output"; failures=$((failures + 1)) ;; esac

output="$("$here/drop-ggml-build.sh" "$target")"
case "$output" in *"no whisper-rs-sys"*) echo "ok   a second run finds nothing" ;; *) echo "FAIL second run: $output"; failures=$((failures + 1)) ;; esac

status=0
"$here/drop-ggml-build.sh" "$work/missing" > /dev/null 2>&1 || status=$?
check "a missing target directory is an error" "$status" "2"

if [ "$failures" -ne 0 ]; then
  echo "$failures check(s) failed"
  exit 1
fi
echo "all checks passed"
