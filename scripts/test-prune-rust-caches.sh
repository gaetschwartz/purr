#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/gh" <<'EOF'
#!/usr/bin/env bash
case "$1 $2" in
  "cache list")
    printf '%s\n' "$*" >> "$LIST_ARGS"
    [ -z "${LIST_FAIL:-}" ] || exit 1
    cat "$FIXTURE"
    ;;
  "cache delete")
    echo "$3" >> "$DELETED"
    case " ${FAIL_IDS:-} " in *" $3 "*) exit 1 ;; esac
    ;;
  *)
    echo "unexpected gh $*" >&2
    exit 99
    ;;
esac
EOF
chmod +x "$work/gh"
export PATH="$work:$PATH"
export FIXTURE="$work/caches.json" LIST_ARGS="$work/list-args" DELETED="$work/deleted"

linux_all="all-ggml-native-OFF-no-vmm-ON-cuda-13.5.0-vulkan-1.4.363.1"
windows_all="cuda-vulkan-ggml-native-OFF-crt-static-Ninja-cuda-13.5.0-vulkan-1.4.363.1"
windows_cuda="cuda-ggml-native-OFF-crt-static-Ninja-cuda-13.5.0-vulkan-1.4.363.1"
mac="metal-ggml-native-OFF-arm-armv8.2-a+dotprod+fp16"

entry() {
  printf '{"id":%s,"key":"%s","ref":"%s","sizeInBytes":%s,"createdAt":"%s"}' "$1" "$2" "${4:-refs/heads/develop}" 1073741824 "$3"
}

cat > "$FIXTURE" <<EOF
[
$(entry 1 "v0-rust-$linux_all-cli-linux-Linux-x64-2fd77104-3390f3d2" 2026-11-13T09:17:00Z),
$(entry 12 "v0-rust-$linux_all-cli-linux-Linux-x64-99999999-3390f3d2" 2026-11-13T09:30:00Z),
$(entry 2 "v0-rust-$linux_all-cli-linux-Linux-x64-2fd77104-313822ff" 2026-11-13T10:04:00Z),
$(entry 3 "v0-rust-$linux_all-desktop-linux-Linux-x64-2fd77104-3390f3d2" 2026-11-13T09:30:00Z),
$(entry 4 "v0-rust-$linux_all-desktop-linux-Linux-x64-2fd77104-313822ff" 2026-11-13T10:09:00Z),
$(entry 8 "v0-rust-$linux_all-cli-linux-Linux-x64-2fd77104-313822ff" 2026-11-13T10:30:00Z refs/heads/feature),
$(entry 9 "v0-rust-$linux_all-desktop-linux-Linux-x64-2fd77104-313822ff" 2026-11-13T10:40:00Z refs/pull/5/merge),
$(entry 5 "v0-rust-$windows_all-desktop-windows-Windows_NT-x64-b4667b02-aaaaaaaa" 2026-11-13T09:00:00Z),
$(entry 6 "v0-rust-$windows_all-cli-windows-Windows_NT-x64-b4667b02-aaaaaaaa" 2026-11-13T09:00:00Z),
$(entry 14 "v0-rust-$windows_cuda-cli-windows-Windows_NT-x64-b4667b02-aaaaaaaa" 2026-11-13T09:00:00Z),
$(entry 7 "v0-rust-web-web-Linux-x64-2fd77104-313822ff" 2026-11-13T10:02:00Z),
$(entry 13 "v0-rust-web-web-extra-Linux-x64-2fd77104-313822ff" 2026-11-13T10:02:00Z),
$(entry 15 "v0-rust-$mac-cli-macos-Darwin-arm64-60948658-313822ff" 2026-11-13T10:04:00Z),
$(entry 16 "v0-rust-$mac-cli-macos-Darwin-arm64-60948658-3390f3d2" 2026-11-13T09:24:00Z),
$(entry 17 "v0-rust-$mac-desktop-macos-Darwin-arm64-60948658-313822ff" 2026-11-13T10:05:00Z),
$(entry 18 "v0-rust-test-Linux-x64-2fd77104-313822ff" 2026-11-13T10:03:00Z),
$(entry 10 "vcpkg-ffmpeg-Linux-efecf268f3a3ec78f4bf1f0c36759bdb23c6f653" 2026-11-11T10:10:00Z),
$(entry 11 "dioxus-cli-Linux-0.8.0-alpha.3" 2026-11-13T09:24:00Z),
$(entry 20 "ccache-Linux-v1-100-1" 2026-11-13T09:00:00Z),
$(entry 21 "ccache-Linux-v1-101-1" 2026-11-13T10:00:00Z),
$(entry 22 "ccache-Linux-v1-102-1" 2026-11-13T11:00:00Z refs/heads/feature),
$(entry 23 "ccache-Linux-v2-5-1" 2026-11-13T08:00:00Z),
$(entry 24 "ccache-Windows-v1-100-1" 2026-11-13T09:00:00Z),
$(entry 25 "ccache-Windows-v1-100-2" 2026-11-13T09:30:00Z),
$(entry 26 "ccache-Linux-v1-extra-103-1" 2026-11-13T09:45:00Z)
]
EOF

failures=0

run() {
  : > "$DELETED"
  : > "$LIST_ARGS"
  status=0
  output="$("$@" 2>&1)" || status=$?
}

deleted() {
  sort < "$DELETED" | tr '\n' ' ' | sed 's/ $//'
}

check() {
  local name="$1" want="$2" want_status="${3:-0}" got
  got="$(deleted)"
  if [ "$got" = "$want" ] && [ "$status" = "$want_status" ]; then
    echo "ok   $name"
  else
    echo "FAIL $name: deleted [$got] want [$want], status $status want $want_status"
    echo "$output" | sed 's/^/     /'
    failures=$((failures + 1))
  fi
}

prune="$here/prune-rust-caches.sh"
own() { env PURR_RUST_CACHE_SAVE=true PURR_RUST_CACHE_KEY="$1" GITHUB_JOB="$2" CACHE_HIT="${3-false}" "$prune" own-family; }

run "$prune" superseded
check "superseded keeps the newest of each family and ignores other refs" "1 12 16 20 24 3"

run own "$linux_all" cli-linux
check "fallback hit deletes the whole family" "1 12 2"
grep -q -- "--ref refs/heads/develop" "$LIST_ARGS" \
  && echo "ok   the listing is limited to develop" \
  || { echo "FAIL the listing is not limited to develop"; failures=$((failures + 1)); }

run own "$linux_all" cli-linux true
check "exact hit keeps everything" ""
[ ! -s "$LIST_ARGS" ] && echo "ok   an exact hit does not even list" || { echo "FAIL exact hit listed"; failures=$((failures + 1)); }

run env PURR_RUST_CACHE_SAVE=false PURR_RUST_CACHE_KEY="$linux_all" GITHUB_JOB=cli-linux CACHE_HIT=false "$prune" own-family
check "a job that does not save deletes nothing" ""

run env PURR_RUST_CACHE_SAVE=true PURR_RUST_CACHE_KEY="$linux_all" GITHUB_JOB=cli-linux "$prune" own-family
check "no cache-hit output counts as a miss" "1 12 2"

run own "$linux_all" desktop-linux
check "same key, other job" "3 4"

run own "$windows_all" desktop-windows
check "windows desktop" "5"

run own "$windows_all" cli-windows
check "windows cli cuda-vulkan" "6"

run own "$windows_cuda" cli-windows
check "windows cli cuda only" "14"

run own "$mac" cli-macos
check "macos cli, with + and . in the key" "15 16"

run own "$mac" desktop-macos
check "macos desktop" "17"

run own web web
check "a key that is a prefix of another family" "7"

run own "" test
check "no key input" "18"

run env FAIL_IDS=12 PURR_RUST_CACHE_SAVE=true PURR_RUST_CACHE_KEY="$linux_all" GITHUB_JOB=cli-linux CACHE_HIT=false "$prune" own-family
check "a failed deletion does not stop the others and fails the script" "1 12 2" 1
case "$output" in *"::warning::could not delete the cache entry 12"*) echo "ok   the failed deletion is annotated" ;; *) echo "FAIL no warning annotation"; failures=$((failures + 1)) ;; esac

compiler() { env PURR_COMPILER_CACHE_PREFIX="$1" PURR_COMPILER_CACHE_KEEP_KEY="$2" "$prune" compiler-cache; }

run compiler "ccache-Linux-v1-" "ccache-Linux-v1-101-1"
check "compiler cache: the saved entry stays, the older ones of the family go" "20"

run compiler "ccache-Windows-v1-" "ccache-Windows-v1-100-2"
check "compiler cache: another OS" "24"

run compiler "ccache-Linux-v1-" "ccache-Linux-v1-999-1"
check "compiler cache: an unlisted saved entry deletes nothing" ""
case "$output" in *"is not listed"*) echo "ok   the missing entry is reported" ;; *) echo "FAIL no message for a missing entry"; failures=$((failures + 1)) ;; esac

run compiler "ccache-Linux-v1-" "ccache-Linux-v1-102-1"
check "compiler cache: an entry of another ref does not count as saved" ""

run env PURR_COMPILER_CACHE_PREFIX="ccache-Linux-v1-" "$prune" compiler-cache
check "compiler cache: the saved key is required" "" 1

run env LIST_FAIL=1 "$prune" superseded
check "a failed listing deletes nothing and fails" "" 1

run env PURR_RUST_CACHE_SAVE=true PURR_RUST_CACHE_KEY="$linux_all" GITHUB_JOB=cli-linux CACHE_HIT=false "$prune" --dry-run own-family
check "dry run deletes nothing" ""

echo "[]" > "$FIXTURE"
run "$prune" superseded
check "no caches" ""

run "$prune" bogus
check "unknown mode" "" 2

if [ "$failures" -ne 0 ]; then
  echo "$failures check(s) failed"
  exit 1
fi
echo "all checks passed"
