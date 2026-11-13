#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: prune-rust-caches.sh [--dry-run] superseded
       prune-rust-caches.sh [--dry-run] own-family
       prune-rust-caches.sh [--dry-run] compiler-cache

Delete the Swatinem/rust-cache and compiler cache (ccache) entries of the develop branch. A Rust
family is the entries whose keys only differ by the trailing environment and lockfile hashes. A
compiler cache family is the entries of one prefix, ccache-<os>-<version>-, which only differ by
the trailing run id and run attempt.

superseded      keep the newest entry of every family, Rust and compiler cache alike
own-family      delete every entry of the calling job's Rust family, to run before the job saves a
                replacement. Reads PURR_RUST_CACHE_KEY (the rust-cache `key` input),
                PURR_RUST_CACHE_SAVE, CACHE_HIT (rust-cache's cache-hit output) and GITHUB_JOB.
                Does nothing when the job does not save or got an exact hit.
compiler-cache  delete every entry of the family PURR_COMPILER_CACHE_PREFIX except the one named
                PURR_COMPILER_CACHE_KEEP_KEY, which the job has just saved. Does nothing when
                that entry is not listed.
EOF
}

dry_run=false
if [ "${1:-}" = "--dry-run" ]; then
  dry_run=true
  shift
fi
mode="${1:-}"

ref="refs/heads/develop"
prefix=""
keep=""
list_key=""
case "$mode" in
  superseded) ;;
  own-family)
    if [ "${PURR_RUST_CACHE_SAVE:-}" != "true" ]; then
      echo "this job does not save a Rust cache, so there is nothing to prune"
      exit 0
    fi
    if [ "${CACHE_HIT:-}" = "true" ]; then
      echo "the Rust cache was an exact hit and will not be saved again, so it is kept"
      exit 0
    fi
    prefix="v0-rust${PURR_RUST_CACHE_KEY:+-$PURR_RUST_CACHE_KEY}-${GITHUB_JOB:?}-"
    list_key="$prefix"
    ;;
  compiler-cache)
    prefix="${PURR_COMPILER_CACHE_PREFIX:?}"
    keep="${PURR_COMPILER_CACHE_KEEP_KEY:?}"
    list_key="$prefix"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# rust-cache keys read <prefix-key>[-<key>]-<job id>-<os>-<arch>-<environment hash>-<lockfile hash>
# compiler cache keys read ccache-<os>-<version>-<run id>-<run attempt>
filter='
def hashes: "-[0-9a-f]{8}-[0-9a-f]{8}$";
def run: "-[0-9]+-[0-9]+$";
def ours: .key | (startswith("v0-rust-") or startswith("ccache-"));
def family: .key | if startswith("v0-rust-") then sub(hashes; "") else sub(run; "") end;
map(select(.ref == $ref))
| if $mode == "superseded" then
    map(select(ours))
    | group_by(family)
    | map(sort_by([.createdAt, .id]) | reverse | .[1:][])
  elif $mode == "compiler-cache" then
    map(select(.key | startswith($prefix) and (.[($prefix | length):] | test("^[0-9]+-[0-9]+$"))))
    | map(select(.key != $keep))
  else
    map(select(.key | startswith($prefix) and (.[($prefix | length):] | test("^[^-]+-[^-]+" + hashes))))
  end
| .[]
| [.id, .key, .sizeInBytes]
| @tsv
'

caches="$(gh cache list --ref "$ref" ${list_key:+--key "$list_key"} --limit 1000 --json id,key,ref,sizeInBytes,createdAt)"

if [ "$mode" = "compiler-cache" ] \
  && ! printf '%s' "$caches" | jq -e --arg keep "$keep" --arg ref "$ref" 'any(.[]; .key == $keep and .ref == $ref)' > /dev/null; then
  echo "the new compiler cache entry $keep is not listed, so the older ones are kept"
  exit 0
fi

doomed="$(printf '%s' "$caches" | jq -r -b --arg mode "$mode" --arg prefix "$prefix" --arg keep "$keep" --arg ref "$ref" "$filter" | tr -d '\r')"

if [ -z "$doomed" ]; then
  echo "no cache entries to delete"
  exit 0
fi

status=0
while IFS=$'\t' read -r id key size; do
  echo "deleting $key ($((size / 1048576)) MiB)"
  if [ "$dry_run" = "false" ] && ! gh cache delete "$id" < /dev/null; then
    echo "::warning::could not delete the cache entry $id ($key)"
    status=1
  fi
done <<< "$doomed"
exit "$status"
