#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: prune-rust-caches.sh [--dry-run] superseded
       prune-rust-caches.sh [--dry-run] own-family

Delete Swatinem/rust-cache entries of the develop branch. A family is the entries whose keys
only differ by the trailing environment and lockfile hashes.

superseded  keep the newest entry of every family
own-family  delete every entry of the calling job's family, to run before the job saves a
            replacement. Reads PURR_RUST_CACHE_KEY (the rust-cache `key` input),
            PURR_RUST_CACHE_SAVE, CACHE_HIT (rust-cache's cache-hit output) and GITHUB_JOB.
            Does nothing when the job does not save or got an exact hit.
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
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# rust-cache keys read <prefix-key>[-<key>]-<job id>-<os>-<arch>-<environment hash>-<lockfile hash>
filter='
def hashes: "-[0-9a-f]{8}-[0-9a-f]{8}$";
map(select(.ref == $ref))
| if $mode == "superseded" then
    group_by(.key | sub(hashes; ""))
    | map(sort_by([.createdAt, .id]) | reverse | .[1:][])
  else
    map(select(.key | startswith($prefix) and (.[($prefix | length):] | test("^[^-]+-[^-]+" + hashes))))
  end
| .[]
| [.id, .key, .sizeInBytes]
| @tsv
'

caches="$(gh cache list --ref "$ref" --key "${prefix:-v0-rust-}" --limit 1000 --json id,key,ref,sizeInBytes,createdAt)"
doomed="$(printf '%s' "$caches" | jq -r -b --arg mode "$mode" --arg prefix "$prefix" --arg ref "$ref" "$filter" | tr -d '\r')"

if [ -z "$doomed" ]; then
  echo "no Rust cache entries to delete"
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
