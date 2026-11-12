#!/usr/bin/env bash
set -euo pipefail

prefix="purr${1:+-$1}"
src=release-artifacts
out="$PWD/release-assets"
mkdir -p "$out"

for dir in "$src"/purr-cli-*; do
  name="$(basename "$dir")"
  asset="$prefix-${name#purr-}"
  case "$name" in
    *windows*) (cd "$dir" && zip -qr "$out/$asset.zip" .) ;;
    *)
      chmod +x "$dir/purr"
      tar czf "$out/$asset.tar.gz" -C "$dir" .
      ;;
  esac
done

for dir in "$src"/purr-desktop-*; do
  name="$(basename "$dir")"
  asset="$prefix-${name#purr-}"
  for file in "$dir"/*; do
    case "$file" in
      *.AppImage) cp "$file" "$out/$asset.AppImage" ;;
      *.deb) cp "$file" "$out/$asset.deb" ;;
      *.dmg) cp "$file" "$out/$asset.dmg" ;;
      *-setup.exe) cp "$file" "$out/$asset-setup.exe" ;;
      *-portable.zip) cp "$file" "$out/$asset-portable.zip" ;;
      *)
        echo "unexpected desktop artifact: $file" >&2
        exit 1
        ;;
    esac
  done
done

tar czf "$out/$prefix-web.tar.gz" -C "$src/purr-web" .

(cd "$out" && sha256sum -- * > SHA256SUMS.txt)
ls -l "$out"
