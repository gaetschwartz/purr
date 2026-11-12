#!/usr/bin/env bash
set -euo pipefail

prefix="purr${1:+-$1}"
src=release-artifacts
out="$PWD/release-assets"
runtime_pack="$prefix-runtime-cuda-windows-x64.zip"
cuda_libraries=('cublas64_*.dll' 'cublasLt64_*.dll' 'cudart64_*.dll')
mkdir -p "$out"

for dir in "$src"/purr-cli-*; do
  name="$(basename "$dir")"
  asset="$prefix-${name#purr-}"
  case "$name" in
    *windows*)
      if compgen -G "$dir/cublas64_*.dll" >/dev/null && [ ! -e "$out/$runtime_pack" ]; then
        (cd "$dir" && zip -q "$out/$runtime_pack" cublas64_*.dll cublasLt64_*.dll)
      fi
      if [ "${PURR_SPLIT_CUDA_RUNTIME:-false}" = true ]; then
        (cd "$dir" && zip -qr "$out/$asset.zip" . -x "${cuda_libraries[@]}")
      else
        (cd "$dir" && zip -qr "$out/$asset.zip" .)
      fi
      ;;
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
      *-portable.zip)
        cp "$file" "$out/$asset-portable.zip"
        if [ "${PURR_SPLIT_CUDA_RUNTIME:-false}" = true ]; then
          status=0
          zip -qd "$out/$asset-portable.zip" "${cuda_libraries[@]}" || status=$?
          [ "$status" -eq 0 ] || [ "$status" -eq 12 ] || exit "$status"
        fi
        ;;
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
