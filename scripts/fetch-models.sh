#!/usr/bin/env bash
# Fetch the ONNX models the booth needs into a git-ignored models directory.
#
# The weights are not committed. Both sources are pinned to a revision so that
# two machines fetching a year apart get the same bytes, and every download is
# checked against a recorded SHA-256.
#
#   ./scripts/fetch-models.sh              # DINOv2 ViT-S/14 (default)
#   AF_DINOV2_SIZE=base ./scripts/fetch-models.sh
#   AF_MODELS_DIR=/srv/models ./scripts/fetch-models.sh
#
# The DINOv2 size chosen here must match `[models.embedder] file` in booth.toml.
# See docs/models.md for the licences and the reasoning.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${AF_MODELS_DIR:-$repo_root/models}"
dinov2_size="${AF_DINOV2_SIZE:-small}"

# YuNet (MIT), OpenCV Zoo, pinned to the commit that last touched the file.
yunet_rev="f12e12798e8314f7c074a6656816c048dcc95b7a"
yunet_file="face_detection_yunet_2023mar.onnx"
yunet_url="https://github.com/opencv/opencv_zoo/raw/${yunet_rev}/models/face_detection_yunet/${yunet_file}"
yunet_sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# DINOv2 (Apache 2.0), ONNX export by onnx-community, pinned per size.
case "$dinov2_size" in
  small)
    dinov2_rev="8b1f705a3a7f6f062f6bdd21986c1583d3ef105d"
    dinov2_sha256="f22797eabf810a75e41de68d378541ebea372122b25c4ce3ef25ff618250c20a"
    ;;
  base)
    dinov2_rev="31ef06cac16d5d301c5930d147002a058c85a5e4"
    dinov2_sha256=""
    ;;
  large)
    dinov2_rev="4b3550c593d51ac1a870d48d411f35eff4eaf353"
    dinov2_sha256=""
    ;;
  *)
    echo "unknown AF_DINOV2_SIZE '$dinov2_size' (expected small, base or large)" >&2
    exit 2
    ;;
esac
dinov2_file="dinov2-${dinov2_size}.onnx"
dinov2_url="https://huggingface.co/onnx-community/dinov2-${dinov2_size}/resolve/${dinov2_rev}/onnx/model.onnx"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

fetch() {
  local url="$1" dest="$2" expected="$3"

  if [[ -f "$dest" ]]; then
    echo "have    $(basename "$dest")"
  else
    echo "fetch   $(basename "$dest")"
    curl --fail --location --progress-bar --output "$dest.part" "$url"
    mv "$dest.part" "$dest"
  fi

  local actual
  actual="$(sha256_of "$dest")"
  if [[ -z "$expected" ]]; then
    echo "        sha256 $actual (not pinned — record it in scripts/fetch-models.sh)"
  elif [[ "$actual" != "$expected" ]]; then
    echo "checksum mismatch for $dest" >&2
    echo "  expected $expected" >&2
    echo "  actual   $actual" >&2
    exit 1
  fi
}

mkdir -p "$models_dir"
fetch "$yunet_url" "$models_dir/$yunet_file" "$yunet_sha256"
fetch "$dinov2_url" "$models_dir/$dinov2_file" "$dinov2_sha256"

echo
echo "models in $models_dir:"
ls -lh "$models_dir"
echo
echo "verify with: cargo run -p afbooth"
