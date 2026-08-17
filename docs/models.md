# Model assets

The two ONNX models are **fetched, never committed** — `/models` and `*.onnx`
are git-ignored. Which models and why is settled in
[ADR-0007](adr/0007-apparent-resemblance-from-permissive-weights.md); where they
come from and how to get them is here.

## Fetch

```sh
./scripts/fetch-models.sh                     # DINOv2 ViT-S/14
AF_DINOV2_SIZE=base ./scripts/fetch-models.sh # ViT-B/14
AF_MODELS_DIR=/srv/models ./scripts/fetch-models.sh
```

Both downloads are pinned to a source revision and checked against a recorded
SHA-256, so two machines fetching a year apart get the same bytes. Re-running is
cheap: an existing file is verified, not re-downloaded.

Then `cargo run -p afbooth`, which reports each model's path, whether it is
present, its `ModelId`, and the execution provider ONNX Runtime selected.

## Sources and licences

| Role | Model | Source | Licence |
| --- | --- | --- | --- |
| Detector | YuNet `face_detection_yunet_2023mar` | [opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT — [`LICENSE`](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/LICENSE), © 2020 Shiqi Yu |
| Embedder | DINOv2 ViT-S/B/L | [onnx-community/dinov2-*](https://huggingface.co/onnx-community/dinov2-small) | Apache 2.0 via [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small) |

The `onnx-community` repositories are ONNX exports of Meta's `facebook/dinov2-*`
weights and state no licence of their own beyond that base model, whose model
card declares `license: apache-2.0`. If that ever needs to be watertight, export
the ONNX directly from `facebook/dinov2-*` with 🤗 Optimum rather than relying on
a third-party mirror. Nothing in the pipeline is research-only (ADR-0007).

## Which ViT size

Open, and it follows the hardware decision (ADR-0006). Nothing in the code bakes
a size in: the file name lives in `booth.toml`, and the `ModelId` is derived from
that file name, so `dinov2-small.onnx` and `dinov2-large.onnx` produce different
identifiers and the Corpus can tell their Embeddings apart. Changing size means
re-fetching, editing `[models.embedder] file`, and re-embedding the Corpus.

Only ViT-S is checksum-pinned; run the script for `base` or `large` and it prints
the SHA-256 to record.

## Configuration

`booth.toml` in the working directory, or wherever `AFBOOTH_CONFIG` points.
Relative paths inside it resolve against the file's own directory.

```toml
[models]
dir = "models"

[models.detector]
file = "face_detection_yunet_2023mar.onnx"
# id = "yunet-2023mar"   # optional; otherwise derived from the file stem
```

A missing file is reported by the self-check with the fetch command and exits
non-zero. It is never a panic — an operator with an incomplete install should be
told what to run, not shown a backtrace.
