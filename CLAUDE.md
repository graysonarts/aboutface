# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**About:Face** is an art project (early development): a booth photographs consenting visitors and displays them alongside similar-looking people. This repo currently contains the C++ image-analysis half — a set of "annotator" command-line tools built on OpenCV.

> **Direction (2026-08-17).** The C++ annotators described below are a 2015 prototype and have been **retired by decision, not yet by deletion** — they still build and still work, but nothing new should be added to them. The piece is being rebuilt in Rust around learned face embeddings, with a wgpu renderer and no openFrameworks. Read [`CONTEXT.md`](CONTEXT.md) for vocabulary, [`docs/adr/`](docs/adr/) for the decisions, and [`docs/implementation-plan.md`](docs/implementation-plan.md) for the staged plan **before** doing design work here. The architecture notes below remain accurate as a description of the existing code.

## Build

Dependencies must be installed on the system: **OpenCV** (2.x-era C API constants like `CV_BGR2GRAY`, `CV_RGB`, `CV_BGR2HSV_FULL` are used, so OpenCV 4 will not compile without changes), **TBB**, and **TCLAP** (`CMake/FindTCLAP.cmake` falls back to the vendored `contrib/tclap/include`).

```sh
mkdir build && cd build && cmake .. && make      # binaries land in build/bin/
./update.sh                                      # alternative: generates build-Xcode/ and build-Ninja/
make install                                     # installs into ./dist/about:face-<version>/ (source tree, not a system prefix)
```

There is no test suite and no build CI (the only workflow is a CodeSee architecture diagram).

## Running the annotators

Binaries go to `<build-dir>/bin/`. The `main.cpp` of each annotator carries comment lines with working sample invocations and pre-computed bounding boxes for `samples/*.jpg` — run from `<build-dir>/bin/` so the `../../..`-relative paths in those comments resolve.

```sh
face_isolator -d ../../../data ../../../samples/1.jpg          # -> "Face: 84,120+346x346"
skin_color_picker ../../../samples/1.jpg 84,120+346x346        # -> "Skin: <avg hue>"
feature_extractor ../../../contrib/asmlib-opencv/data/color_asm75.model ../../../samples/1.jpg 84,120+346x346
```

Every annotator takes `-v`/`--view` to render the result in an OpenCV window instead of printing it, and gets `--version` for free from TCLAP.

## Architecture

**Annotators are independent processes composed via stdout, not a linked pipeline.** Each tool in `annotators/` is a separate executable that reads an image (plus, for downstream stages, a face bounding box) and prints one labeled line. The intended flow is `face_isolator` → (bounding box) → `skin_color_picker` / `feature_extractor`. Nothing wires them together yet; the box is passed by hand or by an outer script.

**`af::common::Rectangle` is the inter-process wire format.** `common/` builds the `aboutface_common` shared library, whose only real content is `Rectangle` and its `toString()`/`fromString()` pair using the geometry syntax `X,Y+WxH` (parsed by the regex in `common/src/rectangle.cpp`). Changing that format breaks the CLI contract between annotators. `common/include/opencv_adapters.h` is header-only and holds the `af::adapters::makeRectangle` / `makeCvRect` conversions — it exists so `Rectangle` stays free of OpenCV types and public annotator headers can forward-declare it.

**Each annotator follows the same shape:** a thin `main.cpp` doing TCLAP argument parsing and printing, and one pimpl class (`FaceIsolator`, `SkinHueAverager`, `PointExtractor`) whose public header keeps OpenCV out of the interface where possible and whose `Impl` in the `.cpp` does all the OpenCV work. Each Impl exposes parallel `extract`/`average`/`find_*` and `display`/`show_*` methods over a shared private helper.

**Versioning is generated at configure time.** `CMake/Version.cmake` embeds the git short SHA into `project_VERSION`, writes `${CMAKE_BINARY_DIR}/bin/version.h` from `version.h.in`, and that `PROJECT_VERSION` macro is what each `main.cpp` passes to `TCLAP::CmdLine`. It also names the install directory `dist/about:face-<version>/`.

**Model and cascade data live in two places:** `data/` holds the Haar cascades `face_isolator` loads by filename (`haarcascade_frontalface_alt2.xml`, `haarcascade_profileface.xml` — names are constants in `face_isolator.h`, resolved against `--datadir`, default `./data`). ASM models for `feature_extractor` live in `contrib/asmlib-opencv/data/` and are passed as an explicit path argument.

## Agent skills

The `mattpocock-skills` plugin is enabled for this repo via `.claude/settings.json`.

### Issue tracker

Issues live in this repo's GitHub Issues (`graysonarts/aboutface`), driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, each using its own name as the label string. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and one `docs/adr/` at the repo root (neither created yet). See `docs/agents/domain.md`.

## contrib/

`contrib/tclap`, `contrib/FindTBB`, and `contrib/asmlib-opencv` are vendored third-party sources, not submodules. Only `asmlib-opencv` is built (as the static `asm` target consumed by `feature_extractor`), and its `src/CMakeLists.txt` has been **patched** for this build — `src/CMakeLists.txt.orig` is the upstream copy kept for diffing. The patch drops the Qt annotator, demo, and Doxygen targets, removes the nested `project()`/`cmake_minimum_required()`, and adds the TBB dependency. Preserve those edits when touching that file. Treat the rest of `contrib/` as read-only.
