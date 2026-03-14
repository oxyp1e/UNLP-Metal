# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated morphological lemmatizer for Ukrainian (UNLP) using Apple Metal. Words are looked up in a trie compiled to binary format and traversed in parallel on the GPU. Research/PhD project targeting macOS with Apple Silicon (or any Metal-capable Mac).

## Build & Run Commands

All source files live under `MetalLemmatizer/`. Build from that directory.

```bash
cd MetalLemmatizer

# Build GPU lemmatizer (one-shot)
clang -framework Foundation -framework Metal -O2 main.m AnalyzerMetal.m -o lemmatizer

# Build GPU lemmatizer (loop benchmark, steady-state throughput)
clang -framework Foundation -framework Metal -O2 main_loop.m AnalyzerMetal.m -o lemmatizer_loop

# Build CPU trie reference test
make

# Run CPU test
./trie_cpu_test running jumps walked

# Clean
make clean
```

There is also an Xcode project (`MetalLemmatizer.xcodeproj`) at the repo root.

### Running the GPU lemmatizer

Three kernel variants selected via flags:

| Flag | Layout | Kernel function |
|------|--------|-----------------|
| _(none)_ | fixed-stride | `lookup_kernel` |
| `--packed` | packed offsets | `lookup_kernel_packed` |
| `--packed-col` | packed + index output | `lookup_kernel_index` |

```bash
./lemmatizer articles.txt                  # fixed-stride
./lemmatizer --packed articles.txt         # packed
./lemmatizer --packed-col articles.txt     # packed-col (fastest)

# Loop benchmark: kernel-only throughput for N seconds
./lemmatizer_loop --packed-col articles.txt 30
```

## Data Pipeline

Binary trie data must be generated before running either executable:

```bash
cd MetalLemmatizer/scripts && python3 build_trie.py
# Outputs: resources/gpu_states.bin, resources/gpu_transitions.bin, resources/gpu_lemmas.bin
```

The dataset file `scripts/uk_lemmatizer_dataset.txt` is gitignored and must be provided separately.

## Architecture

### Execution paths

| Path | Entry point | Purpose |
|------|-------------|---------|
| CPU reference | `trie_cpu_test.m` | Validation, CPU baseline. Uses `dispatch_apply` for parallelism. |
| GPU one-shot | `main.m` → `AnalyzerMetal` | End-to-end latency measurement (preprocess → pack → kernel → decode). |
| GPU loop | `MetalLemmatizerLoop/main_loop.m` → `AnalyzerMetal` | Steady-state GPU throughput. Uploads once, hammers kernel for N seconds. |

### Trie data structures (shared between CPU, GPU, and Python)

- **`GpuState`** (12 bytes): `transition_start_idx: u32`, `num_transitions: u32`, `lemma_offset: i32`. `lemma_offset == -1` means non-terminal.
- **`GpuTransition`** (5 bytes): `c: u8`, `next_state: u32`. Sorted by character within each state for binary search.
- **Lemma buffer**: packed null-terminated UTF-8 strings indexed by `lemma_offset`.
- `MAX_WORD_LEN = 37` — hard limit in Python, ObjC, and Metal code.

### GPU kernel (`lookup_kernel.metal`)

One Metal thread per word. Each thread traverses the trie using binary search over transitions. On no-match, the original word is echoed (identity fallback). Buffer bindings: `[0]` input words, `[1]` states, `[2]` transitions, `[3]` lemma buffer, `[4]` output, `[5]` max word length.

### Batching (`AnalyzerMetal.m`)

Input split into sub-batches of 100,000 words. Each sub-batch dispatched as an async Metal command buffer. `dispatch_semaphore` (count = active CPU cores) limits in-flight batches. Uses `MTLResourceStorageModeShared` (unified memory) — no explicit H2D/D2H copies needed.

### `build_trie.py`

Builds trie from `(word, lemma)` pairs → BFS flattening → sorted transitions per state → binary serialization.

## Timing Phases

| Phase | Measures |
|-------|----------|
| Preprocess | File I/O + tokenize + filter |
| Pack | CPU writes into shared `MTLBuffer` (H2D equivalent) |
| Kernel | Pure GPU trie traversal (hardware timestamps) |
| Decode | Result readback + `NSString` construction |

Loop benchmarks report only Kernel time (pack is one-time, decode is skipped).
