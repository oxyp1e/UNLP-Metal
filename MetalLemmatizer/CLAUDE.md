# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated morphological lemmatizer for Ukrainian (UNLP) using Apple Metal. Words are looked up in a trie compiled to binary format and traversed in parallel on the GPU. This is a research/PhD project targeting macOS with Apple Silicon.

## Build & Run Commands

The Makefile targets the CPU reference implementation (`trie_cpu_test`). The GPU lemmatizer (`lemmatizer`, via `main.m` + `AnalyzerMetal`) is built manually with clang.

```bash
# Build CPU trie test
make

# Run CPU trie test with sample words
make run
# or directly:
./trie_cpu_test running jumps walked

# Clean build artifacts
make clean

# Build the GPU Metal lemmatizer
clang -framework Foundation -framework Metal -O2 main.m AnalyzerMetal.m -o lemmatizer

# Run GPU lemmatizer on a word list file
./lemmatizer input.txt
```

## Data Pipeline

Binary trie data must be generated before running either executable:

```bash
# Generate binary trie files (used by both CPU and GPU paths)
cd scripts && python3 build_trie.py
# Outputs: resources/gpu_states.bin, resources/gpu_transitions.bin, resources/gpu_lemmas.bin
```

The dataset file `scripts/uk_lemmatizer_dataset.txt` is gitignored and must be provided separately.

## Architecture

### Two execution paths

| Path | Entry | Trie data |
|------|-------|-----------|
| CPU reference | `trie_cpu_test.m` (standalone `main`) | `resources/gpu_states.bin` + `resources/gpu_transitions.bin` + `resources/gpu_lemmas.bin` |
| GPU (Metal) | `main.m` → `AnalyzerMetal` | `resources/gpu_states.bin` + `resources/gpu_transitions.bin` + `resources/gpu_lemmas.bin` |

### Data structures (shared between CPU and GPU)

- **`GpuState`** (`transition_start_idx: u32`, `num_transitions: u32`, `lemma_offset: i32`) — 12 bytes each. `lemma_offset == -1` means no lemma at this state (non-terminal).
- **`GpuTransition`** (`c: u8/char`, `next_state: u32`) — 5 bytes each. Transitions within a state are stored **sorted by character** to enable binary search on the GPU.
- **Lemma buffer** — packed null-terminated UTF-8 strings; `lemma_offset` indexes into it.

### GPU kernel (`lookup_kernel.metal`)

One Metal thread per word. Each thread traverses the trie using **binary search** over transitions. On no-match, the original word is echoed as the output (identity fallback). Buffer bindings: `[0]` input words, `[1]` states, `[2]` transitions, `[3]` lemma buffer, `[4]` output lemmas, `[5]` max word length.

### Batching (`AnalyzerMetal.m`)

`lemmatizeBatch:` splits input into sub-batches of 100,000 words. Each sub-batch is dispatched as an async Metal command buffer. A `dispatch_semaphore` (count = active CPU cores) limits in-flight batches. Results are written back to a shared `NSMutableArray` under `@synchronized`.

### Buffer loading

`AnalyzerMetal` uses `newBufferWithBytesNoCopy:` with `MTLResourceStorageModeShared` (unified memory) to avoid copying trie data — the `NSData` backing must stay alive for the lifetime of the analyzer.

### `build_trie.py`

Builds trie from `(word, lemma)` pairs → flattens BFS order into states/transitions arrays → sorts transitions per state → serializes to binary. `MAX_WORD_LEN = 37` is a hard limit in both Python and ObjC/Metal code.
