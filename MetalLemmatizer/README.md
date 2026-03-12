# UNLP-Metal — Benchmark Guide

GPU-accelerated Ukrainian morphological lemmatizer using Apple Metal. Trie lookup runs on the GPU via Metal compute shaders. macOS / Apple Silicon only.

---

## Prerequisites

- macOS with Apple Silicon (or any Metal-capable Mac)
- Xcode command-line tools (`xcode-select --install`)
- Trie binary files must be pre-built (see below)

---

## Build

```bash
# GPU lemmatizer (one-shot)
clang -framework Foundation -framework Metal -O2 main.m AnalyzerMetal.m -o lemmatizer

# GPU lemmatizer (loop benchmark)
clang -framework Foundation -framework Metal -O2 main_loop.m AnalyzerMetal.m -o lemmatizer_loop

# CPU trie reference test
make
```

Or build both GPU binaries at once:

```bash
clang -framework Foundation -framework Metal -O2 main.m AnalyzerMetal.m -o lemmatizer && \
clang -framework Foundation -framework Metal -O2 main_loop.m AnalyzerMetal.m -o lemmatizer_loop
```

---

## Generate trie data (required before first run)

```bash
cd scripts && python3 build_trie.py
# Outputs: resources/gpu_states.bin, resources/gpu_transitions.bin, resources/gpu_lemmas.bin
```

The dataset file `scripts/uk_lemmatizer_dataset.txt` must be provided separately (gitignored).

---

## Input files

Use the same `articles.txt` (~761k words) used in the CUDA project for apples-to-apples comparison.

---

## Benchmark suite

Three kernel variants are available via flags:

| Flag | Layout | Kernel |
|------|--------|--------|
| _(none)_ | fixed-stride | `lookup_kernel` |
| `--packed` | packed offsets | `lookup_kernel_packed` |
| `--packed-col` | packed + index output | `lookup_kernel_index` |

---

### 1. GPU one-shot (`lemmatizer`)

Single pass end-to-end: preprocess → pack → kernel → decode. Measures real-world latency.

```bash
# Fixed-stride layout
./lemmatizer articles.txt

# Packed layout
./lemmatizer --packed articles.txt

# Packed-column layout (index output, fastest kernel)
./lemmatizer --packed-col articles.txt
```

Output breakdown:
```
[packed-col] Words: 761625
  Preprocess (file I/O + tokenize + filter):  ??? ms
  Pack (CPU→MTLBuffer, H2D equivalent):       ??? ms
  Kernel:                                     ??? ms  (??? words/sec)
  Decode+dispatch overhead (approx):          ??? ms
  GPU total (pack+kernel+decode):             ??? ms
  End-to-end (preprocess+GPU):                ??? ms
```

Run all three variants back to back:

```bash
./lemmatizer articles.txt && \
./lemmatizer --packed articles.txt && \
./lemmatizer --packed-col articles.txt
```

---

### 2. GPU loop — steady-state throughput (`lemmatizer_loop`)

Uploads data once, hammers the kernel for N seconds. Measures pure GPU throughput after GPU caches warm up.

```bash
# Fixed-stride, 30 seconds
./lemmatizer_loop articles.txt 30

# Packed layout, 30 seconds
./lemmatizer_loop --packed articles.txt 30

# Packed-column layout, 30 seconds
./lemmatizer_loop --packed-col articles.txt 30

# Quick 10-second runs
./lemmatizer_loop articles.txt 10
./lemmatizer_loop --packed articles.txt 10
./lemmatizer_loop --packed-col articles.txt 10
```

Output:
```
Preprocess (file I/O + tokenize + filter):  ??? ms
Pack (build MTLBuffer + memcpy):            ??? ms
Running packed-col loop for 30.0s  words=761625
iter   100  avg ?.??? ms  throughput ???M words/sec
...
=== Final ===
  Iters:       ???
  Words/iter:  761625
  Avg kernel:  ?.??? ms
  Peak kernel: ?.??? ms   ← cold first iter
  Throughput:  ???M words/sec
```

---

### 3. Run the full suite at once

```bash
INPUT=articles.txt

echo "=== One-shot: fixed-stride ===" && ./lemmatizer $INPUT 1>/dev/null && \
echo "=== One-shot: packed ===" && ./lemmatizer --packed $INPUT 1>/dev/null && \
echo "=== One-shot: packed-col ===" && ./lemmatizer --packed-col $INPUT 1>/dev/null && \
echo "=== Loop: fixed-stride (30s) ===" && ./lemmatizer_loop $INPUT 30 && \
echo "=== Loop: packed (30s) ===" && ./lemmatizer_loop --packed $INPUT 30 && \
echo "=== Loop: packed-col (30s) ===" && ./lemmatizer_loop --packed-col $INPUT 30
```

---

## Timing phases explained

| Phase | What it measures | Mechanism |
|-------|-----------------|-----------|
| Preprocess | File read + `componentsSeparatedByCharactersInSet` + filter | `clock_gettime(CLOCK_MONOTONIC)` |
| Pack | Build `MTLBuffer` + `strncpy`/`memcpy` words into it (H2D equivalent — unified memory) | `clock_gettime` |
| Kernel | Pure GPU trie traversal | `cmdBuf.GPUEndTime - GPUStartTime` (hardware timestamp) |
| Decode+dispatch | Result readback + `NSString` construction + dispatch overhead | wall − pack − kernel (approx) |
| GPU total | Pack + kernel + decode — wall clock inside `lemmatizeBatch*` | `clock_gettime` wall |
| End-to-end | Preprocess + GPU total | sum |

Loop benchmarks report only **Kernel** time (pack is one-time setup, decode is skipped).

---

## Comparing with CUDA results

The timing phases are designed to be directly comparable to `cuda-lemmatizer-cpp`:

| Phase | CUDA | Metal |
|-------|------|-------|
| Preprocess | ICU `lowercase_ukr()` + tokenize | `componentsSeparatedByCharactersInSet` + filter |
| H2D / Pack | `cudaMemcpy` → device | CPU write into `MTLResourceStorageModeShared` buffer |
| Kernel | `cudaEvent` hardware timestamp | `GPUEndTime - GPUStartTime` hardware timestamp |
| D2H / Decode | `cudaMemcpy` → host | read `outputBuffer.contents` + build `NSString` |

**Note:** Metal uses unified memory — there is no physical copy for H2D/D2H. The "Pack" time is purely the CPU cost of writing the data into the shared buffer. This makes Metal's H2D+D2H inherently faster than discrete GPU CUDA transfers.
