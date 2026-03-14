# GPU Pipeline & Performance Analysis

Deep code-level analysis of the Metal GPU lemmatizer data pipeline, kernel
implementations, host-side orchestration, and performance characteristics.

---

## 1. Data Pipeline: Python → Binary → Metal Buffers

### 1.1 Trie Construction (`scripts/build_trie.py`)

The trie is built in-memory as nested Python dicts, then flattened in BFS order
into three flat arrays:

```
(word, lemma) pairs
       │
       ▼
  nested dict trie
       │  BFS flatten
       ▼
┌─────────────┐  ┌──────────────────┐  ┌────────────────┐
│ GpuState[]  │  │ GpuTransition[]  │  │ lemma_buffer[] │
│ 12 B each   │  │ 5 B each         │  │ packed UTF-8   │
└─────────────┘  └──────────────────┘  └────────────────┘
       │                │                      │
       ▼                ▼                      ▼
  gpu_states.bin   gpu_transitions.bin    gpu_lemmas.bin
```

Each `GpuState` is serialized as `(transition_start_idx: u32,
num_transitions: u32, lemma_offset: i32)` — 12 bytes via `struct.pack("IIi")`.
A `lemma_offset` of `-1` (actually `0` at init, overwritten only on terminal
nodes) marks non-terminal states.

Each `GpuTransition` is `(c: u8, next_state: u32)` — 5 bytes via
`struct.pack("B I")`.

Transitions within each state are stored **sorted by character**
(`sorted(node.keys())` at line 35), which prepares the data for binary search
on the GPU — though the current kernels do not yet exploit this ordering (see
§4.3).

### 1.2 Binary File Mismatch

`build_trie.py` (line 80) writes a **single** `trie.bin` with a 12-byte header
(`III` = 3×uint32 for section sizes), while `AnalyzerMetal.m` (lines 48–50)
loads **three separate files** (`gpu_states.bin`, `gpu_transitions.bin`,
`gpu_lemmas.bin`). This is a data pipeline integration gap — the Python script
and the ObjC loader disagree on the file format. Either the Python script needs
to be updated to write three files, or the loader needs to parse the combined
format.

### 1.3 Zero-Copy Buffer Loading (`AnalyzerMetal.m:48–54`)

```objc
_statesData = [NSData dataWithContentsOfFile:@"resources/gpu_states.bin"];
_statesBuffer = [device newBufferWithBytesNoCopy:(void *)_statesData.bytes
                 length:_statesData.length
                 options:MTLResourceStorageModeShared
                 deallocator:nil];
```

`newBufferWithBytesNoCopy:` wraps the existing `NSData` backing memory as a
Metal buffer **without any memcpy**. The `NSData` objects are retained as
properties (`_statesData`, `_transitionsData`, `_lemmasData`) to keep the
backing pages alive for the lifetime of the analyzer.

On Apple Silicon unified memory, this means the GPU reads directly from the
same physical pages — **no host-to-device transfer exists at all**. This is the
single most important architectural advantage over CUDA discrete-GPU designs.

**Alignment note:** `newBufferWithBytesNoCopy:` requires the pointer to be
page-aligned (typically 16 KB on Apple Silicon). `NSData dataWithContentsOfFile:`
uses `mmap` internally, which returns page-aligned pointers — so this works
correctly, but there is no explicit alignment check. If `NSData` were ever
initialized differently (e.g., via `dataWithBytes:`), it would fail or crash
silently.

---

## 2. Kernel Compilation & Pipeline Setup

`AnalyzerMetal.m:30–46` compiles the Metal source at runtime:

```objc
NSString *kernelSource = [NSString stringWithContentsOfFile:@"lookup_kernel.metal" ...];
id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
```

Three compute pipeline states are created from three kernel functions:

| Pipeline property | Kernel function | Flag |
|-------------------|-----------------|------|
| `_pipeline` | `lookup_kernel` | _(none)_ |
| `_pipelinePacked` | `lookup_kernel_packed` | `--packed` |
| `_pipelineIndex` | `lookup_kernel_index` | `--packed-col` |

Runtime compilation means the Metal compiler can optimize for the specific GPU
architecture at launch. The trade-off is a one-time startup cost (~100–300 ms)
that would be avoided by pre-compiling a `.metallib`.

---

## 3. Three Kernel Variants — Detailed Comparison

### 3.1 `lookup_kernel` (Fixed-Stride) — `lookup_kernel.metal:105–159`

**Memory layout:**
```
Input buffer:  [ word₀ padded to max_word_len | word₁ padded | ... ]
Output buffer: [ lemma₀ padded to max_word_len | lemma₁ padded | ... ]
```

Each thread computes its offset as `gid * max_word_len`. Every word occupies
exactly `max_word_len` bytes regardless of actual length.

**Host-side packing** (`AnalyzerMetal.m:94–109`):

```objc
NSUInteger max_word_len = 0;
for (NSString *word in batch)
    max_word_len = MAX(max_word_len, [word lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);
NSUInteger bufferSize = batch.count * max_word_len;
// ... memset(0) + strncpy per word
```

The entire buffer is zeroed (`memset`) then each word is `strncpy`'d into its
fixed-size slot. For 100k words with max length 37, this allocates ~3.7 MB
per input buffer.

**Performance characteristics:**

- **Wasted bandwidth:** If max word is 37 bytes but average word is ~8 bytes,
  roughly 78% of input reads are zero-padding. For 761k words the input buffer
  is `761625 × 37 ≈ 27 MB` instead of ~6 MB of actual data.
- **Stride regularity:** All threads access memory at perfectly uniform strides,
  which is excellent for GPU memory coalescing — adjacent threads in a SIMD
  group read adjacent `max_word_len` blocks.
- **Output cost:** Writes full strings (lemma or identity fallback) to the
  output buffer, requiring byte-by-byte copy in the kernel.

### 3.2 `lookup_kernel_packed` — `lookup_kernel.metal:17–65`

**Memory layout:**
```
Input buffer:   [ "cat\0" "dogs\0" "apple\0" ... ]  (tightly packed)
Offsets buffer: [ 0, 4, 9, 15, ... ]                 (N+1 uint32s)
Output buffer:  same packed layout as input
```

Each thread reads `offsets[gid]` and `offsets[gid+1]` to find its word
boundaries.

**Host-side packing** (`AnalyzerMetal.m:200–219`):

A prefix-sum over word byte lengths produces the offsets array. Words are
then `memcpy`'d contiguously into the input buffer — no padding, no zeroing.

**Performance characteristics:**

- **Compact input:** No padding waste. Input buffer is exactly the size of the
  character data.
- **Irregular access patterns:** Adjacent threads in a SIMD group access words
  at different offsets with different lengths. This destroys memory
  coalescing — thread 0 might read from byte 0, thread 1 from byte 5,
  thread 2 from byte 12, etc. The GPU memory controller must service
  scattered reads.
- **Extra indirection:** Two dependent loads per thread (offsets → offset value
  → character data) before any trie traversal begins.
- **Full string output:** Same byte-copy output as fixed-stride, but at
  irregular offsets.

### 3.3 `lookup_kernel_index` — `lookup_kernel.metal:67–103`

**Memory layout:**
```
Input buffer:   packed (same as 3.2)
Offsets buffer: same as 3.2
Output buffer:  int32[N] — just lemma_offset indices, NOT full strings
```

Each thread writes a single `int32_t` to `out_indices[gid]`: the
`lemma_offset` if found, or `-1` for no match.

**Host-side decode** (`AnalyzerMetal.m:356–389`):

The completed handler performs a two-pass CPU decode:
1. Prefix-sum to compute output offsets from lemma lengths.
2. `memcpy` each lemma from the shared lemma buffer into a packed output.

```objc
const char *src = (indices[j] >= 0)
    ? (lemmaBytes + indices[j])    // trie hit  → read from lemma buffer
    : (inputRaw + off[j]);         // trie miss → identity fallback
```

**Performance characteristics:**

- **Minimal GPU output:** Instead of copying variable-length strings, each
  thread writes exactly 4 bytes. This is a massive reduction in GPU memory
  write bandwidth.
- **Perfectly coalesced output writes:** 32 threads in a SIMD group write a
  contiguous 128-byte block — a single cache line transaction.
- **Decode shifted to CPU:** Variable-length string assembly happens on the
  CPU where it is more natural. On unified memory, the CPU reads the lemma
  buffer at full DRAM bandwidth with no transfer overhead.

### 3.4 Kernel Output Bandwidth Comparison

| Kernel | Output per thread | Write pattern | Relative cost |
|--------|------------------|---------------|---------------|
| `lookup_kernel` | up to 37 bytes | Strided, sequential per thread | High |
| `lookup_kernel_packed` | variable bytes | Scattered, sequential per thread | High |
| `lookup_kernel_index` | exactly 4 bytes | Perfectly coalesced `int32[gid]` | Low |

`lookup_kernel_index` wins on output bandwidth by roughly an order of
magnitude.

---

## 4. GPU Kernel Execution — Performance Deep-Dive

### 4.1 Thread Dispatch

```objc
MTLSize gridSize = MTLSizeMake(batch.count, 1, 1);          // 1D grid
MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);             // w = threadExecutionWidth
```

`threadExecutionWidth` is the SIMD width — 32 on Apple Silicon GPUs.
`dispatchThreads:threadsPerThreadgroup:` handles non-multiple-of-32 batch
sizes via partial SIMD groups.

For a 100k-word batch: `100000 / 32 = 3125 SIMD groups`. Apple Silicon GPUs
have 128 execution units (largest config), giving ~24× oversubscription —
ample parallelism to hide memory latency.

### 4.2 Memory Access Pattern: Pointer-Chasing Trie Traversal

The core of every kernel is a serial dependency chain:

```metal
for each character ch in word:
    GpuState s = states[state];                                    // random read
    for k in 0..s.num_transitions:
        GpuTransition t = transitions[s.transition_start_idx + k]; // sequential scan
        if t.c == ch: state = t.next_state; break;                 // data-dependent next
```

- `states[state]` is **data-dependent** — the address depends on the result of
  the previous iteration. This creates a serial chain that cannot be pipelined
  within a single thread.
- Adjacent threads in a SIMD group process **different words**, so they hit
  **different states** — the reads scatter across the states array with no
  spatial locality between threads.
- For the transitions array, `s.transition_start_idx + k` is a sequential scan
  within a contiguous block, but different threads scan different blocks. This
  is a gather pattern from the GPU's perspective.

**Cache behavior:** The trie data structures are read-only and shared across all
threads. A Ukrainian dictionary with ~500k states occupies ~6 MB of state data.
Apple Silicon GPU L2 cache is 4–8 MB depending on the chip. Frequently-accessed
upper trie nodes (root, first few levels) will be hot in cache, but deep trie
nodes will incur L2 misses. This is inherent to the trie data structure.

### 4.3 Linear Scan vs Binary Search

All three kernels use **linear scan** for transition lookup:

```metal
for (uint k = 0; k < s.num_transitions; ++k) {
    GpuTransition t = transitions[s.transition_start_idx + k];
    if (t.c == static_cast<uchar>(ch)) { ... break; }
}
```

The transitions are sorted by character in `build_trie.py`
(`sorted(node.keys())` at line 35), but none of the kernels exploit this
ordering with binary search.

**Impact estimation:** The Ukrainian Cyrillic alphabet plus digits and
punctuation yields ~60+ possible transitions from the root state. A linear scan
averages ~33 comparisons at root; binary search would need ~6 (`log₂(66)`).
Deeper states typically have 1–5 transitions where linear scan is actually
faster (no branch overhead). The net benefit depends on word length
distribution — short words (2–4 characters) spend proportionally more time at
high-fan-out nodes near the root and would benefit most.

A hybrid approach — binary search when `num_transitions > 8`, linear scan
otherwise — would capture the best of both strategies.

### 4.4 SIMD Divergence

Two sources of divergence within a 32-thread SIMD group:

1. **Variable word lengths.** Some threads finish their word after 3 characters,
   others after 20+. Threads that finish early are masked off while the longest
   word in the group completes. Effective utilization depends on word length
   variance within each SIMD group.

2. **Early-break on transition match.** The `for k ... break` loop means
   threads find their match at different iterations. Threads matching on the
   first transition wait for threads that must scan to the last.

3. **Fallback divergence.** Threads that fail trie lookup execute the identity-
   copy path while matched threads read from the lemma buffer. This is a
   branch divergence that forces the SIMD group to execute both code paths
   serially.

Estimated effective SIMD utilization: **30–50%**, depending on input
characteristics.

---

## 5. Host-Side Batching & Concurrency

### 5.1 Batch Dispatch Model (`AnalyzerMetal.m:83–153`)

```
For each 100k-word sub-batch:
  1. dispatch_semaphore_wait()    — throttle to #CPUs in-flight batches
  2. Pack input into MTLBuffer    — CPU work, timed separately
  3. Encode + commit cmdBuf       — non-blocking enqueue
  4. addCompletedHandler fires    — accumulate GPU time, decode results
  5. dispatch_semaphore_signal()  — release slot for next batch
Wait on dispatch_group for all batches.
```

The semaphore count is `[[NSProcessInfo processInfo] activeProcessorCount]`. On
an M1 Max (10 cores) this allows 10 in-flight batches. Each batch allocates
2–3 new MTL buffers. With 100k words × 37 bytes max ≈ 3.7 MB per input buffer,
10 in-flight batches consume ~74 MB of transient buffer memory. This is within
Apple Silicon's unified memory budget but puts pressure on the Metal heap
allocator for rapid alloc/dealloc cycles.

### 5.2 Concurrency Correctness

**Result array writes:**

Results are written into a pre-allocated `NSMutableArray` (pre-filled with
empty strings at line 67). Each batch writes to a **disjoint** index range
(`range.location + j`), so no two completed handlers ever write to the same
index.

- Fixed-stride and packed variants (`lemmatizeBatch:` and
  `lemmatizeBatchPacked:`) acquire `@synchronized(allResults)` **per element**
  inside the loop (lines 144–146, 254–256). This is O(N) lock acquisitions per
  batch — correct but unnecessarily expensive.
- The index variant (`lemmatizeBatchPackedColumn:`) acquires the lock **once**
  for the entire extraction loop (line 382) — O(1) lock acquisitions per batch.

Since the index ranges are disjoint, the lock is not needed for correctness in
any variant. `NSMutableArray` element replacement (`replaceObjectAtIndex:`) is
safe without locking when no structural mutations (add/remove/resize) occur,
and the array is pre-sized. Removing the `@synchronized` would eliminate a
measurable CPU-side bottleneck during decode.

### 5.3 Timing Accumulation

GPU times from each batch are accumulated via:

```objc
double batchGpuMs = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0;
@synchronized (gpuTimeLock) { gpuTimeAccumMs += batchGpuMs; }
```

`GPUEndTime - GPUStartTime` is a **hardware timestamp** from the GPU's own
clock. When multiple batches overlap on the Metal command queue, the sum of
per-batch GPU times can **exceed** the wall time. The reported "Kernel" time is
an aggregate compute-time metric, not wall-clock duration. This is the correct
metric for comparing against CUDA kernel times but can be misleading when
interpreting end-to-end latency.

### 5.4 Per-Batch Buffer Allocation

Every sub-batch creates fresh `MTLBuffer` objects:

```objc
id<MTLBuffer> inputBuffer  = [self.device newBufferWithLength:bufferSize
                              options:MTLResourceStorageModeShared];
id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:bufferSize
                              options:MTLResourceStorageModeShared];
```

These are autoreleased after the completed handler finishes. For the benchmark
use case this is fine, but for a production streaming pipeline, reusing a pool
of pre-allocated buffers (double- or triple-buffered) would eliminate allocation
overhead and reduce memory fragmentation.

---

## 6. Loop Benchmark Path

The `benchLoop*` methods (`AnalyzerMetal.m:424–648`) pack data **once**, then
run the kernel in a tight synchronous loop:

```objc
while (elapsed < seconds) {
    // encode + commit + waitUntilCompleted (synchronous)
    double ms = (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
    totalKernelMs += ms;
}
```

Key differences from the one-shot path:
- **No decode overhead.** Output buffers are written but never read back.
- **Synchronous dispatch.** `waitUntilCompleted` blocks the CPU until each
  kernel finishes, ensuring clean per-iteration timing. No pipelining.
- **Cache warming.** After the first iteration, the trie data is hot in GPU
  cache. Steady-state throughput reflects the GPU's best-case performance.
- **Same input every iteration.** The input buffer is reused, so the GPU may
  benefit from input-data caching as well.

The loop path is the correct way to measure pure GPU compute throughput,
isolating it from CPU packing, buffer allocation, and string decode costs.

---

## 7. Consolidated Performance Findings

### Strengths

| Aspect | Detail |
|--------|--------|
| Zero-copy trie loading | `newBufferWithBytesNoCopy` on unified memory avoids all H2D cost |
| `lookup_kernel_index` output design | 4 bytes/thread, perfectly coalesced, CPU-side decode |
| Hardware GPU timestamps | `GPUEndTime - GPUStartTime` gives accurate kernel-only timing |
| Async batch pipelining | Semaphore-bounded overlap of pack/encode/execute across batches |
| Sorted transitions in data | `build_trie.py` sorts by character, ready for binary search |

### Findings

| # | Finding | Severity | Location |
|---|---------|----------|----------|
| 1 | Linear scan on transitions instead of binary search | Medium | `lookup_kernel.metal`, all 3 kernels |
| 2 | `build_trie.py` outputs single `trie.bin`; ObjC expects 3 separate `.bin` files | Bug | `build_trie.py:80` vs `AnalyzerMetal.m:48–50` |
| 3 | Fixed-stride layout wastes ~78% memory bandwidth on padding | Medium | `lookup_kernel` + `lemmatizeBatch:` |
| 4 | Per-element `@synchronized` in fixed-stride/packed decode | Low | `AnalyzerMetal.m:144–146`, `254–256` |
| 5 | No page-alignment check on `newBufferWithBytesNoCopy` pointers | Low | `AnalyzerMetal.m:52–54` |
| 6 | Per-batch `MTLBuffer` allocation (no buffer pool reuse) | Low | `AnalyzerMetal.m:100–102`, `210–212`, `320–322` |
| 7 | Pointer-chasing trie traversal limits GPU pipeline utilization | Inherent | All kernels |
| 8 | SIMD divergence from variable word lengths and early-break | Inherent | All kernels |
| 9 | Summed GPU times across overlapping batches can exceed wall time | Informational | `AnalyzerMetal.m:136–139` |
| 10 | Runtime Metal compilation adds startup latency (~100–300 ms) | Low | `AnalyzerMetal.m:30–34` |

### Optimization Opportunities

1. **Binary search for transitions** when `num_transitions > 8`, linear scan
   otherwise. The data is already sorted; only the kernel code needs to change.

2. **Remove `@synchronized`** from the per-element result writes in
   `lemmatizeBatch:` and `lemmatizeBatchPacked:`. The disjoint index ranges
   make locking unnecessary.

3. **Pre-compile `.metallib`** to eliminate runtime shader compilation overhead.

4. **Buffer pool** for input/output buffers to avoid per-batch allocation and
   deallocation.

5. **Word-length sorting** before dispatch: grouping words by similar length
   would reduce SIMD divergence from the variable-length outer loop. This adds
   CPU preprocessing cost but may improve GPU occupancy.

---

## 8. Architecture Verdict

The `--packed-col` path (`lookup_kernel_index` + CPU-side decode) is the
architecturally superior design:

- **Minimal GPU output bandwidth** (4 B/thread vs up to 37 B/thread).
- **Perfectly coalesced writes** (contiguous `int32[]`).
- **Leverages unified memory** for the decode pass — CPU reads the lemma buffer
  directly without any transfer.
- **Clean separation of concerns** — GPU does the trie traversal (its strength),
  CPU does variable-length string assembly (its strength).

The main remaining bottleneck across all kernel variants is the inherent
pointer-chasing nature of trie traversal, which limits instruction-level
parallelism within each thread. This is a fundamental property of the data
structure and not something that can be resolved at the kernel level alone.
