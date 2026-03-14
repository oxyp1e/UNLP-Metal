//
//  main.m
//  MetalLemmatizerRaw
//
//  Zero-NSString, zero-COW hot path:
//    mmap (read-only) → scan \n for offsets → GPU
//  The file is never written to; no page faults beyond the initial read.
//

#import <Foundation/Foundation.h>
#import <time.h>
#import <stdlib.h>
#import "AnalyzerMetal.h"

static inline double ms_diff(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <input_file.txt>\n", argv[0]);
            return 1;
        }

        NSString *inputFileName = [NSString stringWithUTF8String:argv[1]];
        NSString *path;
        if ([inputFileName hasPrefix:@"/"]) {
            path = inputFileName;
        } else {
            path = [[[NSFileManager defaultManager] currentDirectoryPath]
                    stringByAppendingPathComponent:inputFileName];
        }
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            fprintf(stderr, "Error: file not found: %s\n", [path UTF8String]);
            return 1;
        }

        // --- PREPROCESS: read-only mmap (no COW, no writes ever) ---
        struct timespec preprocessStart, preprocessEnd;
        clock_gettime(CLOCK_MONOTONIC, &preprocessStart);

        NSError *readError = nil;
        NSData *fileData = [NSData dataWithContentsOfFile:path
                                                  options:NSDataReadingMappedIfSafe
                                                    error:&readError];
        if (!fileData || readError) {
            fprintf(stderr, "Failed to read file: %s\n", [path UTF8String]);
            if (readError) fprintf(stderr, "Error: %s\n", [[readError localizedDescription] UTF8String]);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &preprocessEnd);
        double preprocessMs = ms_diff(preprocessStart, preprocessEnd);

        // --- PACK: scan \n to build offsets — no writes to the mmap ---
        struct timespec packStart, packEnd;
        clock_gettime(CLOCK_MONOTONIC, &packStart);

        const char *bytes = (const char *)fileData.bytes;
        NSUInteger fileLen = fileData.length;

        // Single pass: scan for \n, fill offsets directly.
        //
        // offsets[j]   = byte start of line j
        // offsets[j+1] = byte start of line j+1 (one past the \n)
        // slot_len = offsets[j+1] - offsets[j] = word_len + 1
        //
        // lookup_kernel_index loops (slot_len - 1) times — exactly the word
        // bytes, never touching the \n.  Empty lines: slot_len = 1 → 0
        // iterations → lemma_offset = -1 (identity fallback).
        //
        // Estimate capacity from file size (avg ~10 bytes/line).
        // If the estimate is low, realloc doubles it — rare in practice.
        NSUInteger capacity = fileLen / 10 + 1024;
        uint32_t *offsets = (uint32_t *)malloc((capacity + 1) * sizeof(uint32_t));
        if (!offsets) { fprintf(stderr, "OOM allocating offsets\n"); return 1; }

        NSUInteger lineCount = 0;
        offsets[0] = 0;
        for (NSUInteger i = 0; i < fileLen; i++) {
            if (bytes[i] == '\n') {
                lineCount++;
                if (__builtin_expect(lineCount > capacity, 0)) {
                    capacity *= 2;
                    uint32_t *tmp = realloc(offsets, (capacity + 1) * sizeof(uint32_t));
                    if (!tmp) { free(offsets); fprintf(stderr, "OOM growing offsets\n"); return 1; }
                    offsets = tmp;
                }
                offsets[lineCount] = (uint32_t)(i + 1);
            }
        }
        // Handle last line with no trailing \n (virtual sentinel)
        if (fileLen > 0 && bytes[fileLen - 1] != '\n') {
            lineCount++;
            if (lineCount > capacity) {
                uint32_t *tmp = realloc(offsets, (lineCount + 1) * sizeof(uint32_t));
                if (!tmp) { free(offsets); fprintf(stderr, "OOM growing offsets\n"); return 1; }
                offsets = tmp;
            }
            offsets[lineCount] = (uint32_t)(fileLen + 1);
        }

        if (lineCount == 0) {
            fprintf(stderr, "No lines found in input.\n");
            free(offsets);
            return 1;
        }

        clock_gettime(CLOCK_MONOTONIC, &packEnd);
        double packMs = ms_diff(packStart, packEnd);

        // --- GPU ---
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this device.\n");
            free(offsets);
            return 1;
        }

        AnalyzerMetal *analyzer = [[AnalyzerMetal alloc] initWithDevice:device];

        double kernelMs = 0.0, gpuPackMs = 0.0, totalMs = 0.0;
        NSUInteger processed = [analyzer lemmatizeBatchPackedColumnRaw:bytes
                                                               offsets:offsets
                                                                 count:lineCount
                                                             kernelMs:&kernelMs
                                                               packMs:&gpuPackMs
                                                              totalMs:&totalMs];
        free(offsets);

        double throughput = processed / (kernelMs / 1000.0);
        fprintf(stderr, "[raw-packed-col] Words: %lu\n", (unsigned long)processed);
        fprintf(stderr, "  Preprocess (mmap file read):                %.3f ms\n", preprocessMs);
        fprintf(stderr, "  Pack (scan \\n + build offsets, CPU):        %.3f ms\n", packMs);
        fprintf(stderr, "  Pack (MTLBuffer wrap + alloc, GPU-side):    %.3f ms\n", gpuPackMs);
        fprintf(stderr, "  Kernel:                                     %.3f ms  (%.0f words/sec)\n", kernelMs, throughput);
        fprintf(stderr, "  GPU total (wrap+kernel):                    %.3f ms\n", totalMs);
        fprintf(stderr, "  End-to-end (mmap+pack+GPU):                 %.3f ms\n", preprocessMs + packMs + totalMs);
    }
    return EXIT_SUCCESS;
}
