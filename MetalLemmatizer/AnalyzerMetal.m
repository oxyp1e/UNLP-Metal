#import "AnalyzerMetal.h"
#import <Metal/Metal.h>
#import <time.h>



@interface AnalyzerMetal ()
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelinePacked;
@property (nonatomic, strong) id<MTLComputePipelineState> pipelineIndex;
@property (nonatomic, strong) id<MTLCommandQueue> commandQueue;
@property (nonatomic, strong) id<MTLBuffer> statesBuffer;
@property (nonatomic, strong) id<MTLBuffer> transitionsBuffer;
@property (nonatomic, strong) id<MTLBuffer> lemmaBuffer;
// Retained to keep the backing memory alive for the NoCopy buffers above
@property (nonatomic, strong) NSData *statesData;
@property (nonatomic, strong) NSData *transitionsData;
@property (nonatomic, strong) NSData *lemmasData;
@end

@implementation AnalyzerMetal

- (instancetype)initWithDevice:(id<MTLDevice>)device {
    if (self = [super init]) {
        _device = device;
        _commandQueue = [device newCommandQueue];

        NSError *error = nil;
        NSString *kernelSource = [NSString stringWithContentsOfFile:@"lookup_kernel.metal" encoding:NSUTF8StringEncoding error:&error];
        if (error) { NSLog(@" Failed to load Metal source: %@", error); return nil; }

        id<MTLLibrary> library = [device newLibraryWithSource:kernelSource options:nil error:&error];
        if (error) { NSLog(@" Failed to compile Metal: %@", error); return nil; }

        id<MTLFunction> kernel = [library newFunctionWithName:@"lookup_kernel"];
        _pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
        if (error) { NSLog(@" Failed to create pipeline: %@", error); return nil; }

        id<MTLFunction> kernelPacked = [library newFunctionWithName:@"lookup_kernel_packed"];
        _pipelinePacked = [device newComputePipelineStateWithFunction:kernelPacked error:&error];
        if (error) { NSLog(@" Failed to create packed pipeline: %@", error); return nil; }

        id<MTLFunction> kernelIndex = [library newFunctionWithName:@"lookup_kernel_index"];
        _pipelineIndex = [device newComputePipelineStateWithFunction:kernelIndex error:&error];
        if (error) { NSLog(@" Failed to create index pipeline: %@", error); return nil; }

        _statesData = [NSData dataWithContentsOfFile:@"resources/gpu_states.bin"];
        _transitionsData = [NSData dataWithContentsOfFile:@"resources/gpu_transitions.bin"];
        _lemmasData = [NSData dataWithContentsOfFile:@"resources/gpu_lemmas.bin"];

        _statesBuffer = [device newBufferWithBytesNoCopy:(void *)_statesData.bytes length:_statesData.length options:MTLResourceStorageModeShared deallocator:nil];
        _transitionsBuffer = [device newBufferWithBytesNoCopy:(void *)_transitionsData.bytes length:_transitionsData.length options:MTLResourceStorageModeShared deallocator:nil];
        _lemmaBuffer = [device newBufferWithBytesNoCopy:(void *)_lemmasData.bytes length:_lemmasData.length options:MTLResourceStorageModeShared deallocator:nil];
    }
    return self;
}



- (NSArray<NSString *> *)lemmatizeBatch:(NSArray<NSString *> *)words
                               kernelMs:(double *)outKernelMs
                                 packMs:(double *)outPackMs
                                totalMs:(double *)outTotalMs {
    NSUInteger total = words.count;
    NSMutableArray<NSString *> *allResults = [NSMutableArray arrayWithCapacity:total];
    for (NSUInteger i = 0; i < total; i++) {
        [allResults addObject:@""];
    }

    dispatch_group_t dispatchGroup = dispatch_group_create();
    dispatch_semaphore_t semaphore = dispatch_semaphore_create([[NSProcessInfo processInfo] activeProcessorCount]);

    __block double gpuTimeAccumMs = 0.0;
    __block double packTimeAccumMs = 0.0;
    NSObject *gpuTimeLock = [NSObject new];

    NSUInteger BATCH_SIZE = 100000;

    struct timespec wallStart, wallEnd;
    clock_gettime(CLOCK_MONOTONIC, &wallStart);

    for (NSUInteger i = 0; i < total; i += BATCH_SIZE) {
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        dispatch_group_enter(dispatchGroup);

        NSRange range = NSMakeRange(i, MIN(BATCH_SIZE, total - i));
        NSArray<NSString *> *batch = [words subarrayWithRange:range];

        // --- PACK TIMER START ---
        struct timespec packStart, packEnd;
        clock_gettime(CLOCK_MONOTONIC, &packStart);

        NSUInteger max_word_len = 0;
        for (NSString *word in batch) {
            max_word_len = MAX(max_word_len, [word lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);
        }

        NSUInteger bufferSize = batch.count * max_word_len;
        id<MTLBuffer> inputBuffer = [self.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> maxWordLenBuffer = [self.device newBufferWithBytes:&max_word_len length:sizeof(max_word_len) options:MTLResourceStorageModeShared];

        char *inputWords = (char *)inputBuffer.contents;
        memset(inputWords, 0, bufferSize);
        for (NSUInteger j = 0; j < batch.count; ++j) {
            const char *cstr = [batch[j] UTF8String];
            strncpy(inputWords + j * max_word_len, cstr, max_word_len - 1);
        }

        clock_gettime(CLOCK_MONOTONIC, &packEnd);
        double batchPackMs = (packEnd.tv_sec - packStart.tv_sec) * 1000.0
                           + (packEnd.tv_nsec - packStart.tv_nsec) / 1e6;
        @synchronized (gpuTimeLock) { packTimeAccumMs += batchPackMs; }
        // --- PACK TIMER END ---

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:self.pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:self.statesBuffer offset:0 atIndex:1];
        [encoder setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [encoder setBuffer:self.lemmaBuffer offset:0 atIndex:3];
        [encoder setBuffer:outputBuffer offset:0 atIndex:4];
        [encoder setBuffer:maxWordLenBuffer offset:0 atIndex:5];

        MTLSize gridSize = MTLSizeMake(batch.count, 1, 1);
        NSUInteger w = self.pipeline.threadExecutionWidth;
        MTLSize threadgroupSize = MTLSizeMake(w, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double batchGpuMs = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0;
            @synchronized (gpuTimeLock) {
                gpuTimeAccumMs += batchGpuMs;
            }

            char *outputPtr = (char *)outputBuffer.contents;
            for (NSUInteger j = 0; j < batch.count; ++j) {
                NSString *s = [NSString stringWithUTF8String:outputPtr + j * max_word_len];
                @synchronized (allResults) {
                    allResults[range.location + j] = s ?: @"";
                }
            }
            dispatch_semaphore_signal(semaphore);
            dispatch_group_leave(dispatchGroup);
        }];

        [cmdBuf commit];
    }

    dispatch_group_wait(dispatchGroup, DISPATCH_TIME_FOREVER);

    clock_gettime(CLOCK_MONOTONIC, &wallEnd);
    double wallMs = (wallEnd.tv_sec - wallStart.tv_sec) * 1000.0
                  + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e6;

    if (outKernelMs) *outKernelMs = gpuTimeAccumMs;
    if (outPackMs)   *outPackMs   = packTimeAccumMs;
    if (outTotalMs)  *outTotalMs  = wallMs;

    return allResults;
}

- (NSArray<NSString *> *)lemmatizeBatchPacked:(NSArray<NSString *> *)words
                                     kernelMs:(double *)outKernelMs
                                       packMs:(double *)outPackMs
                                      totalMs:(double *)outTotalMs {
    NSUInteger total = words.count;
    NSMutableArray<NSString *> *allResults = [NSMutableArray arrayWithCapacity:total];
    for (NSUInteger i = 0; i < total; i++) [allResults addObject:@""];

    dispatch_group_t dispatchGroup = dispatch_group_create();
    dispatch_semaphore_t semaphore = dispatch_semaphore_create([[NSProcessInfo processInfo] activeProcessorCount]);

    __block double gpuTimeAccumMs = 0.0;
    __block double packTimeAccumMs = 0.0;
    NSObject *gpuTimeLock = [NSObject new];

    NSUInteger BATCH_SIZE = 100000;

    struct timespec wallStart, wallEnd;
    clock_gettime(CLOCK_MONOTONIC, &wallStart);

    for (NSUInteger i = 0; i < total; i += BATCH_SIZE) {
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        dispatch_group_enter(dispatchGroup);

        NSRange range = NSMakeRange(i, MIN(BATCH_SIZE, total - i));
        NSArray<NSString *> *batch = [words subarrayWithRange:range];
        NSUInteger count = batch.count;

        // --- PACK TIMER START ---
        struct timespec packStart, packEnd;
        clock_gettime(CLOCK_MONOTONIC, &packStart);

        // Build offsets and packed input buffer
        NSMutableData *offsetsData = [NSMutableData dataWithLength:(count + 1) * sizeof(uint32_t)];
        uint32_t *offsets = (uint32_t *)offsetsData.mutableBytes;
        offsets[0] = 0;
        for (NSUInteger j = 0; j < count; ++j) {
            NSUInteger byteLen = [batch[j] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1;
            offsets[j + 1] = offsets[j] + (uint32_t)byteLen;
        }
        NSUInteger packedSize = offsets[count];

        id<MTLBuffer> inputBuffer   = [self.device newBufferWithLength:packedSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer  = [self.device newBufferWithLength:packedSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> offsetsBuffer = [self.device newBufferWithBytes:offsets length:(count + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];

        char *inputPtr = (char *)inputBuffer.contents;
        for (NSUInteger j = 0; j < count; ++j) {
            const char *cstr = [batch[j] UTF8String];
            NSUInteger byteLen = offsets[j + 1] - offsets[j];
            memcpy(inputPtr + offsets[j], cstr, byteLen);
        }

        clock_gettime(CLOCK_MONOTONIC, &packEnd);
        double batchPackMs = (packEnd.tv_sec - packStart.tv_sec) * 1000.0
                           + (packEnd.tv_nsec - packStart.tv_nsec) / 1e6;
        @synchronized (gpuTimeLock) { packTimeAccumMs += batchPackMs; }
        // --- PACK TIMER END ---

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:self.pipelinePacked];
        [encoder setBuffer:inputBuffer   offset:0 atIndex:0];
        [encoder setBuffer:self.statesBuffer offset:0 atIndex:1];
        [encoder setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [encoder setBuffer:self.lemmaBuffer offset:0 atIndex:3];
        [encoder setBuffer:outputBuffer  offset:0 atIndex:4];
        [encoder setBuffer:offsetsBuffer offset:0 atIndex:5];

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(self.pipelinePacked.threadExecutionWidth, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        NSData *offsetsSnapshot = [offsetsData copy];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double batchGpuMs = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0;
            @synchronized (gpuTimeLock) { gpuTimeAccumMs += batchGpuMs; }

            const uint32_t *off = (const uint32_t *)offsetsSnapshot.bytes;
            const char *outputPtr = (const char *)outputBuffer.contents;
            for (NSUInteger j = 0; j < count; ++j) {
                NSString *s = [NSString stringWithUTF8String:outputPtr + off[j]];
                @synchronized (allResults) {
                    allResults[range.location + j] = s ?: @"";
                }
            }
            dispatch_semaphore_signal(semaphore);
            dispatch_group_leave(dispatchGroup);
        }];

        [cmdBuf commit];
    }

    dispatch_group_wait(dispatchGroup, DISPATCH_TIME_FOREVER);

    clock_gettime(CLOCK_MONOTONIC, &wallEnd);
    double wallMs = (wallEnd.tv_sec - wallStart.tv_sec) * 1000.0
                  + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e6;

    if (outKernelMs) *outKernelMs = gpuTimeAccumMs;
    if (outPackMs)   *outPackMs   = packTimeAccumMs;
    if (outTotalMs)  *outTotalMs  = wallMs;

    return allResults;
}

- (NSArray<NSString *> *)lemmatizeBatchPackedColumn:(NSArray<NSString *> *)words
                                           kernelMs:(double *)outKernelMs
                                             packMs:(double *)outPackMs
                                            totalMs:(double *)outTotalMs {
    NSUInteger total = words.count;
    NSMutableArray<NSString *> *allResults = [NSMutableArray arrayWithCapacity:total];
    for (NSUInteger i = 0; i < total; i++) [allResults addObject:@""];

    dispatch_group_t dispatchGroup = dispatch_group_create();
    dispatch_semaphore_t semaphore = dispatch_semaphore_create([[NSProcessInfo processInfo] activeProcessorCount]);

    __block double gpuTimeAccumMs = 0.0;
    __block double packTimeAccumMs = 0.0;
    NSObject *gpuTimeLock = [NSObject new];

    NSUInteger BATCH_SIZE = 100000;

    struct timespec wallStart, wallEnd;
    clock_gettime(CLOCK_MONOTONIC, &wallStart);

    for (NSUInteger i = 0; i < total; i += BATCH_SIZE) {
        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
        dispatch_group_enter(dispatchGroup);

        NSRange range = NSMakeRange(i, MIN(BATCH_SIZE, total - i));
        NSArray<NSString *> *batch = [words subarrayWithRange:range];
        NSUInteger count = batch.count;

        // --- PACK TIMER START ---
        struct timespec packStart, packEnd;
        clock_gettime(CLOCK_MONOTONIC, &packStart);

        // Build packed input + offsets[N+1]
        NSMutableData *offsetsData = [NSMutableData dataWithLength:(count + 1) * sizeof(uint32_t)];
        uint32_t *offsets = (uint32_t *)offsetsData.mutableBytes;
        offsets[0] = 0;
        for (NSUInteger j = 0; j < count; ++j) {
            NSUInteger byteLen = [batch[j] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1;
            offsets[j + 1] = offsets[j] + (uint32_t)byteLen;
        }
        NSUInteger packedSize = offsets[count];

        id<MTLBuffer> inputBuffer   = [self.device newBufferWithLength:packedSize options:MTLResourceStorageModeShared];
        id<MTLBuffer> offsetsBuffer = [self.device newBufferWithBytes:offsets length:(count + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> indicesBuffer = [self.device newBufferWithLength:count * sizeof(int32_t) options:MTLResourceStorageModeShared];

        char *inputPtr = (char *)inputBuffer.contents;
        for (NSUInteger j = 0; j < count; ++j) {
            const char *cstr = [batch[j] UTF8String];
            NSUInteger byteLen = offsets[j + 1] - offsets[j];
            memcpy(inputPtr + offsets[j], cstr, byteLen);
        }

        clock_gettime(CLOCK_MONOTONIC, &packEnd);
        double batchPackMs = (packEnd.tv_sec - packStart.tv_sec) * 1000.0
                           + (packEnd.tv_nsec - packStart.tv_nsec) / 1e6;
        @synchronized (gpuTimeLock) { packTimeAccumMs += batchPackMs; }
        // --- PACK TIMER END ---

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:self.pipelineIndex];
        [encoder setBuffer:inputBuffer   offset:0 atIndex:0];
        [encoder setBuffer:self.statesBuffer offset:0 atIndex:1];
        [encoder setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [encoder setBuffer:indicesBuffer offset:0 atIndex:3];
        [encoder setBuffer:offsetsBuffer offset:0 atIndex:4];

        MTLSize gridSize = MTLSizeMake(count, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(self.pipelineIndex.threadExecutionWidth, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        NSData *offsetsSnapshot = [offsetsData copy];
        const char *lemmaBytes = (const char *)self.lemmasData.bytes;

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
            double batchGpuMs = (buffer.GPUEndTime - buffer.GPUStartTime) * 1000.0;
            @synchronized (gpuTimeLock) { gpuTimeAccumMs += batchGpuMs; }

            const int32_t  *indices  = (const int32_t *)indicesBuffer.contents;
            const uint32_t *off      = (const uint32_t *)offsetsSnapshot.bytes;
            const char     *inputRaw = (const char *)inputBuffer.contents;

            // Compute output offsets (prefix sum of lemma lengths)
            NSMutableData *outOffsetsData = [NSMutableData dataWithLength:(count + 1) * sizeof(uint32_t)];
            uint32_t *outOffsets = (uint32_t *)outOffsetsData.mutableBytes;
            outOffsets[0] = 0;
            for (NSUInteger j = 0; j < count; ++j) {
                const char *src = (indices[j] >= 0) ? (lemmaBytes + indices[j]) : (inputRaw + off[j]);
                outOffsets[j + 1] = outOffsets[j] + (uint32_t)(strlen(src) + 1);
            }

            // Allocate packed output and memcpy each lemma
            NSMutableData *outData = [NSMutableData dataWithLength:outOffsets[count]];
            char *outPtr = (char *)outData.mutableBytes;
            for (NSUInteger j = 0; j < count; ++j) {
                const char *src = (indices[j] >= 0) ? (lemmaBytes + indices[j]) : (inputRaw + off[j]);
                memcpy(outPtr + outOffsets[j], src, outOffsets[j + 1] - outOffsets[j]);
            }

            // Extract NSStrings
            @synchronized (allResults) {
                for (NSUInteger j = 0; j < count; ++j) {
                    NSString *s = [NSString stringWithUTF8String:outPtr + outOffsets[j]];
                    allResults[range.location + j] = s ?: @"";
                }
            }
            dispatch_semaphore_signal(semaphore);
            dispatch_group_leave(dispatchGroup);
        }];

        [cmdBuf commit];
    }

    dispatch_group_wait(dispatchGroup, DISPATCH_TIME_FOREVER);

    clock_gettime(CLOCK_MONOTONIC, &wallEnd);
    double wallMs = (wallEnd.tv_sec - wallStart.tv_sec) * 1000.0
                  + (wallEnd.tv_nsec - wallStart.tv_nsec) / 1e6;

    if (outKernelMs) *outKernelMs = gpuTimeAccumMs;
    if (outPackMs)   *outPackMs   = packTimeAccumMs;
    if (outTotalMs)  *outTotalMs  = wallMs;

    return allResults;
}

// ---------------------------------------------------------------------------
// Loop benchmarks — no D2H, no result decoding, pure throughput measurement
// ---------------------------------------------------------------------------

static void _benchPrintFinal(int numIters, NSUInteger numWords,
                              double totalKernelMs, double peakKernelMs) {
    double avgMs = totalKernelMs / numIters;
    double tp    = (double)numWords * numIters / (totalKernelMs / 1000.0);
    fprintf(stderr, "\n=== Final ===\n");
    fprintf(stderr, "  Iters:       %d\n",       numIters);
    fprintf(stderr, "  Words/iter:  %lu\n",      (unsigned long)numWords);
    fprintf(stderr, "  Avg kernel:  %.3f ms\n",  avgMs);
    fprintf(stderr, "  Peak kernel: %.3f ms\n",  peakKernelMs);
    fprintf(stderr, "  Throughput:  %.2fM words/sec\n", tp / 1e6);
}

- (void)benchLoopFixedStride:(NSArray<NSString *> *)words duration:(double)seconds {
    NSUInteger count = words.count;
    if (count == 0) { fprintf(stderr, "No words.\n"); return; }

    NSUInteger maxWordLen = 0;
    for (NSString *w in words)
        maxWordLen = MAX(maxWordLen, [w lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1);

    // --- PACK TIMER START ---
    struct timespec packStart, packEnd;
    clock_gettime(CLOCK_MONOTONIC, &packStart);

    id<MTLBuffer> inputBuffer  = [self.device newBufferWithLength:count * maxWordLen options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [self.device newBufferWithLength:count * maxWordLen options:MTLResourceStorageModeShared];
    id<MTLBuffer> maxLenBuf    = [self.device newBufferWithBytes:&maxWordLen length:sizeof(maxWordLen) options:MTLResourceStorageModeShared];

    char *inputPtr = (char *)inputBuffer.contents;
    memset(inputPtr, 0, count * maxWordLen);
    for (NSUInteger j = 0; j < count; ++j)
        strncpy(inputPtr + j * maxWordLen, [words[j] UTF8String], maxWordLen - 1);

    clock_gettime(CLOCK_MONOTONIC, &packEnd);
    double packMs = (packEnd.tv_sec - packStart.tv_sec) * 1000.0
                  + (packEnd.tv_nsec - packStart.tv_nsec) / 1e6;
    // --- PACK TIMER END ---

    MTLSize gridSize        = MTLSizeMake(count, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(self.pipeline.threadExecutionWidth, 1, 1);

    int numIters = 0;
    double totalKernelMs = 0.0, peakKernelMs = 0.0;

    fprintf(stderr, "Pack (build MTLBuffer + strncpy):           %.3f ms\n", packMs);
    fprintf(stderr, "Running fixed-stride loop for %.1fs  words=%lu\n", seconds, (unsigned long)count);

    struct timespec wallStart;
    clock_gettime(CLOCK_MONOTONIC, &wallStart);

    while (true) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - wallStart.tv_sec) + (now.tv_nsec - wallStart.tv_nsec) / 1e9;
        if (elapsed >= seconds) break;

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:self.pipeline];
        [enc setBuffer:inputBuffer            offset:0 atIndex:0];
        [enc setBuffer:self.statesBuffer      offset:0 atIndex:1];
        [enc setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [enc setBuffer:self.lemmaBuffer       offset:0 atIndex:3];
        [enc setBuffer:outputBuffer           offset:0 atIndex:4];
        [enc setBuffer:maxLenBuf              offset:0 atIndex:5];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        double ms = (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
        totalKernelMs += ms;
        if (ms > peakKernelMs) peakKernelMs = ms;
        ++numIters;

        if (numIters % 100 == 0) {
            double tp = (double)count * numIters / (totalKernelMs / 1000.0);
            fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
                    numIters, totalKernelMs / numIters, tp / 1e6);
        }
    }
    _benchPrintFinal(numIters, count, totalKernelMs, peakKernelMs);
}

- (void)benchLoopPacked:(NSArray<NSString *> *)words duration:(double)seconds {
    NSUInteger count = words.count;
    if (count == 0) { fprintf(stderr, "No words.\n"); return; }

    // --- PACK TIMER START ---
    struct timespec packStart2, packEnd2;
    clock_gettime(CLOCK_MONOTONIC, &packStart2);

    uint32_t *offsets = (uint32_t *)malloc((count + 1) * sizeof(uint32_t));
    offsets[0] = 0;
    for (NSUInteger j = 0; j < count; ++j) {
        NSUInteger byteLen = [words[j] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1;
        offsets[j + 1] = offsets[j] + (uint32_t)byteLen;
    }
    NSUInteger packedSize = offsets[count];

    id<MTLBuffer> inputBuffer   = [self.device newBufferWithLength:packedSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer  = [self.device newBufferWithLength:packedSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsetsBuffer = [self.device newBufferWithBytes:offsets length:(count + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    free(offsets);

    char *inputPtr = (char *)inputBuffer.contents;
    const uint32_t *off = (const uint32_t *)offsetsBuffer.contents;
    for (NSUInteger j = 0; j < count; ++j) {
        NSUInteger byteLen = off[j + 1] - off[j];
        memcpy(inputPtr + off[j], [words[j] UTF8String], byteLen);
    }

    clock_gettime(CLOCK_MONOTONIC, &packEnd2);
    double packMs2 = (packEnd2.tv_sec - packStart2.tv_sec) * 1000.0
                   + (packEnd2.tv_nsec - packStart2.tv_nsec) / 1e6;
    // --- PACK TIMER END ---

    MTLSize gridSize        = MTLSizeMake(count, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(self.pipelinePacked.threadExecutionWidth, 1, 1);

    int numIters = 0;
    double totalKernelMs = 0.0, peakKernelMs = 0.0;

    fprintf(stderr, "Pack (build MTLBuffer + memcpy):            %.3f ms\n", packMs2);
    fprintf(stderr, "Running packed loop for %.1fs  words=%lu\n", seconds, (unsigned long)count);

    struct timespec wallStart;
    clock_gettime(CLOCK_MONOTONIC, &wallStart);

    while (true) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - wallStart.tv_sec) + (now.tv_nsec - wallStart.tv_nsec) / 1e9;
        if (elapsed >= seconds) break;

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:self.pipelinePacked];
        [enc setBuffer:inputBuffer            offset:0 atIndex:0];
        [enc setBuffer:self.statesBuffer      offset:0 atIndex:1];
        [enc setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [enc setBuffer:self.lemmaBuffer       offset:0 atIndex:3];
        [enc setBuffer:outputBuffer           offset:0 atIndex:4];
        [enc setBuffer:offsetsBuffer          offset:0 atIndex:5];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        double ms = (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
        totalKernelMs += ms;
        if (ms > peakKernelMs) peakKernelMs = ms;
        ++numIters;

        if (numIters % 100 == 0) {
            double tp = (double)count * numIters / (totalKernelMs / 1000.0);
            fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
                    numIters, totalKernelMs / numIters, tp / 1e6);
        }
    }
    _benchPrintFinal(numIters, count, totalKernelMs, peakKernelMs);
}

- (void)benchLoopPackedColumn:(NSArray<NSString *> *)words duration:(double)seconds {
    NSUInteger count = words.count;
    if (count == 0) { fprintf(stderr, "No words.\n"); return; }

    // --- PACK TIMER START ---
    struct timespec packStart3, packEnd3;
    clock_gettime(CLOCK_MONOTONIC, &packStart3);

    uint32_t *offsets = (uint32_t *)malloc((count + 1) * sizeof(uint32_t));
    offsets[0] = 0;
    for (NSUInteger j = 0; j < count; ++j) {
        NSUInteger byteLen = [words[j] lengthOfBytesUsingEncoding:NSUTF8StringEncoding] + 1;
        offsets[j + 1] = offsets[j] + (uint32_t)byteLen;
    }
    NSUInteger packedSize = offsets[count];

    id<MTLBuffer> inputBuffer   = [self.device newBufferWithLength:packedSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> offsetsBuffer = [self.device newBufferWithBytes:offsets length:(count + 1) * sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> indicesBuffer = [self.device newBufferWithLength:count * sizeof(int32_t) options:MTLResourceStorageModeShared];
    free(offsets);

    char *inputPtr = (char *)inputBuffer.contents;
    const uint32_t *off = (const uint32_t *)offsetsBuffer.contents;
    for (NSUInteger j = 0; j < count; ++j) {
        NSUInteger byteLen = off[j + 1] - off[j];
        memcpy(inputPtr + off[j], [words[j] UTF8String], byteLen);
    }

    clock_gettime(CLOCK_MONOTONIC, &packEnd3);
    double packMs3 = (packEnd3.tv_sec - packStart3.tv_sec) * 1000.0
                   + (packEnd3.tv_nsec - packStart3.tv_nsec) / 1e6;
    // --- PACK TIMER END ---

    MTLSize gridSize        = MTLSizeMake(count, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(self.pipelineIndex.threadExecutionWidth, 1, 1);

    int numIters = 0;
    double totalKernelMs = 0.0, peakKernelMs = 0.0;

    fprintf(stderr, "Pack (build MTLBuffer + memcpy):            %.3f ms\n", packMs3);
    fprintf(stderr, "Running packed-col loop for %.1fs  words=%lu\n", seconds, (unsigned long)count);

    struct timespec wallStart;
    clock_gettime(CLOCK_MONOTONIC, &wallStart);

    while (true) {
        struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - wallStart.tv_sec) + (now.tv_nsec - wallStart.tv_nsec) / 1e9;
        if (elapsed >= seconds) break;

        id<MTLCommandBuffer> cmdBuf = [self.commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:self.pipelineIndex];
        [enc setBuffer:inputBuffer            offset:0 atIndex:0];
        [enc setBuffer:self.statesBuffer      offset:0 atIndex:1];
        [enc setBuffer:self.transitionsBuffer offset:0 atIndex:2];
        [enc setBuffer:indicesBuffer          offset:0 atIndex:3];
        [enc setBuffer:offsetsBuffer          offset:0 atIndex:4];
        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        double ms = (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
        totalKernelMs += ms;
        if (ms > peakKernelMs) peakKernelMs = ms;
        ++numIters;

        if (numIters % 100 == 0) {
            double tp = (double)count * numIters / (totalKernelMs / 1000.0);
            fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
                    numIters, totalKernelMs / numIters, tp / 1e6);
        }
    }
    _benchPrintFinal(numIters, count, totalKernelMs, peakKernelMs);
}

@end
