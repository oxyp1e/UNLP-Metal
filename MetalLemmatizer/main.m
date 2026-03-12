//
//  main.m
//  MetalLemmatizer
//
//  Created by Illia Fedorovych on 12.03.2026.
//

#import <Foundation/Foundation.h>
#import <time.h>
#import "AnalyzerMetal.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s [--packed|--packed-col] <input_file.txt>\n", argv[0]);
            return 1;
        }

        BOOL usePacked    = (argc >= 3) && (strcmp(argv[1], "--packed") == 0);
        BOOL usePackedCol = (argc >= 3) && (strcmp(argv[1], "--packed-col") == 0);
        BOOL hasFlag = usePacked || usePackedCol;
        // --- PREPROCESS TIMER START (file I/O + tokenize + filter) ---
        struct timespec preprocessStart, preprocessEnd;
        clock_gettime(CLOCK_MONOTONIC, &preprocessStart);

        NSString *path = [NSString stringWithUTF8String:hasFlag ? argv[2] : argv[1]];
        NSError *readError = nil;
        NSString *content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&readError];

        if (readError || !content) {
            fprintf(stderr, "Failed to read file: %s\n", argv[1]);
            return 1;
        }

        NSArray<NSString *> *words = [content componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
        words = [words filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"length > 0"]];

        clock_gettime(CLOCK_MONOTONIC, &preprocessEnd);
        double preprocessMs = (preprocessEnd.tv_sec - preprocessStart.tv_sec) * 1000.0
                            + (preprocessEnd.tv_nsec - preprocessStart.tv_nsec) / 1e6;
        // --- PREPROCESS TIMER END ---

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal is not supported on this device.\n");
            return 1;
        }

        AnalyzerMetal *analyzer = [[AnalyzerMetal alloc] initWithDevice:device];

        double kernelMs = 0.0, packMs = 0.0, totalMs = 0.0;
        NSArray<NSString *> *results;
        if (usePackedCol) {
            results = [analyzer lemmatizeBatchPackedColumn:words kernelMs:&kernelMs packMs:&packMs totalMs:&totalMs];
        } else if (usePacked) {
            results = [analyzer lemmatizeBatchPacked:words kernelMs:&kernelMs packMs:&packMs totalMs:&totalMs];
        } else {
            results = [analyzer lemmatizeBatch:words kernelMs:&kernelMs packMs:&packMs totalMs:&totalMs];
        }

        for (NSUInteger i = 0; i < results.count; i++) {
            printf("%s → %s\n", [words[i] UTF8String], [results[i] UTF8String]);
        }

        double throughput = results.count / (kernelMs / 1000.0);
        const char *label = usePackedCol ? "packed-col" : (usePacked ? "packed" : "fixed-stride");
        double decodeMs = totalMs - packMs - kernelMs;  // approximate (async overlap)
        fprintf(stderr, "[%s] Words: %lu\n", label, (unsigned long)results.count);
        fprintf(stderr, "  Preprocess (file I/O + tokenize + filter):  %.3f ms\n", preprocessMs);
        fprintf(stderr, "  Pack (CPU→MTLBuffer, H2D equivalent):       %.3f ms\n", packMs);
        fprintf(stderr, "  Kernel:                                     %.3f ms  (%.0f words/sec)\n", kernelMs, throughput);
        fprintf(stderr, "  Decode+dispatch overhead (approx):          %.3f ms\n", decodeMs);
        fprintf(stderr, "  GPU total (pack+kernel+decode):             %.3f ms\n", totalMs);
        fprintf(stderr, "  End-to-end (preprocess+GPU):                %.3f ms\n", preprocessMs + totalMs);
    }
    return EXIT_SUCCESS;
}
