#import <Foundation/Foundation.h>
#import <Metal/Metal.h>


@interface AnalyzerMetal : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (NSArray<NSString *> *)lemmatizeBatch:(NSArray<NSString *> *)words
                               kernelMs:(double *)outKernelMs
                                 packMs:(double *)outPackMs
                                totalMs:(double *)outTotalMs;

- (NSArray<NSString *> *)lemmatizeBatchPacked:(NSArray<NSString *> *)words
                                     kernelMs:(double *)outKernelMs
                                       packMs:(double *)outPackMs
                                      totalMs:(double *)outTotalMs;

- (NSArray<NSString *> *)lemmatizeBatchPackedColumn:(NSArray<NSString *> *)words
                                           kernelMs:(double *)outKernelMs
                                             packMs:(double *)outPackMs
                                            totalMs:(double *)outTotalMs;

// Loop benchmarks — upload once, hammer kernel until `seconds` wall-clock time
- (void)benchLoopFixedStride:(NSArray<NSString *> *)words duration:(double)seconds;
- (void)benchLoopPacked:(NSArray<NSString *> *)words duration:(double)seconds;
- (void)benchLoopPackedColumn:(NSArray<NSString *> *)words duration:(double)seconds;

@end
