#import <Foundation/Foundation.h>
#import <mach/mach_time.h>

#define MAX_WORD_LEN 37

typedef struct {
    uint32_t transition_start_idx;
    uint32_t num_transitions;
    int32_t lemma_offset;
} GpuState;

typedef struct {
    char c;
    uint32_t next_state;
} GpuTransition;

NSData *loadFile(NSString *path) {
    return [NSData dataWithContentsOfFile:path];
}

const char *lookupLemma(const char *word, const GpuState *states, const GpuTransition *transitions, const char *lemmas) {
    int state = 0;

    for (int j = 0; j < MAX_WORD_LEN && word[j] != '\0'; ++j) {
        char ch = word[j];
        const GpuState s = states[state];
        int found = 0;

        for (uint32_t k = 0; k < s.num_transitions; ++k) {
            GpuTransition t = transitions[s.transition_start_idx + k];
            if (t.c == ch) {
                state = t.next_state;
                found = 1;
                break;
            }
        }

        if (!found) return NULL;
    }

    int32_t offset = states[state].lemma_offset;
    return (offset >= 0) ? (lemmas + offset) : NULL;
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            NSLog(@"Usage: ./trie_cpu_test <input_file.txt> | ./trie_cpu_test word1 word2 ...");
            return 1;
        }

        NSData *statesData = loadFile(@"resources/gpu_states.bin");
        NSData *transData = loadFile(@"resources/gpu_transitions.bin");
        NSData *lemmasData = loadFile(@"resources/gpu_lemmas.bin");

        const GpuState *states = (const GpuState *)statesData.bytes;
        const GpuTransition *transitions = (const GpuTransition *)transData.bytes;
        const char *lemmas = (const char *)lemmasData.bytes;

        uint64_t start = mach_absolute_time();

        // If a single argument is given and it's an existing file, read words from it
        BOOL isFile = (argc == 2) && [[NSFileManager defaultManager] fileExistsAtPath:[NSString stringWithUTF8String:argv[1]]];

        if (isFile) {
            NSString *path = [NSString stringWithUTF8String:argv[1]];
            NSString *content = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
            NSArray<NSString *> *words = [[content componentsSeparatedByCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]]
                                          filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"length > 0"]];
            NSUInteger count = words.count;
            NSMutableArray<NSString *> *results = [NSMutableArray arrayWithCapacity:count];
            for (NSUInteger i = 0; i < count; i++) [results addObject:@""];

            dispatch_apply(count, DISPATCH_APPLY_AUTO, ^(size_t i) {
                const char *word = [words[i] UTF8String];
                const char *lemma = lookupLemma(word, states, transitions, lemmas);
                results[i] = lemma ? [NSString stringWithUTF8String:lemma] : @"not found in trie";
            });

            for (NSUInteger i = 0; i < count; i++) {
                printf("%s → %s\n", [words[i] UTF8String], [results[i] UTF8String]);
            }
        } else {
            for (int i = 1; i < argc; ++i) {
                const char *lemma = lookupLemma(argv[i], states, transitions, lemmas);
                printf("%s → %s\n", argv[i], lemma ? lemma : "not found in trie");
            }
        }

        uint64_t end = mach_absolute_time();

        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        uint64_t elapsed_ns = (end - start) * info.numer / info.denom;
        printf("CPU time: %.3f ms\n", elapsed_ns / 1e6);
    }
    return 0;
}

