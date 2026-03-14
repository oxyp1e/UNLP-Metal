CC = clang
CFLAGS = -framework Foundation -O2

all: trie_cpu_test lemmatizer lemmatizer_raw

trie_cpu_test: trie_cpu_test.m
	$(CC) $(CFLAGS) trie_cpu_test.m -o trie_cpu_test

lemmatizer: main.m AnalyzerMetal.m AnalyzerMetal.h lookup_kernel.metal
	$(CC) $(CFLAGS) -framework Metal main.m AnalyzerMetal.m -o lemmatizer

lemmatizer_raw: ../MetalLemmatizerRaw/main.m AnalyzerMetal.m AnalyzerMetal.h lookup_kernel.metal
	$(CC) $(CFLAGS) -framework Metal -I . ../MetalLemmatizerRaw/main.m AnalyzerMetal.m -o lemmatizer_raw

clean:
	rm -f trie_cpu_test lemmatizer lemmatizer_raw

run: trie_cpu_test
	./trie_cpu_test running jumps walked

.PHONY: all clean run
