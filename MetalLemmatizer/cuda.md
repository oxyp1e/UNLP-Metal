● Here is a precise diff description for a Claude Code agent to apply to a Metal bench_gpu-style file:

  ---
  Changes from bench_gpu → bench_gpu_loop

  1. Command-line argument: output file → duration

  Remove the output_file argument and the has_output / output_path logic.
  Add a run_duration_s float argument (default 10.0):

  // before
  const bool has_output = (argc >= 3);
  const std::string output_path = has_output ? argv[2] : "";

  // after
  double run_duration_s = (argc >= 3) ? atof(argv[2]) : 10.0;

  2. Remove empty-input early-out that writes blank lines

  The loop benchmark doesn't produce output, so the block that opens fout and writes newlines for empty input should be replaced by a simple return 1 / error.

  3. Keep H2D uploads outside the loop (already the case in both)

  Both files upload trie data and input chars to device once before the loop. No change needed here.

  4. Remove D2H copy and result decoding

  Delete everything after cudaEventSynchronize(ev_stop) that copies results back, decodes GPU pointers into lemma strings, and writes output to file/stdout.

  5. Replace single kernel call with a timed loop

  Before (single shot):
  cudaEventRecord(ev_start);
  lookup_kernel<<<blocks, threads>>>(...);
  cudaEventRecord(ev_stop);
  cudaEventSynchronize(ev_stop);

  float kernel_ms = 0.f;
  cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  After (loop):
  int num_iters = 0;
  double total_kernel_ms = 0.0, peak_kernel_ms = 0.0;

  std::cerr << "Running for " << run_duration_s << "s  words=" << num_words
            << "  blocks=" << blocks << "  threads=" << threads << "\n";

  auto wall_start = std::chrono::high_resolution_clock::now();
  while (true) {
      double elapsed = std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - wall_start).count();
      if (elapsed >= run_duration_s) break;

      cudaEventRecord(ev_start);
      lookup_kernel<<<blocks, threads>>>(...);
      cudaEventRecord(ev_stop);
      cudaEventSynchronize(ev_stop);

      float ms = 0.f;
      cudaEventElapsedTime(&ms, ev_start, ev_stop);
      total_kernel_ms += ms;
      if (ms > peak_kernel_ms) peak_kernel_ms = ms;
      ++num_iters;

      if (num_iters % 100 == 0) {
          double tp = (double)num_words * num_iters / (total_kernel_ms / 1000.0);
          fprintf(stderr, "iter %5d  avg %.3f ms  throughput %.2fM words/sec\n",
                  num_iters, total_kernel_ms / num_iters, tp / 1e6);
      }
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  6. Replace final reporting with loop summary

  Before:
  fprintf(stderr, "Words: %d  Kernel: %.3f ms  Total: %.3f ms  Throughput: %lld words/sec\n", ...);
  // ... write output file ...

  After:
  double avg_ms = total_kernel_ms / num_iters;
  double tp = (double)num_words * num_iters / (total_kernel_ms / 1000.0);
  fprintf(stderr, "\n=== Final ===\n");
  fprintf(stderr, "  Iters:       %d\n",    num_iters);
  fprintf(stderr, "  Words/iter:  %d\n",    num_words);
  fprintf(stderr, "  Avg kernel:  %.3f ms\n", avg_ms);
  fprintf(stderr, "  Peak kernel: %.3f ms\n", peak_kernel_ms);
  fprintf(stderr, "  Throughput:  %.2fM words/sec\n", tp / 1e6);

  ---
  Summary of intent

  bench_gpu measures one-shot latency (kernel + H2D + D2H) and produces correct output.
  bench_gpu_loop measures sustained kernel throughput by holding all data on-device permanently and hammering the kernel repeatedly until a wall-clock deadline, accumulating avg/peak timing and reporting words/sec. No correctness check, no output file — pure throughput measurement.
