#!/bin/bash

echo "================================================================================================"
echo "COMPREHENSIVE BENCHMARK SUITE - ALL BENCHMARKS"
echo "================================================================================================"
echo "Start time: $(date)"
echo ""

# Run SOTA benchmarks
echo "================================================================================================"
echo "1. SOTA BENCHMARKS (NumPy/OpenBLAS/SciPy)"
echo "================================================================================================"
python3 bench_sota.py

echo ""
echo "================================================================================================"
echo "2. Additional Python Benchmarks"
echo "================================================================================================"

# Check if there are other Python benchmark files
if [ -f bench_vs_linalg.mojo ]; then
    echo "Note: bench_vs_linalg.mojo requires Mojo compiler (not available in this environment)"
fi

echo ""
echo "================================================================================================"
echo "SUMMARY & COMPLETION"
echo "================================================================================================"
echo "End time: $(date)"
echo ""
echo "All available benchmarks have been executed."
echo "Results saved to combined_benchmark_results.txt"
