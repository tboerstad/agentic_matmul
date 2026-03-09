#!/bin/bash

echo "Starting benchmark run at $(date)"
echo "========================================"

echo ""
echo "RUNNING SOTA BENCHMARKS..."
echo "========================================"
python3 bench_sota.py

echo ""
echo "========================================"
echo "RUNNING MOJO BENCHMARKS..."
echo "========================================"
mojo bench_matmul.mojo

echo ""
echo "========================================"
echo "Benchmark run completed at $(date)"
