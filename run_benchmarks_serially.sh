#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Running Benchmarks Serially"
echo "=========================================="
echo ""

# Run SOTA benchmark (Python)
echo "Step 1: Running SOTA (Python) benchmark..."
echo "-------------------------------------------"
python bench_sota.py
echo ""
echo "SOTA benchmark complete!"
echo ""

# Run Mojo benchmark
echo "Step 2: Running Mojo benchmark..."
echo "-------------------------------------------"
mojo bench_matmul.mojo
echo ""
echo "Mojo benchmark complete!"
echo ""

echo "=========================================="
echo "All benchmarks completed successfully!"
echo "=========================================="
