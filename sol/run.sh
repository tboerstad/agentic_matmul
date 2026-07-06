#!/usr/bin/env bash
# Measure this machine's empirical speed-of-light: AVX-512 FMA peak (f64/f32,
# 1 and 4 threads) and read bandwidth at DRAM/L3/L2/L1 footprints.
# Run in the same boot as any kernel benchmark you want to state as % of SOL.
set -euo pipefail
cd "$(dirname "$0")"
CC=${CC:-gcc}
$CC -O2 -march=native -o sol_fma sol_fma.c -lpthread
$CC -O2 -march=native -o sol_bw sol_bw.c -lpthread
echo "== FMA peak (3 runs each; take the best, VM contention drops single runs) =="
for i in 1 2 3; do ./sol_fma 1; done
for i in 1 2 3; do ./sol_fma "$(nproc)"; done
for i in 1 2 3; do ./sol_fma 1 f32; done
for i in 1 2 3; do ./sol_fma "$(nproc)" f32; done
echo "== Read bandwidth =="
./sol_bw
