#!/usr/bin/env python3
"""
verify.py — Correctness checker for the serial scaler
------------------------------------------------------
Compares the C++ scaler output against the expected StandardScaler / MinMaxScaler
reference implementation using a streaming, block-based verification approach.
"""

import argparse
import numpy as np


def read_block(f, count, dtype):
    itemsize = np.dtype(dtype).itemsize
    data = f.read(count * itemsize)
    if len(data) != count * itemsize:
        raise IOError(f"Expected {count * itemsize} bytes, got {len(data)} bytes")
    return np.frombuffer(data, dtype=dtype)


def compute_stats(input_path, N, D, dtype, block_rows):
    sum_ = np.zeros(D, dtype=np.float64)
    sum_sq = np.zeros(D, dtype=np.float64)
    min_ = np.full(D, np.inf, dtype=np.float64)
    max_ = np.full(D, -np.inf, dtype=np.float64)
    rows_remaining = N

    with open(input_path, "rb") as f:
        while rows_remaining > 0:
            rows_this = min(block_rows, rows_remaining)
            elems = rows_this * D
            block = read_block(f, elems, dtype).astype(np.float64).reshape(rows_this, D)
            sum_ += block.sum(axis=0)
            sum_sq += np.square(block).sum(axis=0)
            min_ = np.minimum(min_, block.min(axis=0))
            max_ = np.maximum(max_, block.max(axis=0))
            rows_remaining -= rows_this

    mean = sum_ / N
    var = sum_sq / N - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean, var, std, min_, max_


def verify_blocked(input_path, cpp_output_path, N, D, mode, dtype, block_rows):
    mean, _, std, min_, max_ = compute_stats(input_path, N, D, dtype, block_rows)
    range_ = max_ - min_

    if mode == "standard":
        shift = mean
        scale = np.where(std != 0.0, 1.0 / std, 0.0)
    else:
        shift = min_
        scale = np.where(range_ != 0.0, 1.0 / range_, 0.0)

    max_abs_err = 0.0
    sum_abs_err = 0.0
    output_sum = np.zeros(D, dtype=np.float64)
    output_sum_sq = np.zeros(D, dtype=np.float64)
    output_min = np.full(D, np.inf, dtype=np.float64)
    output_max = np.full(D, -np.inf, dtype=np.float64)

    with open(input_path, "rb") as fin, open(cpp_output_path, "rb") as fout:
        rows_remaining = N
        while rows_remaining > 0:
            rows_this = min(block_rows, rows_remaining)
            elems = rows_this * D
            block = read_block(fin, elems, dtype).astype(np.float64).reshape(rows_this, D)
            block_out = read_block(fout, elems, np.float64).reshape(rows_this, D)

            ref = (block - shift) * scale
            diff = np.abs(block_out - ref)
            max_abs_err = max(max_abs_err, diff.max())
            sum_abs_err += diff.sum()

            output_sum += block_out.sum(axis=0)
            output_sum_sq += np.square(block_out).sum(axis=0)
            output_min = np.minimum(output_min, block_out.min(axis=0))
            output_max = np.maximum(output_max, block_out.max(axis=0))
            rows_remaining -= rows_this

    mean_abs_err = sum_abs_err / (N * D)
    output_mean = output_sum / N
    output_var = output_sum_sq / N - np.square(output_mean)
    output_std = np.sqrt(np.maximum(output_var, 0.0))
    return max_abs_err, mean_abs_err, output_mean, output_std, output_min, output_max


def main():
    parser = argparse.ArgumentParser(description="Verify scaler output.")
    parser.add_argument("--input",      required=True, help="Original binary input file")
    parser.add_argument("--cpp-output", required=True, help="C++ scaler output binary file")
    parser.add_argument("--N",   type=int, required=True, help="Number of rows")
    parser.add_argument("--D",   type=int, required=True, help="Number of columns")
    parser.add_argument("--mode", required=True, choices=["standard", "minmax"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--block-rows", type=int, default=100000,
                        help="Rows per verification block for streaming verification")
    args = parser.parse_args()

    dtype = np.float64 if args.dtype == "float64" else np.float32

    print(f"Verifying input   : {args.input}")
    print(f"Verifying output  : {args.cpp_output}")
    print(f"Mode              : {args.mode}")
    print(f"Block rows        : {args.block_rows}")

    max_abs_err, mean_abs_err, output_mean, output_std, output_min, output_max = \
        verify_blocked(args.input, args.cpp_output, args.N, args.D, args.mode, dtype, args.block_rows)

    print()
    print("=== Correctness Report ===")
    print(f"  Max  absolute error : {max_abs_err:.6e}")
    print(f"  Mean absolute error : {mean_abs_err:.6e}")

    tol = 1e-9
    if max_abs_err < tol:
        print(f"  ✓ PASS  (max error < {tol})")
    else:
        print(f"  ✗ FAIL  (max error >= {tol})")

    if args.mode == "standard":
        print(f"\n  Post-scale column means  — max|mean|    : {np.abs(output_mean).max():.6e}")
        print(f"  Post-scale column std    — max|std - 1| : {np.abs(output_std - 1).max():.6e}")
    else:
        print(f"\n  Post-scale column min    — max|min|    : {np.abs(output_min).max():.6e}")
        print(f"  Post-scale column max    — max|max-1| : {np.abs(output_max - 1).max():.6e}")


if __name__ == "__main__":
    main()
