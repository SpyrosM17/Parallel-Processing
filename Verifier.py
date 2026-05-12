#!/usr/bin/env python3
"""
verify.py — Correctness checker for the serial scaler
------------------------------------------------------
Compares the C++ scaler output against scikit-learn's
StandardScaler / MinMaxScaler reference implementation.

Usage:
    python3 verify.py --input data.bin --cpp-output out.bin \
                      --N 10000 --D 128 --mode standard

    python3 verify.py --input data.bin --cpp-output out.bin \
                      --N 10000 --D 128 --mode minmax
"""

import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def main():
    parser = argparse.ArgumentParser(description="Verify scaler output.")
    parser.add_argument("--input",      required=True, help="Original binary input file")
    parser.add_argument("--cpp-output", required=True, help="C++ scaler output binary file")
    parser.add_argument("--N",   type=int, required=True, help="Number of rows")
    parser.add_argument("--D",   type=int, required=True, help="Number of columns")
    parser.add_argument("--mode", required=True, choices=["standard", "minmax"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    args = parser.parse_args()

    dtype = np.float64 if args.dtype == "float64" else np.float32

    print(f"Loading original data  : {args.input}")
    X = np.fromfile(args.input, dtype=dtype).reshape(args.N, args.D)

    print(f"Loading C++ output     : {args.cpp_output}")
    X_cpp = np.fromfile(args.cpp_output, dtype=np.float64).reshape(args.N, args.D)

    print(f"Computing reference    : sklearn {args.mode} scaler")
    if args.mode == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_ref = scaler.fit_transform(X.astype(np.float64))

    # ---- Compute errors ----
    diff = np.abs(X_cpp - X_ref)
    max_abs_err  = diff.max()
    mean_abs_err = diff.mean()

    print()
    print("=== Correctness Report ===")
    print(f"  Max  absolute error : {max_abs_err:.6e}")
    print(f"  Mean absolute error : {mean_abs_err:.6e}")

    tol = 1e-9
    if max_abs_err < tol:
        print(f"  ✓ PASS  (max error < {tol})")
    else:
        print(f"  ✗ FAIL  (max error >= {tol})")

    # ---- Per-column quick sanity check ----
    if args.mode == "standard":
        col_means = X_cpp.mean(axis=0)
        col_stds  = X_cpp.std(axis=0)
        print(f"\n  Post-scale column means  — max|mean|    : {np.abs(col_means).max():.6e}")
        print(f"  Post-scale column std    — max|std - 1| : {np.abs(col_stds - 1).max():.6e}")
    else:
        col_min = X_cpp.min(axis=0)
        col_max = X_cpp.max(axis=0)
        print(f"\n  Post-scale column min    — max|min|    : {np.abs(col_min).max():.6e}")
        print(f"  Post-scale column max    — max|max-1| : {np.abs(col_max - 1).max():.6e}")


if __name__ == "__main__":
    main()
