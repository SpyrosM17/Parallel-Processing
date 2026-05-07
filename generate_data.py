~#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.datasets import make_regression


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic regression data and store X as raw binary."
    )

    parser.add_argument("--samples", type=int, required=True,
                        help="Number of samples/rows N")
    parser.add_argument("--features", type=int, required=True,
                        help="Number of features/columns D")
    parser.add_argument("--output", type=str, required=True,
                        help="Output binary file")
    parser.add_argument("--dtype", type=str, default="float64",
                        choices=["float32", "float64"],
                        help="Data type for output array")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise level for make_regression")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--targets-output", type=str, default=None,
                        help="Optional output binary file for y targets")

    args = parser.parse_args()

    print("Generating data...")
    print(f"N = {args.samples}")
    print(f"D = {args.features}")
    print(f"dtype = {args.dtype}")

    X, y = make_regression(
        n_samples=args.samples,
        n_features=args.features,
        noise=args.noise,
        random_state=args.seed
    )

    if args.dtype == "float32":
        X = X.astype(np.float32)
        y = y.astype(np.float32)
    else:
        X = X.astype(np.float64)
        y = y.astype(np.float64)

    X.tofile(args.output)

    print()
    print("Dataset generated successfully.")
    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"X output file: {args.output}")
    print(f"X file size: {os.path.getsize(args.output)} bytes")
    print(f"X file size: {os.path.getsize(args.output) / (1024 ** 3):.3f} GB")

    if args.targets_output is not None:
        y.tofile(args.targets_output)
        print()
        print(f"y shape: {y.shape}")
        print(f"y dtype: {y.dtype}")
        print(f"y output file: {args.targets_output}")
        print(f"y file size: {os.path.getsize(args.targets_output)} bytes")
        print(f"y file size: {os.path.getsize(args.targets_output) / (1024 ** 3):.3f} GB")


if __name__ == "__main__":
    main()
