#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.datasets import make_regression

def main():
    parser = argparse.ArgumentParser(
        description="Generates the dummy binary data for testing. Does it in chunks so Python doesn't eat all the RAM."
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
    parser.add_argument("--chunk-size", type=int, default=1000000,
                        help="Number of rows to generate per chunk to prevent Out Of Memory errors")

    args = parser.parse_args()

    print("Generating data in chunks...")
    print(f"N = {args.samples}")
    print(f"D = {args.features}")
    print(f"dtype = {args.dtype}")
    print(f"chunk_size = {args.chunk_size}")

    chunks = args.samples // args.chunk_size
    remainder = args.samples % args.chunk_size

    # Open files in append-binary mode
    f_x = open(args.output, 'wb')
    f_y = open(args.targets_output, 'wb') if args.targets_output else None

    try:
        for i in range(chunks):
            X_chunk, y_chunk = make_regression(
                n_samples=args.chunk_size,
                n_features=args.features,
                noise=args.noise,
                random_state=args.seed + i # Increment seed so each chunk has unique random data
            )

            if args.dtype == "float32":
                X_chunk = X_chunk.astype(np.float32)
                y_chunk = y_chunk.astype(np.float32)
            else:
                X_chunk = X_chunk.astype(np.float64)
                y_chunk = y_chunk.astype(np.float64)

            X_chunk.tofile(f_x)
            if f_y:
                y_chunk.tofile(f_y)
            
            print(f"  Wrote chunk {i+1}/{chunks}")

        # Handle any leftover rows that didn't fit perfectly into the chunk sizes
        if remainder > 0:
            X_chunk, y_chunk = make_regression(
                n_samples=remainder,
                n_features=args.features,
                noise=args.noise,
                random_state=args.seed + chunks
            )

            if args.dtype == "float32":
                X_chunk = X_chunk.astype(np.float32)
                y_chunk = y_chunk.astype(np.float32)
            else:
                X_chunk = X_chunk.astype(np.float64)
                y_chunk = y_chunk.astype(np.float64)

            X_chunk.tofile(f_x)
            if f_y:
                y_chunk.tofile(f_y)
            print("  Wrote remainder chunk")

    finally:
        # Ensure files are properly closed even if an error occurs
        f_x.close()
        if f_y:
            f_y.close()

    print()
    print("Dataset generated successfully.")
    print(f"X shape: ({args.samples}, {args.features})")
    print(f"X dtype: {args.dtype}")
    print(f"X output file: {args.output}")
    print(f"X file size: {os.path.getsize(args.output)} bytes")
    print(f"X file size: {os.path.getsize(args.output) / (1024 ** 3):.3f} GB")

    if args.targets_output is not None:
        print()
        print(f"y shape: ({args.samples},)")
        print(f"y dtype: {args.dtype}")
        print(f"y output file: {args.targets_output}")
        print(f"y file size: {os.path.getsize(args.targets_output)} bytes")
        print(f"y file size: {os.path.getsize(args.targets_output) / (1024 ** 3):.3f} GB")

if __name__ == "__main__":
    main()