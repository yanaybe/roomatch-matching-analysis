"""Drop aggregate rows whose counts could identify an individual.

Usage: python privacy/suppress_small_cells.py data/real [--k 10]

Every numeric count column in every CSV is checked. A row is dropped if any
*denominator-like* count (the first count column: presses / attempts /
installs / failures) is below k. Dropped rows are reported, not silently lost.
"""

import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--k", type=int, default=10)
args = parser.parse_args()

for csv in sorted(args.folder.glob("*.csv")):
    df = pd.read_csv(csv)
    counts = df.select_dtypes("number").columns
    if counts.empty:
        continue
    base = counts[0]
    small = df[base] < args.k
    if small.any():
        print(f"{csv.name}: dropping {int(small.sum())} row(s) with {base} < {args.k}")
    df[~small].to_csv(csv, index=False)
print("done")
