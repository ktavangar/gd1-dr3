"""
Combine the 60 PS1 FITS catalogs in data/PS1_data/ into a single deduplicated
table, keyed on objID (Pan-STARRS1 unique object identifier).

9.3 GB of input data comfortably fits in memory (tested machine has 32 GB RAM),
so this does a straightforward in-memory concat + dedup rather than any
chunked/out-of-core processing.

Run with the stream-env python, e.g.:
    ./stream-env/bin/python scripts/combine_ps1_data.py
"""
from pathlib import Path

import numpy as np
from astropy.table import Table, vstack

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "PS1_data"
OUT_PATH = DATA_DIR.parent / "PS1_combined_clean.fits"


def main():
    fits_files = sorted(DATA_DIR.glob("*.fit"))
    print(f"Found {len(fits_files)} files in {DATA_DIR}")

    tables = []
    for f in fits_files:
        t = Table.read(f, hdu=1)
        tables.append(t)
        print(f"  {f.name}: {len(t)} rows")

    full = vstack(tables, metadata_conflicts="silent")
    n_total = len(full)
    print(f"\nTotal rows before dedup: {n_total}")

    # Keep the first occurrence of each objID, preserving original row order.
    _, unique_idx = np.unique(full["objID"], return_index=True)
    unique_idx.sort()
    deduped = full[unique_idx]
    n_dupes = n_total - len(deduped)
    print(f"Duplicate rows removed: {n_dupes} ({n_dupes / n_total:.1%})")
    print(f"Rows after dedup: {len(deduped)}")

    deduped.write(OUT_PATH, overwrite=True)
    print(f"\nWrote clean combined table to {OUT_PATH}")


if __name__ == "__main__":
    main()
