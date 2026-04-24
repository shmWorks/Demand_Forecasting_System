"""One-time conversion script: CSV → Parquet (zstd compressed).

Run once from project root:
    python scripts/convert_to_parquet.py

Creates data/parquet/*.parquet. After conversion, load_raw_data()
will automatically prefer Parquet (~10-20x faster I/O).
"""
import sys
from pathlib import Path

# Ensure retail_iq is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import polars as pl
from retail_iq.config import RAW_DATA_DIR, PARQUET_DATA_DIR

FILES = [
    ("train", True),
    ("test", True),
    ("stores", False),
    ("oil", True),
    ("holidays_events", True),
    ("transactions", True),
]


def convert_all() -> None:
    """Read CSVs, write zstd-compressed Parquet, print size comparison."""
    PARQUET_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, has_date in FILES:
        csv_path = RAW_DATA_DIR / f"{name}.csv"
        parquet_path = PARQUET_DATA_DIR / f"{name}.parquet"

        if not csv_path.exists():
            print(f"  SKIP {name}.csv — not found")
            continue

        print(f"  Converting {name}.csv ...", end=" ", flush=True)
        df = pl.read_csv(csv_path, try_parse_dates=has_date)
        df.write_parquet(parquet_path, compression="zstd")

        csv_mb = csv_path.stat().st_size / 1_048_576
        pq_mb = parquet_path.stat().st_size / 1_048_576
        ratio = csv_mb / pq_mb if pq_mb > 0 else 0
        print(f"done  {csv_mb:.1f}MB -> {pq_mb:.1f}MB  ({ratio:.1f}x smaller)")

    print("\nAll Parquet files written to:", PARQUET_DATA_DIR)


if __name__ == "__main__":
    convert_all()
