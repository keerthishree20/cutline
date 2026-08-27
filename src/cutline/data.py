"""Load the raw CSVs once, convert to parquet, and never touch the CSVs again.

`train_transaction.csv` is ~590k rows x ~394 columns. Read with pandas' default
float64 dtypes it is close to 2 GB in RAM and takes a minute to parse. Downcast
to float32, write parquet once, and every later phase loads it in seconds.
"""

from __future__ import annotations

import pandas as pd

from . import config

ID_COL = "TransactionID"
TARGET = "isFraud"
AMOUNT = "TransactionAmt"


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, wide ints -> narrow, object -> category. In place-ish."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in _string_columns(df):
        df[col] = df[col].astype("category")

    return df


def _string_columns(df: pd.DataFrame) -> list[str]:
    """Text columns, across the pandas 2 -> 3 dtype change.

    pandas 2 gives text columns `object` dtype; pandas 3 gives them StringDtype
    and warns if you select them via "object". Ask the dtype directly instead.
    """
    return [
        col
        for col, dtype in df.dtypes.items()
        if dtype == object or pd.api.types.is_string_dtype(dtype)
    ]


def build_parquet(force: bool = False) -> pd.DataFrame:
    """Merge transaction + identity, add `has_identity`, write parquet.

    The identity join is a LEFT join and it is mostly misses — identity rows
    cover only a minority of transactions, so DeviceType, DeviceInfo and the
    id_* columns are null for most rows. `has_identity` captures that absence,
    which is itself predictive, and stops later phases from quietly building an
    explanation story on columns that are empty three times out of four.
    """
    if config.TRAIN_PARQUET.exists() and not force:
        print(f"parquet already built: {config.TRAIN_PARQUET}")
        return load()

    tx_path = config.RAW / "train_transaction.csv"
    id_path = config.RAW / "train_identity.csv"
    for path in (tx_path, id_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run `python scripts/download_data.py` first"
            )

    print(f"reading {tx_path.name} ...")
    tx = downcast(pd.read_csv(tx_path, low_memory=False))
    print(f"  {len(tx):,} rows x {tx.shape[1]} cols")

    print(f"reading {id_path.name} ...")
    ident = downcast(pd.read_csv(id_path, low_memory=False))
    print(f"  {len(ident):,} rows x {ident.shape[1]} cols")

    df = tx.merge(ident, on=ID_COL, how="left")
    df["has_identity"] = df[ID_COL].isin(ident[ID_COL]).astype("int8")

    coverage = df["has_identity"].mean()
    print(f"\nidentity coverage: {coverage:.1%} of transactions")
    print(f"  fraud rate WITH identity:    {df.loc[df.has_identity == 1, TARGET].mean():.3%}")
    print(f"  fraud rate WITHOUT identity: {df.loc[df.has_identity == 0, TARGET].mean():.3%}")
    if coverage < 0.5:
        print("  -> device/browser features are null for most rows. Explanation copy")
        print("     must fall back to transaction-side features. This is expected.")

    config.INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TRAIN_PARQUET, index=False, compression="zstd")
    size_mb = config.TRAIN_PARQUET.stat().st_size / 1e6
    print(f"\nwrote {config.TRAIN_PARQUET} ({size_mb:.0f} MB)")
    print("the raw CSVs are now redundant — delete them to reclaim ~700 MB")

    return df


def load(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the prepared parquet. Pass `columns` to read only what you need."""
    if not config.TRAIN_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_PARQUET} missing — run `python scripts/prepare_data.py`"
        )
    return pd.read_parquet(config.TRAIN_PARQUET, columns=columns)
