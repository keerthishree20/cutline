#!/usr/bin/env python
"""Exercise the whole ingestion path on synthetic CSVs, in a temp directory.

`build_parquet` otherwise runs for the first time on a 700 MB file after a
multi-minute parse — a bad place to discover that a dtype or a compression
codec behaves differently than expected. This does the same round trip at
1/100th the scale in a few seconds:

    synthetic frame -> two CSVs on disk -> build_parquet -> read back -> split

The identity CSV deliberately contains only the covered rows, so the merge is a
genuine left join and `has_identity` has to be *derived*, not copied.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import config, data, metrics, split, synthetic  # noqa: E402

IDENTITY_COLS = ["id_01", "id_02", "DeviceType", "DeviceInfo"]


def main() -> None:
    df = synthetic.make_frame(4_000)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Point the package at throwaway paths for the duration of the test.
        config.RAW = root / "raw"
        config.INTERIM = root / "interim"
        config.TRAIN_PARQUET = config.INTERIM / "train.parquet"
        config.RAW.mkdir(parents=True)
        config.INTERIM.mkdir(parents=True)

        covered = df[df["has_identity"] == 1]
        tx = df.drop(columns=IDENTITY_COLS + ["has_identity"])
        ident = covered[["TransactionID"] + IDENTITY_COLS]

        tx.to_csv(config.RAW / "train_transaction.csv", index=False)
        ident.to_csv(config.RAW / "train_identity.csv", index=False)
        print(f"wrote {len(tx):,} transaction rows, {len(ident):,} identity rows\n")

        built = data.build_parquet(force=True)
        reloaded = data.load()

        # --- the assertions that matter ---
        assert len(reloaded) == len(tx), "row count changed through the round trip"
        coverage = reloaded["has_identity"].mean()
        assert 0.05 < coverage < 0.95, (
            f"has_identity is {coverage:.1%} — it was copied, not derived from the join"
        )
        assert reloaded["DeviceType"].isna().sum() > 0, "left join produced no misses"
        assert reloaded.loc[reloaded.has_identity == 0, "DeviceType"].isna().all(), \
            "uncovered rows should have no device data"
        assert reloaded["TransactionAmt"].dtype == "float32", "downcast lost through parquet"

        parts = split.time_ordered_split(reloaded)
        assert parts.train.TransactionDT.max() <= parts.calib.TransactionDT.min()
        assert parts.calib.TransactionDT.max() <= parts.test.TransactionDT.min()

        print("\n" + parts.describe())
        print(f"\ndtypes surviving parquet: {dict(reloaded.dtypes.value_counts().items())}")
        print("\nsmoke test PASSED — ingestion, dtypes, join and split all sound.")


if __name__ == "__main__":
    main()
