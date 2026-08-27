#!/usr/bin/env python
"""Fetch only the two files this project can actually use.

The IEEE-CIS test files (test_transaction / test_identity) are ~600 MB and
their labels were never released, so they cannot be evaluated against. Our test
set comes from time-splitting *within* train instead — see cutline.split.

Requires ~/.kaggle/kaggle.json AND accepting the competition rules once, by
hand, at https://www.kaggle.com/c/ieee-fraud-detection/rules — the API cannot
accept them for you. A 403 here almost always means that click is missing.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cutline import config  # noqa: E402

COMPETITION = "ieee-fraud-detection"
FILES = ["train_transaction.csv", "train_identity.csv"]


def fetch(name: str) -> None:
    target = config.RAW / name
    if target.exists():
        print(f"{name}: already present ({target.stat().st_size / 1e6:.0f} MB)")
        return

    print(f"{name}: downloading ...")
    # Use the venv's kaggle binary, not whatever is on PATH.
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    result = subprocess.run(
        [str(kaggle_bin) if kaggle_bin.exists() else "kaggle",
         "competitions", "download", "-c", COMPETITION,
         "-f", name, "-p", str(config.RAW)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        print(f"  FAILED: {err}", file=sys.stderr)
        if "403" in err or "Forbidden" in err:
            print(
                "\n  -> Accept the competition rules once, in a browser:\n"
                f"     https://www.kaggle.com/c/{COMPETITION}/rules\n"
                "     then re-run this script.",
                file=sys.stderr,
            )
        sys.exit(1)

    # Kaggle wraps single-file downloads in a zip.
    archive = config.RAW / f"{name}.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(config.RAW)
        archive.unlink()
    print(f"  done ({target.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    config.RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name)
    print("\nnext: python scripts/prepare_data.py")


if __name__ == "__main__":
    main()
