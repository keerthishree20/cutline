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
from cutline import config, data, features, metrics, split, synthetic  # noqa: E402

IDENTITY_COLS = ["id_01", "id_02", "DeviceType", "DeviceInfo"]


def test_serving_parity() -> None:
    """The model must score identically after a save/load round trip.

    Phase 4 loads this bundle and serves from it. If the encoder is rebuilt
    from a second copy of the logic, or category levels are re-derived at load
    time, the service returns scores that quietly disagree with the metrics
    table — no exception, no warning, just wrong numbers in production.
    """
    import joblib
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    df = features.add_history_features(synthetic.make_frame(12_000))
    parts = split.time_ordered_split(df)

    builder = features.FeatureBuilder().fit(parts.train)
    X_tr, X_te = builder.transform(parts.train), builder.transform(parts.test)

    # The silent killer: independent .astype("category") calls give the same
    # string different integer codes on either side.
    for col in builder.categorical_:
        assert list(X_tr[col].cat.categories) == list(X_te[col].cat.categories), (
            f"{col}: category coding differs between train and test"
        )

    # Assert per high-cardinality column, not on the max. Low-cardinality keys
    # like P_emaildomain legitimately sit at 0.0 (every level appears in train),
    # so a max() check keeps passing on card1 alone even if a bug zeroed the
    # rest — which is precisely the failure this test exists to catch.
    unseen = builder.unseen_rate(parts.test)
    for col in ("card1", "addr1"):
        if col in unseen:
            assert unseen[col] > 0.0, (
                f"{col}: no unseen keys on held-out data — the encoder was "
                f"fitted on more than the train slice"
            )

    model = lgb.LGBMClassifier(n_estimators=60, num_leaves=15, verbose=-1,
                               random_state=config.RANDOM_STATE)
    model.fit(X_tr, parts.train["isFraud"])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.predict_proba(builder.transform(parts.calib))[:, 1],
            parts.calib["isFraud"])

    expected = iso.predict(model.predict_proba(X_te)[:, 1])

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bundle.joblib"
        joblib.dump({"feature_builder": builder, "model": model, "isotonic": iso}, path)
        loaded = joblib.load(path)

    got = loaded["isotonic"].predict(
        loaded["model"].predict_proba(loaded["feature_builder"].transform(parts.test))[:, 1]
    )

    import numpy as np
    assert np.array_equal(expected, got), "scores changed across the save/load round trip"
    print(f"serving parity: {len(got):,} scores identical after joblib round trip")
    print(f"  unseen-key rates held out: {({k: round(v, 4) for k, v in unseen.items()})}")


def test_serving_matches_batch() -> None:
    """The claim the README makes, actually checked.

    Batch scoring runs `add_history_features` over the whole frame. Serving
    rebuilds one row's history from a `HistoryStore`. Those are different code
    paths, and the promise is that they agree exactly. Nothing verified that
    until this test — and a silent mismatch here means the dashboard disagrees
    with the metrics table, which is the worst possible thing to discover while
    someone is watching.
    """
    import numpy as np

    from cutline import bundle as bundle_mod
    from cutline import serving

    raw = synthetic.make_frame(8_000)
    featurised = features.add_history_features(raw)
    batch_scores = bundle_mod.score(bundle_mod.load(), featurised)

    key = features.HISTORY_KEY
    # A card with real history, scored late enough to have accumulated some.
    counts = featurised[key].value_counts()
    busy = counts[counts >= 5].index[0]
    positions = np.flatnonzero((featurised[key] == busy).to_numpy())
    target = int(positions[-1])

    b = bundle_mod.load()
    scorer_bundle = b
    # Seed with every raw row strictly before the target, in time order.
    seed = featurised.iloc[:target][raw.columns].reset_index(drop=True)
    store = serving.HistoryStore(seed=seed)

    row = featurised.iloc[target][raw.columns].to_dict()
    rebuilt = store.add(row)
    served = bundle_mod.score(scorer_bundle, rebuilt)[0]

    expected = float(batch_scores[target])
    assert abs(served - expected) < 1e-9, (
        f"serving and batch disagree: {served:.12f} vs {expected:.12f}. "
        f"The two feature paths have diverged."
    )
    print(f"serving/batch parity: card {busy} at row {target:,} — "
          f"{served:.10f} both paths")


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
        print("\ningestion OK — dtypes, join and split all sound.\n")

    test_serving_parity()
    test_serving_matches_batch()
    print("\nsmoke test PASSED.")


if __name__ == "__main__":
    main()
