import subprocess
import sys

import pandas as pd

from features import _hash_bucket, build_entity_history, engineer_features, FEATURE_COLUMNS
from graph import COLD_START_SENTINEL


def test_hash_bucket_is_stable_within_a_process():
    assert _hash_bucket("/resource/5") == _hash_bucket("/resource/5")


def test_hash_bucket_is_stable_across_processes():
    # Regression test: the built-in hash() is salted per-process (PYTHONHASHSEED),
    # so resource_hash used to silently change value every run even with a fixed
    # RNG_SEED, breaking reproducibility of the trained model's scores.
    code = "from features import _hash_bucket; print(_hash_bucket('/resource/5'))"
    results = set()
    for _ in range(3):
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        )
        results.add(out.stdout.strip())
    assert len(results) == 1


def _minimal_df():
    return pd.DataFrame([
        {
            "entity_id": "e1", "entity_type": "user", "timestamp": "2026-01-01T10:00:00",
            "label": "normal", "geo_location": "US-East", "resource_accessed": "/resource/1",
            "device_fingerprint": "fp1", "session_duration": 100.0, "success": True,
            "command_sequence": '["read"]',
        }
    ])


def test_cold_start_entity_gets_sentinel_distance_and_flag():
    df = _minimal_df()
    history = build_entity_history(df)
    feat = engineer_features(df, history, access_graph=None)
    row = feat.iloc[0]
    assert row["cold_start"] == 1
    assert row["graph_distance"] == COLD_START_SENTINEL
    for col in FEATURE_COLUMNS:
        assert col in feat.columns
