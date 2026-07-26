import json
import numpy as np
import pandas as pd
from graph import graph_distance as compute_graph_distance, COLD_START_SENTINEL

RESOURCE_VOCAB_SIZE = 64


def _hash_bucket(value, buckets=RESOURCE_VOCAB_SIZE):
    return hash(value) % buckets


def build_entity_history(df: pd.DataFrame, rolling_days: int = None) -> dict:
    history = {}
    df_sorted = df.sort_values("timestamp")

    if rolling_days is not None:
        max_ts = pd.to_datetime(df_sorted["timestamp"]).max()
        cutoff = max_ts - pd.Timedelta(days=rolling_days)
        df_sorted = df_sorted[pd.to_datetime(df_sorted["timestamp"]) >= cutoff]

    for entity_id, g in df_sorted.groupby("entity_id"):
        normal_g = g[g["label"] == "normal"]

        if rolling_days is not None:
            seed = normal_g if len(normal_g) > 0 else g
        else:
            seed = normal_g.head(20) if len(normal_g) >= 5 else g.head(5)

        if len(seed) == 0:
            seed = g.head(5)

        history[entity_id] = {
            "known_geos": set(seed["geo_location"]),
            "known_resources": set(seed["resource_accessed"]),
            "known_fp": set(seed["device_fingerprint"]),
            "known_hours": set(pd.to_datetime(seed["timestamp"]).dt.hour),
            "n_seen": len(g),
            "avg_duration": seed["session_duration"].mean() if len(seed) else 300.0,
        }
    return history


def engineer_features(df: pd.DataFrame, history: dict, access_graph=None) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["dow"] = pd.to_datetime(df["timestamp"]).dt.dayofweek

    def row_features(row):
        h = history.get(row["entity_id"], None)
        cold_start = h is None or h["n_seen"] < 5

        if cold_start:
            new_geo = 1
            new_resource = 1
            new_device = 1
            hour_deviation = 0
            dur_z = 0.0
            gdist = COLD_START_SENTINEL
        else:
            new_geo = int(row["geo_location"] not in h["known_geos"])
            new_resource = int(row["resource_accessed"] not in h["known_resources"])
            new_device = int(row["device_fingerprint"] not in h["known_fp"])
            hour_deviation = int(row["hour"] not in h["known_hours"])
            dur_z = (row["session_duration"] - h["avg_duration"]) / (h["avg_duration"] + 1e-6)

            if access_graph is not None:
                gdist = compute_graph_distance(row["entity_id"], row["resource_accessed"], access_graph)
            else:
                gdist = 0.0

        try:
            n_cmds = len(json.loads(row["command_sequence"]))
        except Exception:
            n_cmds = 1

        return pd.Series({
            "new_geo": new_geo,
            "new_resource": new_resource,
            "new_device": new_device,
            "hour_deviation": hour_deviation,
            "duration_z": np.clip(dur_z, -5, 5),
            "n_commands": n_cmds,
            "auth_failed": int(not row.get("success", True)),
            "resource_hash": _hash_bucket(row["resource_accessed"]) / RESOURCE_VOCAB_SIZE,
            "cold_start": int(cold_start),
            "graph_distance": gdist,
        })

    feat = df.apply(row_features, axis=1)
    out = pd.concat([df, feat], axis=1)
    return out


FEATURE_COLUMNS = [
    "new_geo", "new_resource", "new_device", "hour_deviation",
    "duration_z", "n_commands", "auth_failed", "resource_hash",
    "graph_distance",
]
