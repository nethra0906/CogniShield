import time
import numpy as np
import pandas as pd
import torch

from features import FEATURE_COLUMNS, engineer_features
from sequences import WINDOW
from classify_explain import classify_and_explain
from severity import compute_severity
from mitre import get_mitre


def _score_window(model, window_array: np.ndarray, device: str = "cpu") -> float:
    model.eval()
    with torch.no_grad():
        t = torch.tensor(window_array[np.newaxis], dtype=torch.float32).to(device)
        recon = model(t)
        err = float(((recon - t) ** 2).mean().item())
    return err


def stream_events(
    df: pd.DataFrame,
    history: dict,
    access_graph,
    clf,
    explainer,
    model,
    score_95th: float,
    sensitive_resources: set,
    batch_delay: float = 0.05,
    device: str = "cpu",
):
    entity_buffers: dict = {}

    for _, raw_row in df.sort_values("timestamp").iterrows():
        entity_id = raw_row["entity_id"]

        single_row_df = pd.DataFrame([raw_row])
        feat_df = engineer_features(single_row_df, history, access_graph)
        feat_row = feat_df.iloc[0]

        feat_vec = feat_row[FEATURE_COLUMNS].values.astype(np.float32)
        if entity_id not in entity_buffers:
            entity_buffers[entity_id] = []
        entity_buffers[entity_id].append(feat_vec)

        buf = entity_buffers[entity_id]
        if len(buf) < WINDOW:
            pad_rows = WINDOW - len(buf)
            pad = np.zeros((pad_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
            window = np.vstack([pad] + buf)
        else:
            window = np.vstack(buf[-WINDOW:])

        anomaly_score = _score_window(model, window, device)

        off_hours = bool(feat_row.get("hour_deviation", 0) == 1)
        new_geo = bool(feat_row.get("new_geo", 0) == 1)
        auth_failed = bool(feat_row.get("auth_failed", 0) == 1)

        severity_score, severity_tier = compute_severity(
            anomaly_score=anomaly_score,
            score_95th=score_95th,
            entity_type=str(raw_row.get("entity_type", "user")),
            resource=str(raw_row.get("resource_accessed", "")),
            sensitive_resources=sensitive_resources,
            off_hours=off_hours,
            new_geo=new_geo,
            auth_failed=auth_failed,
        )

        explain = classify_and_explain(clf, explainer, feat_row, FEATURE_COLUMNS)
        mitre_info = get_mitre(explain["predicted_type"])

        result = {
            "entity_id": entity_id,
            "entity_type": str(raw_row.get("entity_type", "unknown")),
            "timestamp": str(raw_row["timestamp"]),
            "anomaly_score": round(anomaly_score, 6),
            "severity_score": severity_score,
            "severity_tier": severity_tier,
            "true_label": str(raw_row.get("label", "unknown")),
            "predicted_type": explain["predicted_type"],
            "confidence": explain["confidence"],
            "reasons": "; ".join(explain["reasons"]),
            "mitre_id": mitre_info["technique_id"],
            "mitre_name": mitre_info["technique_name"],
        }

        if batch_delay > 0:
            time.sleep(batch_delay)

        yield result
