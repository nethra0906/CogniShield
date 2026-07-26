import numpy as np
import pandas as pd
import torch

DRIFT_THRESHOLD_SIGMA = 2.5
MIN_BASELINE_SESSIONS = 3

BEHAVIORAL_DRIFT_MITRE = {
    "mitre_id": "T1078",
    "mitre_name": "Valid Accounts (Behavioral Anomaly)",
    "mitre_tactic": "Defense Evasion / Persistence",
}

def encode_sequences(model, X: np.ndarray, device: str = "cpu", batch_size: int = 256) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            _, (h, _) = model.encoder(batch)
            z = model.to_latent(h[-1])
            embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings)

def build_entity_fingerprints(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    rolling_days: int = 14,
) -> dict:
    meta = meta_df.copy().reset_index(drop=True)
    meta["_emb"] = list(embeddings)
    normal_meta = meta[meta["label"] == "normal"]
    fingerprints = {}

    for entity_id, g in normal_meta.groupby("entity_id"):
        max_ts = pd.to_datetime(g["timestamp"]).max()
        cutoff = max_ts - pd.Timedelta(days=rolling_days)
        window = g[pd.to_datetime(g["timestamp"]) >= cutoff]
        if len(window) < MIN_BASELINE_SESSIONS:
            window = g

        embs = np.stack(window["_emb"].values)
        centroid = embs.mean(axis=0)
        drifts_from_centroid = [
            float(1.0 - np.dot(e, centroid) / (
                np.linalg.norm(e) * np.linalg.norm(centroid) + 1e-9
            ))
            for e in embs
        ]
        fingerprints[entity_id] = {
            "centroid": centroid,
            "n_sessions": len(window),
            "intra_drift_mean": float(np.mean(drifts_from_centroid)),
            "intra_drift_std": float(np.std(drifts_from_centroid)),
        }

    return fingerprints

def compute_drift_scores(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    fingerprints: dict,
) -> np.ndarray:
    drifts = []
    for i, (_, row) in enumerate(meta_df.iterrows()):
        fp = fingerprints.get(row["entity_id"])
        if fp is None:
            drifts.append(0.0)
        else:
            c = fp["centroid"]
            e = embeddings[i]
            sim = np.dot(e, c) / (np.linalg.norm(e) * np.linalg.norm(c) + 1e-9)
            drifts.append(float(1.0 - sim))
    return np.array(drifts)

def detect_behavioral_drift_alerts(
    embeddings: np.ndarray,
    meta_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    fingerprints: dict,
    recon_scores: np.ndarray,
    recon_alert_threshold: float,
) -> tuple:
    drift_scores = compute_drift_scores(embeddings, meta_df, fingerprints)

    meta_reset = meta_df.reset_index(drop=True)
    normal_mask = (meta_reset["label"] == "normal").values
    if normal_mask.sum() > 10:
        drift_mean = float(drift_scores[normal_mask].mean())
        drift_std = float(drift_scores[normal_mask].std())
    else:
        drift_mean = float(drift_scores.mean())
        drift_std = float(drift_scores.std())

    drift_threshold = drift_mean + DRIFT_THRESHOLD_SIGMA * drift_std

    high_drift = drift_scores > drift_threshold
    low_recon = recon_scores < recon_alert_threshold

    alerts = []
    feat_reset = feat_df.reset_index(drop=True)

    for i, row in meta_reset.iterrows():
        if not (high_drift[i] and low_recon[i]):
            continue

        drift = float(drift_scores[i])
        drift_ratio = drift / max(drift_threshold, 1e-9)
        severity_score = round(min(drift_ratio * 55.0, 100.0), 2)
        severity_tier = (
            "Critical" if severity_score >= 80
            else "High" if severity_score >= 60
            else "Medium"
        )

        df_idx = row.get("df_index", i)
        entity_type = "unknown"
        geo = ""
        src_ip = ""
        if df_idx in feat_reset.index:
            fr = feat_reset.loc[df_idx]
            entity_type = str(fr.get("entity_type", "unknown"))
            geo = str(fr.get("geo_location", ""))
            src_ip = str(fr.get("source_ip", ""))

        alerts.append({
            "entity_id": row["entity_id"],
            "entity_type": entity_type,
            "timestamp": str(row["timestamp"]),
            "anomaly_score": round(float(recon_scores[i]), 6),
            "severity_score": severity_score,
            "severity_tier": severity_tier,
            "true_label": str(row.get("label", "unknown")),
            "predicted_type": "behavioral_drift",
            "confidence": round(min(drift_ratio, 1.0), 3),
            "reasons": (
                "behavioral fingerprint drifted from 14-day baseline; "
                "low reconstruction error suggests evasion-aware or slow-burn attack"
            ),
            "shap_values": "{}",
            "mitre_id": BEHAVIORAL_DRIFT_MITRE["mitre_id"],
            "mitre_name": BEHAVIORAL_DRIFT_MITRE["mitre_name"],
            "mitre_tactic": BEHAVIORAL_DRIFT_MITRE["mitre_tactic"],
            "kill_chain_stage": "dormant",
            "next_stage": "initial_access",
            "next_stage_prob": 0.0,
            "escalation_warning": False,
            "campaign_id": "",
            "drift_score": round(drift, 5),
            "geo_location": geo,
            "source_ip": src_ip,
        })

    return alerts, drift_scores, float(drift_threshold)