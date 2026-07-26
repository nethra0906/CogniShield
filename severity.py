import pandas as pd

SEVERITY_TIERS = [
    (80.0, "Critical"),
    (60.0, "High"),
    (35.0, "Medium"),
    (0.0, "Low"),
]

PRIVILEGED_ENTITY_TYPES = {"service_account"}


def compute_severity(
    anomaly_score: float,
    score_95th: float,
    entity_type: str,
    resource: str,
    sensitive_resources: set,
    off_hours: bool,
    new_geo: bool,
    auth_failed: bool,
) -> tuple:
    base = min(anomaly_score / max(score_95th, 1e-9), 1.0) * 60.0

    multiplier = 1.0
    if entity_type in PRIVILEGED_ENTITY_TYPES:
        multiplier += 0.20
    if resource in sensitive_resources:
        multiplier += 0.15
    if off_hours and new_geo:
        multiplier += 0.15
    if auth_failed:
        multiplier += 0.10

    score = min(base * multiplier, 100.0)

    tier = "Low"
    for threshold, label in SEVERITY_TIERS:
        if score >= threshold:
            tier = label
            break

    return round(score, 2), tier


def identify_sensitive_resources(df: pd.DataFrame, top_pct: float = 0.20) -> set:
    resource_counts = df["resource_accessed"].value_counts()
    n = max(1, int(len(resource_counts) * top_pct))
    return set(resource_counts.head(n).index)
