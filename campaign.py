import pandas as pd
import numpy as np

CAMPAIGN_WINDOW_MINUTES = 30
MIN_CAMPAIGN_ENTITIES = 3

CAMPAIGN_RISK_WEIGHTS = {
    "Critical": 40,
    "High": 20,
    "Medium": 10,
    "Low": 5,
}

_CAMPAIGN_TYPE_MAP = [
    ({"brute_force", "credential_stuffing"}, "Credential Spray Campaign"),
    ({"brute_force", "lateral_movement"}, "Breach-and-Pivot Campaign"),
    ({"lateral_movement", "low_and_slow_exfil"}, "Lateral Exfiltration Campaign"),
    ({"brute_force"}, "Distributed Brute Force"),
    ({"credential_stuffing"}, "Credential Stuffing Campaign"),
    ({"lateral_movement"}, "Coordinated Lateral Movement"),
    ({"low_and_slow_exfil"}, "Synchronized Exfiltration"),
    ({"behavioral_drift"}, "Insider Threat Cluster"),
]

def _classify_campaign(attack_types: set) -> str:
    base = {t.replace("_like", "") for t in attack_types}
    for required, label in _CAMPAIGN_TYPE_MAP:
        if required.issubset(base):
            return label
    return "Multi-Vector APT Campaign"

def _ip_subnet24(ip: str) -> str:
    parts = str(ip).split(".")
    return ".".join(parts[:3]) if len(parts) >= 3 else ""

def assign_campaign_ids(alert_df: pd.DataFrame, start_count: int = 0) -> pd.DataFrame:
    df = alert_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "campaign_id" not in df.columns:
        df["campaign_id"] = ""
    if "geo_location" not in df.columns:
        df["geo_location"] = ""
    if "source_ip" not in df.columns:
        df["source_ip"] = ""

    df["_subnet"] = df["source_ip"].apply(_ip_subnet24)
    assigned = {}
    counter = start_count

    for i in range(len(df)):
        if i in assigned:
            continue

        t_i = df.at[i, "timestamp"]
        geo_i = str(df.at[i, "geo_location"])
        subnet_i = str(df.at[i, "_subnet"])
        window_end = t_i + pd.Timedelta(minutes=CAMPAIGN_WINDOW_MINUTES)

        cluster = [i]
        for j in range(i + 1, len(df)):
            if df.at[j, "timestamp"] > window_end:
                break
            if j in assigned:
                continue
            if df.at[j, "entity_id"] == df.at[i, "entity_id"]:
                continue

            geo_j = str(df.at[j, "geo_location"])
            subnet_j = str(df.at[j, "_subnet"])

            geo_match = geo_i and geo_j and geo_i == geo_j and geo_i not in ("", "nan")
            subnet_match = subnet_i and subnet_j and subnet_i == subnet_j and subnet_i not in ("", "nan")

            if geo_match or subnet_match:
                cluster.append(j)

        unique_entities = df.loc[cluster, "entity_id"].nunique()
        if unique_entities >= MIN_CAMPAIGN_ENTITIES:
            counter += 1
            cid = f"C-{counter:03d}"
            for idx in cluster:
                if idx not in assigned:
                    assigned[idx] = cid

    for idx, cid in assigned.items():
        df.at[idx, "campaign_id"] = cid

    df = df.drop(columns=["_subnet"])
    return df

def compute_campaign_summary(alert_df: pd.DataFrame) -> pd.DataFrame:
    camp = alert_df[alert_df["campaign_id"].astype(str).str.startswith("C-")].copy()
    if len(camp) == 0:
        return pd.DataFrame(columns=[
            "campaign_id", "campaign_type", "entities_involved", "entity_list",
            "attack_types", "campaign_risk_score", "n_alerts", "t_start", "t_end",
        ])

    camp["timestamp"] = pd.to_datetime(camp["timestamp"])
    rows = []

    for cid, g in camp.groupby("campaign_id"):
        attack_types = set(g["predicted_type"].dropna().unique())
        tier_counts = g["severity_tier"].value_counts().to_dict()
        risk_score = min(
            sum(CAMPAIGN_RISK_WEIGHTS.get(t, 5) * n for t, n in tier_counts.items()),
            100,
        )
        duration_min = round(
            (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 60, 1
        )
        geo_list = [str(x) for x in g.get("geo_location", pd.Series()).unique() if x and str(x) not in ("", "nan")]

        rows.append({
            "campaign_id": cid,
            "campaign_type": _classify_campaign(attack_types),
            "entities_involved": g["entity_id"].nunique(),
            "entity_list": "; ".join(sorted(g["entity_id"].unique())[:12]),
            "attack_types": ", ".join(sorted(attack_types)),
            "campaign_risk_score": risk_score,
            "n_alerts": len(g),
            "t_start": g["timestamp"].min(),
            "t_end": g["timestamp"].max(),
            "duration_minutes": duration_min,
            "geo_locations": "; ".join(geo_list),
            "tier_breakdown": str(tier_counts),
        })

    return pd.DataFrame(rows).sort_values("campaign_risk_score", ascending=False).reset_index(drop=True)