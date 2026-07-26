import random
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

ENTITY_TYPES = ["user", "service_account", "edge_device"]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
RESOURCE_POOL = [f"/resource/{i}" for i in range(1, 60)]
GEO_POOL = [
    ("US-East", 40.7, -74.0), ("US-West", 37.7, -122.4), ("EU-West", 51.5, -0.1),
    ("EU-Central", 48.1, 11.6), ("APAC-SE", 1.35, 103.8), ("APAC-East", 35.7, 139.7),
    ("SA-East", -23.5, -46.6), ("ME", 25.2, 55.3),
]


def haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


class Entity:
    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.home_geo = random.choice(GEO_POOL)
        self.home_ip_prefix = f"{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}"
        self.typical_hours = sorted(random.sample(range(24), k=random.randint(3, 8)))
        self.typical_resources = random.sample(RESOURCE_POOL, k=random.randint(3, 10))
        self.auth_method = random.choice(AUTH_METHODS)
        self.device_fp = f"{fake.user_agent()}|{fake.mac_address()}"
        self.avg_session_s = random.uniform(60, 1800)


def make_entities(n_users=150, n_service=30, n_devices=40):
    entities = []
    for i in range(n_users):
        entities.append(Entity(f"user_{i:04d}", "user"))
    for i in range(n_service):
        entities.append(Entity(f"svc_{i:04d}", "service_account"))
    for i in range(n_devices):
        entities.append(Entity(f"dev_{i:04d}", "edge_device"))
    return entities


def _base_row(entity, ts, label):
    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "timestamp": ts.isoformat(),
        "label": label,
        "chain_id": "",
        "campaign_id": "",
        "success": True,
    }


def normal_session(entity, ts):
    r = _base_row(entity, ts, "normal")
    r.update({
        "source_ip": f"{entity.home_ip_prefix}.{random.randint(1,254)}",
        "geo_location": entity.home_geo[0],
        "resource_accessed": random.choice(entity.typical_resources),
        "auth_method": entity.auth_method,
        "session_duration": max(5, np.random.normal(entity.avg_session_s, entity.avg_session_s * 0.2)),
        "command_sequence": json.dumps(random.sample(["read", "list", "write", "query"], k=random.randint(1, 3))),
        "device_fingerprint": entity.device_fp,
    })
    return r


def brute_force_session(entity, ts):
    r = _base_row(entity, ts, "brute_force")
    r.update({
        "source_ip": f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "geo_location": random.choice(GEO_POOL)[0],
        "resource_accessed": entity.typical_resources[0],
        "auth_method": entity.auth_method,
        "session_duration": random.uniform(1, 5),
        "command_sequence": json.dumps(["auth_attempt"]),
        "device_fingerprint": "unknown|" + fake.mac_address(),
        "success": False,
    })
    return r


def impossible_travel_session(entity, ts):
    geo_name, lat, lon = random.choice([g for g in GEO_POOL if g != entity.home_geo])
    r = _base_row(entity, ts, "impossible_travel")
    r.update({
        "source_ip": f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}",
        "geo_location": geo_name,
        "resource_accessed": random.choice(entity.typical_resources),
        "auth_method": entity.auth_method,
        "session_duration": np.random.normal(entity.avg_session_s, entity.avg_session_s * 0.2),
        "command_sequence": json.dumps(["read", "list"]),
        "device_fingerprint": entity.device_fp,
        "_geo_lat": lat, "_geo_lon": lon,
    })
    return r


def credential_stuffing_session(entity, ts, shared_ip):
    r = _base_row(entity, ts, "credential_stuffing")
    r.update({
        "source_ip": shared_ip,
        "geo_location": random.choice(GEO_POOL)[0],
        "resource_accessed": entity.typical_resources[0] if entity.typical_resources else random.choice(RESOURCE_POOL),
        "auth_method": entity.auth_method,
        "session_duration": random.uniform(1, 3),
        "command_sequence": json.dumps(["auth_attempt"]),
        "device_fingerprint": "unknown|" + fake.mac_address(),
        "success": random.random() < 0.05,
    })
    return r


def lateral_movement_session(entity, ts):
    unseen = [r for r in RESOURCE_POOL if r not in entity.typical_resources]
    r = _base_row(entity, ts, "lateral_movement")
    r.update({
        "source_ip": f"{entity.home_ip_prefix}.{random.randint(1,254)}",
        "geo_location": entity.home_geo[0],
        "resource_accessed": random.choice(unseen) if unseen else random.choice(RESOURCE_POOL),
        "auth_method": entity.auth_method,
        "session_duration": np.random.normal(entity.avg_session_s, entity.avg_session_s * 0.3),
        "command_sequence": json.dumps(["list", "read", "write", "delete", "query"]),
        "device_fingerprint": entity.device_fp,
    })
    return r


def device_spoofing_session(entity, ts):
    r = _base_row(entity, ts, "device_spoofing")
    r.update({
        "source_ip": f"{entity.home_ip_prefix}.{random.randint(1,254)}",
        "geo_location": entity.home_geo[0],
        "resource_accessed": random.choice(entity.typical_resources),
        "auth_method": entity.auth_method,
        "session_duration": np.random.normal(entity.avg_session_s, entity.avg_session_s * 0.2),
        "command_sequence": json.dumps(["read"]),
        "device_fingerprint": fake.user_agent() + "|" + fake.mac_address(),
    })
    return r


def low_slow_exfil_session(entity, ts, step):
    unseen = [r for r in RESOURCE_POOL if r not in entity.typical_resources]
    r = _base_row(entity, ts, "low_and_slow_exfil")
    r.update({
        "source_ip": f"{entity.home_ip_prefix}.{random.randint(1,254)}",
        "geo_location": entity.home_geo[0],
        "resource_accessed": random.choice(unseen) if unseen else random.choice(RESOURCE_POOL),
        "auth_method": entity.auth_method,
        "session_duration": np.random.normal(entity.avg_session_s * (1 + step * 0.1), 20),
        "command_sequence": json.dumps(["read", "export"]),
        "device_fingerprint": entity.device_fp,
    })
    return r


def inject_chain(entity, start_ts, chain_id):
    rows = []

    for _ in range(random.randint(8, 15)):
        r = brute_force_session(entity, start_ts + timedelta(seconds=random.randint(0, 120)))
        r["chain_id"] = chain_id
        rows.append(r)

    lm_ts = start_ts + timedelta(hours=random.uniform(3, 9))
    r = lateral_movement_session(entity, lm_ts)
    r["chain_id"] = chain_id
    rows.append(r)

    exfil_ts = start_ts + timedelta(hours=random.uniform(16, 32))
    for step in range(random.randint(3, 5)):
        r = low_slow_exfil_session(entity, exfil_ts + timedelta(hours=step * 4), step)
        r["chain_id"] = chain_id
        rows.append(r)

    return rows


def inject_campaign(entities, start_ts, campaign_id):
    n_entities = random.randint(4, 7)
    selected = random.sample(entities, min(n_entities, len(entities)))
    shared_geo = random.choice(GEO_POOL)[0]
    shared_ip_prefix = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}"

    rows = []
    for entity in selected:
        ts = start_ts + timedelta(minutes=random.randint(0, 27))
        if random.random() < 0.6:
            for _ in range(random.randint(5, 12)):
                r = brute_force_session(entity, ts + timedelta(seconds=random.randint(0, 60)))
                r["geo_location"] = shared_geo
                r["source_ip"] = f"{shared_ip_prefix}.{random.randint(1, 254)}"
                r["campaign_id"] = campaign_id
                rows.append(r)
        else:
            r = credential_stuffing_session(entity, ts, f"{shared_ip_prefix}.{random.randint(1, 254)}")
            r["geo_location"] = shared_geo
            r["campaign_id"] = campaign_id
            rows.append(r)
    return rows


def generate_dataset(n_days=14, sessions_per_entity_per_day=6, anomaly_rate=0.02, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    entities = make_entities()
    start = datetime(2026, 1, 1)
    rows = []

    n_chains = n_days // 2
    chain_entities = random.sample(entities, n_chains)
    chain_schedule = {}
    for i, entity in enumerate(chain_entities):
        chain_day = i * 2
        chain_ts = start + timedelta(days=chain_day, hours=random.randint(1, 20), minutes=random.randint(0, 59))
        chain_schedule[chain_day] = (entity, chain_ts, f"chain_{i+1:03d}")

    campaign_days = sorted(random.sample(range(2, n_days - 2), min(3, n_days - 4)))
    campaign_schedule = {}
    for j, camp_day in enumerate(campaign_days):
        camp_ts = start + timedelta(days=camp_day, hours=random.randint(0, 21), minutes=random.randint(0, 59))
        campaign_schedule[camp_day] = (camp_ts, f"camp_{j+1:03d}")

    for day in range(n_days):
        for entity in entities:
            n_sessions = np.random.poisson(sessions_per_entity_per_day)
            for _ in range(n_sessions):
                hour = random.choice(entity.typical_hours) if random.random() > 0.1 else random.randint(0, 23)
                ts = start + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
                rows.append(normal_session(entity, ts))

        n_anomalies = max(1, int(len(entities) * anomaly_rate))
        for _ in range(n_anomalies):
            entity = random.choice(entities)
            ts = start + timedelta(days=day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            pattern = random.choice([
                "brute_force", "impossible_travel", "credential_stuffing",
                "lateral_movement", "device_spoofing", "low_and_slow_exfil",
            ])
            if pattern == "brute_force":
                for _ in range(random.randint(5, 15)):
                    rows.append(brute_force_session(entity, ts + timedelta(seconds=random.randint(0, 60))))
            elif pattern == "impossible_travel":
                rows.append(normal_session(entity, ts))
                rows.append(impossible_travel_session(entity, ts + timedelta(minutes=random.randint(5, 30))))
            elif pattern == "credential_stuffing":
                shared_ip = f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
                for _ in range(random.randint(8, 20)):
                    e2 = random.choice(entities)
                    rows.append(credential_stuffing_session(e2, ts + timedelta(seconds=random.randint(0, 120)), shared_ip))
            elif pattern == "lateral_movement":
                rows.append(lateral_movement_session(entity, ts))
            elif pattern == "device_spoofing":
                rows.append(device_spoofing_session(entity, ts))
            elif pattern == "low_and_slow_exfil":
                for step in range(random.randint(3, 6)):
                    rows.append(low_slow_exfil_session(entity, ts + timedelta(days=step), step))

        if day in chain_schedule:
            entity, chain_ts, chain_id = chain_schedule[day]
            rows.extend(inject_chain(entity, chain_ts, chain_id))

        if day in campaign_schedule:
            camp_ts, campaign_id = campaign_schedule[day]
            rows.extend(inject_campaign(entities, camp_ts, campaign_id))

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["success"] = df["success"].fillna(True)
    df["chain_id"] = df["chain_id"].fillna("")
    df["campaign_id"] = df["campaign_id"].fillna("")
    return df


if __name__ == "__main__":
    df = generate_dataset(n_days=14)
    df.to_csv("data/access_logs.csv", index=False)
    print(df.shape)
    print(df["label"].value_counts())
    print("\nChains injected:", df[df["chain_id"] != ""]["chain_id"].nunique())
    print("Campaigns injected:", df[df["campaign_id"] != ""]["campaign_id"].nunique())
