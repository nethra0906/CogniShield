MITRE_MAP = {
    "brute_force": {
        "technique_id": "T1110.001",
        "technique_name": "Brute Force: Password Guessing",
        "tactic": "Credential Access",
    },
    "impossible_travel": {
        "technique_id": "T1078",
        "technique_name": "Valid Accounts (Geographic Anomaly)",
        "tactic": "Initial Access / Defense Evasion",
    },
    "credential_stuffing": {
        "technique_id": "T1110.004",
        "technique_name": "Brute Force: Credential Stuffing",
        "tactic": "Credential Access",
    },
    "lateral_movement": {
        "technique_id": "T1021",
        "technique_name": "Remote Services",
        "tactic": "Lateral Movement",
    },
    "device_spoofing": {
        "technique_id": "T1200",
        "technique_name": "Hardware Additions",
        "tactic": "Initial Access",
    },
    "low_and_slow_exfil": {
        "technique_id": "T1030",
        "technique_name": "Data Transfer Size Limits",
        "tactic": "Exfiltration",
    },
}

UNKNOWN_TECHNIQUE = {
    "technique_id": "T0000",
    "technique_name": "Unclassified Anomaly",
    "tactic": "Unknown",
}


def get_mitre(predicted_type: str) -> dict:
    normalized = predicted_type.replace("_like", "").strip()
    return MITRE_MAP.get(normalized, UNKNOWN_TECHNIQUE)
