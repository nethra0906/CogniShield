import json
import numpy as np
import pandas as pd

STAGE_ORDER = ["dormant", "initial_access", "lateral_movement", "exfiltration"]
STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}

ATTACK_TO_STAGE = {
    "brute_force": "initial_access",
    "brute_force_like": "initial_access",
    "credential_stuffing": "initial_access",
    "credential_stuffing_like": "initial_access",
    "impossible_travel": "initial_access",
    "impossible_travel_like": "initial_access",
    "device_spoofing": "initial_access",
    "device_spoofing_like": "initial_access",
    "lateral_movement": "lateral_movement",
    "lateral_movement_like": "lateral_movement",
    "low_and_slow_exfil": "exfiltration",
    "low_and_slow_exfil_like": "exfiltration",
}

STAGE_NEXT_HUMAN = {
    "initial_access": "lateral movement across internal systems",
    "lateral_movement": "data staging and exfiltration",
    "exfiltration": "objective complete - data likely extracted",
}

DOMAIN_PRIORS = {
    "initial_access": {"lateral_movement": 2.0, "exfiltration": 0.5, "dormant": 0.5},
    "lateral_movement": {"exfiltration": 2.0, "dormant": 0.5},
    "exfiltration": {"dormant": 1.0},
}

class KillChainEngine:
    def __init__(self):
        self.transition_matrix = {}
        self.entity_states = {}

    def fit(self, df: pd.DataFrame):
        chain_df = df[df["chain_id"] != ""].copy()
        counts = {}

        for chain_id in chain_df["chain_id"].unique():
            g = chain_df[chain_df["chain_id"] == chain_id].sort_values("timestamp")
            stages = [ATTACK_TO_STAGE[lbl] for lbl in g["label"] if lbl in ATTACK_TO_STAGE]
            deduped = []
            for s in stages:
                if not deduped or s != deduped[-1]:
                    deduped.append(s)
            for i in range(len(deduped) - 1):
                from_s, to_s = deduped[i], deduped[i + 1]
                counts.setdefault(from_s, {})[to_s] = counts.get(from_s, {}).get(to_s, 0) + 1

        for from_s, to_counts in DOMAIN_PRIORS.items():
            counts.setdefault(from_s, {})
            for to_s, prior in to_counts.items():
                counts[from_s][to_s] = counts[from_s].get(to_s, 0) + prior

        for from_s, to_counts in counts.items():
            total = sum(to_counts.values())
            self.transition_matrix[from_s] = {k: round(v / total, 4) for k, v in to_counts.items()}

        return self

    def update(self, entity_id: str, detected_type: str, timestamp):
        stage = ATTACK_TO_STAGE.get(detected_type)
        if stage is None:
            return

        state = self.entity_states.setdefault(entity_id, {
            "stage": "dormant",
            "confidence": 0.5,
            "last_ts": str(timestamp),
            "history": [],
        })

        current_rank = STAGE_RANK.get(state["stage"], 0)
        new_rank = STAGE_RANK.get(stage, 0)

        if new_rank >= current_rank:
            state["stage"] = stage
            state["confidence"] = min(0.97, state["confidence"] * 1.1 + 0.12)
        else:
            state["confidence"] = max(0.3, state["confidence"] * 0.85)

        state["history"].append({"ts": str(timestamp), "stage": stage})
        state["last_ts"] = str(timestamp)

    def predict_next(self, entity_id: str) -> dict:
        if entity_id not in self.entity_states:
            return {"next_stage": None, "probability": 0.0, "description": ""}

        current_stage = self.entity_states[entity_id]["stage"]

        if current_stage == "exfiltration":
            return {
                "next_stage": "objective_complete",
                "probability": 1.0,
                "description": STAGE_NEXT_HUMAN["exfiltration"],
            }

        transitions = self.transition_matrix.get(current_stage, {})
        if not transitions:
            return {"next_stage": None, "probability": 0.0, "description": ""}

        best_next = max(transitions, key=transitions.get)
        return {
            "next_stage": best_next,
            "probability": round(transitions[best_next], 3),
            "description": STAGE_NEXT_HUMAN.get(current_stage, ""),
        }

    def get_full_state(self, entity_id: str) -> dict:
        if entity_id not in self.entity_states:
            return {
                "current_stage": "dormant",
                "confidence": 0.0,
                "next_stage": None,
                "next_stage_prob": 0.0,
                "escalation_warning": False,
                "stage_history": [],
            }

        state = self.entity_states[entity_id]
        pred = self.predict_next(entity_id)

        return {
            "current_stage": state["stage"],
            "confidence": round(state["confidence"], 3),
            "next_stage": pred["next_stage"],
            "next_stage_prob": pred["probability"],
            "escalation_warning": (
                pred["probability"] >= 0.5
                and state["stage"] not in ("exfiltration", "dormant")
            ),
            "stage_history": state["history"][-5:],
        }

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "transition_matrix": self.transition_matrix,
                "entity_states": self.entity_states,
            }, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            payload = json.load(f)
        engine = cls()
        engine.transition_matrix = payload["transition_matrix"]
        engine.entity_states = payload["entity_states"]
        return engine