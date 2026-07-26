import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from features import FEATURE_COLUMNS

FEATURE_EXPLANATIONS = {
    "new_geo": "login from an unfamiliar geographic location",
    "new_resource": "access to a resource never touched by this entity before",
    "new_device": "device fingerprint mismatch (possible spoofing)",
    "hour_deviation": "activity outside this entity's typical hours",
    "duration_z": "session duration far outside this entity's historical norm",
    "n_commands": "unusual number of commands in the session",
    "auth_failed": "failed authentication attempt",
    "resource_hash": "atypical resource category accessed",
    "graph_distance": "access to a structurally distant resource in the entity-resource graph",
}


def train_classifier(feat_df: pd.DataFrame):
    labeled = feat_df[feat_df["label"] != "normal"]
    n_normal = min(len(labeled) * 3, len(feat_df[feat_df["label"] == "normal"]))
    normals = feat_df[feat_df["label"] == "normal"].sample(n=n_normal, random_state=42)
    train_df = pd.concat([labeled, normals])
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42)
    clf.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return clf, None


def _tree_shap_approx(clf, X_row: np.ndarray, predicted_class: str) -> dict:
    class_idx = list(clf.classes_).index(predicted_class) if predicted_class in clf.classes_ else 0
    baseline = np.zeros(len(FEATURE_COLUMNS))

    for tree in clf.estimators_:
        leaf_values = tree.tree_.value[:, 0, :]
        path = tree.decision_path(X_row).toarray()[0]
        node_ids = np.where(path)[0]
        for depth, node_id in enumerate(node_ids[:-1]):
            feat_idx = tree.tree_.feature[node_id]
            child_node = node_ids[depth + 1]
            parent_val = leaf_values[node_id][class_idx] / max(leaf_values[node_id].sum(), 1e-9)
            child_val = leaf_values[child_node][class_idx] / max(leaf_values[child_node].sum(), 1e-9)
            baseline[feat_idx] += (child_val - parent_val)

    baseline /= len(clf.estimators_)
    return {col: round(float(v), 5) for col, v in zip(FEATURE_COLUMNS, baseline)}


def classify_and_explain(clf, explainer, feat_row: pd.Series, feature_cols=FEATURE_COLUMNS):
    X = pd.DataFrame([feat_row[feature_cols].values], columns=feature_cols)
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    confidence = float(np.max(proba))

    if confidence < 0.35:
        pred = rule_based_guess(feat_row)

    try:
        X_arr = X.values.astype(np.float32)
        shap_dict = _tree_shap_approx(clf, X_arr, pred)
        sorted_pairs = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        top_reasons = [
            FEATURE_EXPLANATIONS[col]
            for col, _ in sorted_pairs
            if col in FEATURE_EXPLANATIONS
        ][:3]
    except Exception:
        shap_dict = {}
        top_reasons = _threshold_reasons(feat_row, feature_cols)

    if not top_reasons:
        top_reasons = ["subtle multi-feature deviation from entity baseline"]

    return {
        "predicted_type": pred,
        "confidence": round(confidence, 3),
        "reasons": top_reasons,
        "shap_values": shap_dict,
    }


def _threshold_reasons(feat_row: pd.Series, feature_cols) -> list:
    contributions = []
    for col in feature_cols:
        val = feat_row[col]
        if col in ("new_geo", "new_resource", "new_device", "hour_deviation", "auth_failed") and val >= 1:
            contributions.append((col, 1.0))
        elif col == "duration_z" and abs(val) > 2:
            contributions.append((col, abs(val)))
        elif col == "n_commands" and val > 3:
            contributions.append((col, val))
        elif col == "graph_distance" and val > 3:
            contributions.append((col, val))
    contributions.sort(key=lambda x: x[1], reverse=True)
    return [FEATURE_EXPLANATIONS[c] for c, _ in contributions[:3] if c in FEATURE_EXPLANATIONS]


def rule_based_guess(feat_row: pd.Series):
    if feat_row["auth_failed"] == 1 and feat_row["n_commands"] <= 1:
        return "brute_force_like"
    if feat_row["new_geo"] == 1 and feat_row["new_device"] == 0:
        return "impossible_travel_like"
    if feat_row["new_device"] == 1:
        return "device_spoofing_like"
    if feat_row["graph_distance"] > 3 and feat_row["n_commands"] >= 3:
        return "lateral_movement_like"
    if feat_row["new_resource"] == 1 and feat_row["n_commands"] >= 3:
        return "lateral_movement_like"
    if abs(feat_row["duration_z"]) > 3:
        return "low_and_slow_exfil_like"
    return "unclassified_anomaly"
