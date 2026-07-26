import os
import json
import numpy as np
import pandas as pd
import torch

from data_generator import generate_dataset
from graph import build_access_graph
from features import build_entity_history, engineer_features, FEATURE_COLUMNS
from sequences import make_sequences, WINDOW
from detector import train_autoencoder, reconstruction_scores
from classify_explain import train_classifier, classify_and_explain
from severity import compute_severity, identify_sensitive_resources
from mitre import get_mitre
from streaming import stream_events
from evaluate import save_pr_roc_curves, save_confusion_matrix, compute_fp_comparison, generate_full_report
from kill_chain import KillChainEngine
from fingerprint import encode_sequences, build_entity_fingerprints, detect_behavioral_drift_alerts
from campaign import assign_campaign_ids, compute_campaign_summary

RNG_SEED = 42
ALERT_BUDGET_PCT = 0.01
STREAMING_DEMO_N_EVENTS = 200
STREAMING_DEMO_DELAY = 0.05

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def main():
    print("== 1. Generating synthetic access logs ==")
    df = generate_dataset(n_days=14)
    df.to_csv("data/access_logs.csv", index=False)
    print(f"   {df.shape[0]} rows | label dist:")
    print(df["label"].value_counts().to_string())
    chains_n = df["chain_id"].replace("", pd.NA).dropna().nunique()
    camps_n = df["campaign_id"].replace("", pd.NA).dropna().nunique()
    print(f"   {chains_n} multi-stage chains | {camps_n} coordinated campaigns injected")

    print("\n== 2. Building entity-resource access graph ==")
    access_graph = build_access_graph(df)
    print(f"   Graph: {access_graph.number_of_nodes()} nodes, {access_graph.number_of_edges()} edges")

    print("\n== 3. Fitting kill chain engine on chain sequences ==")
    kill_engine = KillChainEngine()
    kill_engine.fit(df)
    print("   Transition matrix learned:")
    for from_s, to_dict in kill_engine.transition_matrix.items():
        for to_s, prob in to_dict.items():
            print(f"     {from_s:20s} -> {to_s:20s}  p={prob:.3f}")
    kill_engine.save("models/kill_chain.json")

    print("\n== 4. Building entity history (fixed baseline) ==")
    history_fixed = build_entity_history(df, rolling_days=None)

    print("\n== 5. Engineering features ==")
    feat_df_fixed = engineer_features(df, history_fixed, access_graph)

    print("\n== 6. Building sequences ==")
    X, meta = make_sequences(feat_df_fixed, FEATURE_COLUMNS, window=WINDOW)
    train_mask = meta["label"] == "normal"
    X_train = X[train_mask.values]
    print(f"   Total sequences: {len(X)}, normal-only training: {len(X_train)}")

    print("\n== 7. Training LSTM autoencoder ==")
    model = train_autoencoder(X_train, n_features=len(FEATURE_COLUMNS), epochs=12)
    torch.save(model.state_dict(), "models/detector.pt")

    print("\n== 8. Scoring all sequences (fixed baseline) ==")
    scores_fixed = reconstruction_scores(model, X)
    meta["anomaly_score"] = scores_fixed

    print("\n== 9. Extracting behavioral fingerprints from encoder ==")
    embeddings = encode_sequences(model, X)
    fingerprints = build_entity_fingerprints(embeddings, meta)
    print(f"   Fingerprints built for {len(fingerprints)} entities")

    print("\n== 10. Building rolling 30-day history for FP comparison ==")
    history_rolling = build_entity_history(df, rolling_days=30)
    feat_df_rolling = engineer_features(df, history_rolling, access_graph)
    X_rolling, meta_rolling = make_sequences(feat_df_rolling, FEATURE_COLUMNS, window=WINDOW)
    scores_rolling = reconstruction_scores(model, X_rolling)
    meta_rolling["anomaly_score"] = scores_rolling

    print("\n== 11. Training anomaly-type classifier ==")
    clf, explainer = train_classifier(feat_df_fixed)

    print("\n== 12. Selecting alert budget + classifying + explaining ==")
    score_95th = float(np.quantile(scores_fixed, 0.95))
    sensitive_resources = identify_sensitive_resources(df)
    threshold = np.quantile(scores_fixed, 1 - ALERT_BUDGET_PCT)
    alert_mask = meta["anomaly_score"] >= threshold
    alerts_meta = meta[alert_mask].copy()

    feat_df_reset = feat_df_fixed.reset_index(drop=True)
    results = []

    for _, row in alerts_meta.iterrows():
        feat_row = feat_df_reset.loc[row["df_index"]]

        try:
            explain = classify_and_explain(clf, explainer, feat_row, FEATURE_COLUMNS)
        except Exception as exc:
            print(f"   [warn] explain failed for {row['entity_id']}: {exc}")
            continue

        kill_engine.update(row["entity_id"], explain["predicted_type"], row["timestamp"])
        kc_state = kill_engine.get_full_state(row["entity_id"])

        off_hours = bool(feat_row.get("hour_deviation", 0) == 1)
        new_geo = bool(feat_row.get("new_geo", 0) == 1)
        auth_failed = bool(feat_row.get("auth_failed", 0) == 1)

        severity_score, severity_tier = compute_severity(
            anomaly_score=float(row["anomaly_score"]),
            score_95th=score_95th,
            entity_type=str(feat_row.get("entity_type", "user")),
            resource=str(feat_row.get("resource_accessed", "")),
            sensitive_resources=sensitive_resources,
            off_hours=off_hours,
            new_geo=new_geo,
            auth_failed=auth_failed,
        )

        mitre_info = get_mitre(explain["predicted_type"])

        results.append({
            "entity_id": row["entity_id"],
            "entity_type": feat_row.get("entity_type", "unknown"),
            "timestamp": str(row["timestamp"]),
            "anomaly_score": round(float(row["anomaly_score"]), 6),
            "severity_score": severity_score,
            "severity_tier": severity_tier,
            "true_label": row["label"],
            "predicted_type": explain["predicted_type"],
            "confidence": explain["confidence"],
            "reasons": "; ".join(explain["reasons"]),
            "shap_values": json.dumps(explain.get("shap_values", {})),
            "mitre_id": mitre_info["technique_id"],
            "mitre_name": mitre_info["technique_name"],
            "mitre_tactic": mitre_info["tactic"],
            "kill_chain_stage": kc_state["current_stage"],
            "next_stage": kc_state["next_stage"] or "",
            "next_stage_prob": kc_state["next_stage_prob"],
            "escalation_warning": kc_state["escalation_warning"],
            "campaign_id": "",
            "drift_score": 0.0,
            "geo_location": str(feat_row.get("geo_location", "")),
            "source_ip": str(feat_row.get("source_ip", "")),
        })

    print("\n== 13. Detecting behavioral drift alerts ==")
    drift_alerts, drift_scores, drift_threshold = detect_behavioral_drift_alerts(
        embeddings=embeddings,
        meta_df=meta,
        feat_df=feat_df_fixed,
        fingerprints=fingerprints,
        recon_scores=scores_fixed,
        recon_alert_threshold=float(threshold),
    )
    print(f"   Drift threshold: {drift_threshold:.5f} | behavioral_drift alerts: {len(drift_alerts)}")
    results.extend(drift_alerts)

    alert_df = pd.DataFrame(results).sort_values("severity_score", ascending=False).reset_index(drop=True)

    print("\n== 14. Assigning campaign IDs ==")
    alert_df = assign_campaign_ids(alert_df)
    camp_summary = compute_campaign_summary(alert_df)
    n_campaigns = len(camp_summary)
    print(f"   {n_campaigns} campaigns detected")
    if n_campaigns > 0:
        print(camp_summary[["campaign_id", "campaign_type", "entities_involved", "campaign_risk_score"]].to_string(index=False))
    camp_summary.to_csv("data/campaigns.csv", index=False)

    alert_df.to_csv("data/alerts.csv", index=False)
    print(f"\n   Generated {len(alert_df)} ranked alerts "
          f"({sum(alert_df['predicted_type'] == 'behavioral_drift')} behavioral_drift)")

    print("\n== 15. Running evaluation + saving plots ==")
    curve_stats = save_pr_roc_curves(meta, output_dir="data")
    save_confusion_matrix(alert_df, output_dir="data")
    fp_comparison = compute_fp_comparison(meta, scores_fixed, meta_rolling, scores_rolling, ALERT_BUDGET_PCT)
    report = generate_full_report(meta, scores_fixed, alert_df, curve_stats, fp_comparison, ALERT_BUDGET_PCT, "data")
    print(json.dumps(report, indent=2))

    print(f"\n== 16. Streaming demo ({STREAMING_DEMO_N_EVENTS} events) ==")
    demo_df = df.sort_values("timestamp").tail(STREAMING_DEMO_N_EVENTS).reset_index(drop=True)
    n_streamed = 0
    n_flagged = 0
    for streamed_alert in stream_events(
        df=demo_df,
        history=history_fixed,
        access_graph=access_graph,
        clf=clf,
        explainer=explainer,
        model=model,
        score_95th=score_95th,
        sensitive_resources=sensitive_resources,
        batch_delay=STREAMING_DEMO_DELAY,
    ):
        n_streamed += 1
        if streamed_alert["severity_tier"] in ("High", "Critical"):
            n_flagged += 1
            print(
                f"   [STREAM] {streamed_alert['timestamp']} | {streamed_alert['entity_id']} | "
                f"{streamed_alert['severity_tier']} | {streamed_alert['mitre_id']} {streamed_alert['predicted_type']}"
            )
        if n_streamed >= STREAMING_DEMO_N_EVENTS:
            break

    print(f"   Streamed {n_streamed} events | {n_flagged} High/Critical raised in real-time")
    print("\n== Pipeline complete ==")
    print("   data/alerts.csv          - ranked alert queue")
    print("   data/campaigns.csv       - active campaign summary")
    print("   data/eval_report.json    - evaluation report")
    print("   data/pr_curve.png        - precision-recall curve")
    print("   data/roc_curve.png       - ROC-AUC curve")
    print("   data/confusion_matrix.png- classifier confusion matrix")
    print("   models/detector.pt       - trained LSTM autoencoder")
    print("   models/kill_chain.json   - kill chain transition matrix + entity states")


if __name__ == "__main__":
    main()
