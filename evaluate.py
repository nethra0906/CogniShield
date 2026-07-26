import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    average_precision_score,
)

PLOT_STYLE = {
    "bg_base": "#0d1117",
    "bg_surface": "#161b22",
    "border": "#30363d",
    "text": "#8b949e",
    "accent_blue": "#388bfd",
    "accent_orange": "#d29922",
    "accent_red": "#da3633",
    "accent_green": "#3fb950",
}


def _apply_dark_style(fig, ax):
    fig.patch.set_facecolor(PLOT_STYLE["bg_base"])
    ax.set_facecolor(PLOT_STYLE["bg_surface"])
    for spine in ax.spines.values():
        spine.set_color(PLOT_STYLE["border"])
    ax.tick_params(colors=PLOT_STYLE["text"], labelsize=8)
    ax.xaxis.label.set_color(PLOT_STYLE["text"])
    ax.yaxis.label.set_color(PLOT_STYLE["text"])
    ax.title.set_color(PLOT_STYLE["text"])


def save_pr_roc_curves(meta: pd.DataFrame, output_dir: str) -> dict:
    y_true = (meta["label"] != "normal").astype(int).values
    y_score = meta["anomaly_score"].values

    ap = float(average_precision_score(y_true, y_score))
    roc_auc = float(roc_auc_score(y_true, y_score))

    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    budget_points = {}
    total = len(meta)
    for budget_pct in [0.01, 0.05, 0.10]:
        n_budget = int(total * budget_pct)
        threshold = np.quantile(y_score, 1 - budget_pct)
        mask = y_score >= threshold
        tp = int((y_true[mask]).sum())
        fp = int((1 - y_true[mask]).sum())
        fn = int(y_true[~mask].sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        budget_points[budget_pct] = {"precision": round(p, 4), "recall": round(r, 4), "n_alerts": n_budget}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(recall, precision, color=PLOT_STYLE["accent_blue"], lw=1.8, label=f"PR Curve (AP={ap:.3f})")
    for bpct, bdata in budget_points.items():
        ax.scatter(bdata["recall"], bdata["precision"], s=60, zorder=5,
                   label=f"{int(bpct*100)}% budget (P={bdata['precision']:.2f}, R={bdata['recall']:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - LSTM Anomaly Detector")
    ax.legend(fontsize=7, facecolor=PLOT_STYLE["bg_surface"], labelcolor=PLOT_STYLE["text"])
    _apply_dark_style(fig, ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(fpr, tpr, color=PLOT_STYLE["accent_blue"], lw=1.8, label=f"ROC Curve (AUC={roc_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color=PLOT_STYLE["border"], lw=1, ls="--", label="Random baseline")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve - LSTM Anomaly Detector")
    ax2.legend(fontsize=7, facecolor=PLOT_STYLE["bg_surface"], labelcolor=PLOT_STYLE["text"])
    _apply_dark_style(fig2, ax2)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return {"roc_auc": round(roc_auc, 4), "average_precision": round(ap, 4), "budget_points": budget_points}


def save_confusion_matrix(alert_df: pd.DataFrame, output_dir: str):
    anomaly_alerts = alert_df[alert_df["true_label"] != "normal"].copy()
    if len(anomaly_alerts) < 2:
        return

    anomaly_alerts["predicted_normalized"] = (
        anomaly_alerts["predicted_type"].str.replace("_like", "").str.strip()
    )
    true_labels = anomaly_alerts["true_label"]
    pred_labels = anomaly_alerts["predicted_normalized"]

    all_labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(all_labels)))
    ax.set_yticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.set_xlabel("Predicted Type")
    ax.set_ylabel("True Label")
    ax.set_title("Anomaly-Type Classifier - Confusion Matrix")

    for i in range(len(all_labels)):
        for j in range(len(all_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else PLOT_STYLE["text"],
                    fontsize=8)

    _apply_dark_style(fig, ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_fp_comparison(
    meta_fixed: pd.DataFrame,
    scores_fixed: np.ndarray,
    meta_rolling: pd.DataFrame,
    scores_rolling: np.ndarray,
    alert_budget_pct: float,
) -> dict:
    def fp_rate_at_budget(meta, scores, budget_pct):
        threshold = np.quantile(scores, 1 - budget_pct)
        alert_mask = scores >= threshold
        alerted = meta[alert_mask]
        fp = (alerted["label"] == "normal").sum()
        return round(float(fp / max(len(alerted), 1)), 4)

    fp_fixed = fp_rate_at_budget(meta_fixed, scores_fixed, alert_budget_pct)
    fp_rolling = fp_rate_at_budget(meta_rolling, scores_rolling, alert_budget_pct)

    return {
        "fp_rate_fixed_baseline": fp_fixed,
        "fp_rate_rolling_30day_baseline": fp_rolling,
        "fp_reduction_pct": round((fp_fixed - fp_rolling) / max(fp_fixed, 1e-9) * 100, 1),
    }


def generate_full_report(
    meta_fixed: pd.DataFrame,
    scores_fixed: np.ndarray,
    alert_df: pd.DataFrame,
    curve_stats: dict,
    fp_comparison: dict,
    alert_budget_pct: float,
    output_dir: str,
) -> dict:
    n_true_anomalies = int((meta_fixed["label"] != "normal").sum())
    n_caught = int((alert_df["true_label"] != "normal").sum())
    precision = round(n_caught / max(len(alert_df), 1), 4)
    recall = round(n_caught / max(n_true_anomalies, 1), 4)

    report = {
        "total_sequences": int(len(meta_fixed)),
        "true_anomalies": n_true_anomalies,
        "alert_budget_pct": alert_budget_pct,
        "alerts_raised": int(len(alert_df)),
        "true_positives_in_alerts": n_caught,
        "precision_at_budget": precision,
        "recall_at_budget": recall,
        "roc_auc": curve_stats["roc_auc"],
        "average_precision": curve_stats["average_precision"],
        "precision_recall_by_budget": curve_stats["budget_points"],
        "concept_drift_fp_comparison": fp_comparison,
    }

    with open(os.path.join(output_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report
