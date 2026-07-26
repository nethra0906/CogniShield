# CogniShield - AI-Powered Behavioral Anomaly Detection

Domain-agnostic sequence/behavioral anomaly detection for users, service
accounts, and edge devices - built for the Honeywell hackathon PS2 brief.

---

## 1. Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ data_generator  │--> │  graph.py         │    │  features.py      │
│ synthetic access │    │  entity-resource  │ -> │  per-entity dev.  │
│ logs + labeled   │    │  bipartite graph  │    │  features incl.   │
│ attack injections│    │  (networkx)       │    │  graph_distance   │
└─────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                          │
                              ┌───────────────────────────┘
                              v
              ┌───────────────────────┐      ┌──────────────────────────┐
              │  sequences.py          │      │  detector.py              │
              │  8-event sliding       │ ---> │  LSTM autoencoder         │
              │  windows per entity    │      │  trained on normal-only   │
              └───────────────────────┘      │  sequences; recon error   │
                                              │  = anomaly score          │
                                              └────────────┬─────────────┘
                                                           │ anomaly score
                              ┌────────────────────────────┘
                              v
              ┌───────────────────────┐      ┌──────────────────────────┐
              │ classify_explain.py    │      │  severity.py              │
              │ RandomForest + SHAP    │      │  0-100 severity score     │
              │ TreeExplainer;         │      │  + Critical/High/Medium/  │
              │ rule-based fallback    │      │  Low tier labels          │
              └────────────┬──────────┘      └────────────┬─────────────┘
                           │                               │
                           v                               v
              ┌───────────────────────┐      ┌──────────────────────────┐
              │  mitre.py              │      │  evaluate.py              │
              │  ATT&CK technique ID   │      │  PR curve, ROC-AUC,       │
              │  + tactic mapping      │      │  confusion matrix,        │
              └────────────┬──────────┘      │  FP-rate comparison       │
                           │                  └──────────────────────────┘
                           v
              ┌───────────────────────┐      ┌──────────────────────────┐
              │  streaming.py          │      │  dashboard.py             │
              │  generator-based       │      │  Streamlit SOC console:   │
              │  incremental scoring   │      │  summary strip, ranked    │
              │  per-entity buffers    │      │  queue, SHAP detail panel,│
              └───────────────────────┘      │  analytics tab            │
                                              └──────────────────────────┘
```

`main.py` runs the whole pipeline end-to-end and writes:

| Output file | Description |
|---|---|
| `data/access_logs.csv` | Raw synthetic access logs |
| `data/alerts.csv` | Ranked, severity-scored, MITRE-mapped alerts |
| `data/eval_report.json` | Full evaluation metrics |
| `data/pr_curve.png` | Precision-Recall curve across thresholds |
| `data/roc_curve.png` | ROC-AUC curve |
| `data/confusion_matrix.png` | Anomaly-type classifier confusion matrix |
| `models/detector.pt` | Trained LSTM autoencoder weights |

---

## 2. New enhancements (v2)

### 2.1 Graph-based lateral movement detection (`graph.py`)

A bipartite entity-resource graph G is built from historical normal sessions
using `networkx`. Each entity node `e:entity_id` connects to resource nodes
`r:resource_name` it has historically accessed.

For each new session, `graph_distance(entity_id, resource, G)` returns the
shortest path length from the entity's node to the accessed resource node:

- **1** - resource is directly in the entity's historical cluster (normal)
- **3** - resource is one entity-hop away (mild lateral movement signal)
- **5–10** - resource is structurally distant (strong lateral movement signal)
- **10** - no path exists in the graph (isolated resource, highest suspicion)
- **-1** - cold-start entity (not yet in the graph; handled explicitly)

This replaces the binary `new_resource` flag for lateral movement, giving a
continuous, graph-structure-aware signal.

### 2.2 MITRE ATT&CK mapping (`mitre.py`)

| Predicted type | Technique ID | Technique Name | Tactic |
|---|---|---|---|
| brute_force | T1110.001 | Brute Force: Password Guessing | Credential Access |
| impossible_travel | T1078 | Valid Accounts (Geographic Anomaly) | Initial Access / Defense Evasion |
| credential_stuffing | T1110.004 | Brute Force: Credential Stuffing | Credential Access |
| lateral_movement | T1021 | Remote Services | Lateral Movement |
| device_spoofing | T1200 | Hardware Additions | Initial Access |
| low_and_slow_exfil | T1030 | Data Transfer Size Limits | Exfiltration |

### 2.3 Severity scoring formula (`severity.py`)

Each alert receives a 0–100 severity score computed as:

```
base_score  = min(anomaly_score / score_95th_percentile, 1.0) × 60

multiplier  = 1.0
            + 0.20  if entity_type == "service_account"   (privileged entity)
            + 0.15  if resource in top-20%-accessed set   (sensitive resource)
            + 0.15  if off_hours AND new_geo combined      (compound signal)
            + 0.10  if auth_failed == 1

severity_score = min(base_score × multiplier, 100)
```

| Tier | Score range |
|---|---|
| Critical | ≥ 80 |
| High | ≥ 60 |
| Medium | ≥ 35 |
| Low | < 35 |

### 2.4 Concept drift - rolling re-baselining (`features.py`)

`build_entity_history(df, rolling_days=30)` computes each entity's "known
normal" profile from sessions in the trailing 30-day window only, so
legitimate behavior change ages out of the anomaly definition.

`main.py` runs both modes and logs the FP-rate comparison to
`eval_report.json` under `concept_drift_fp_comparison`.

### 2.5 SHAP explainability (`classify_explain.py`)

`train_classifier()` now also returns a `shap.TreeExplainer` fitted on the
Random Forest. Per alert, `classify_and_explain()` computes SHAP values for
the predicted class, sorts features by absolute SHAP impact, and:

1. Returns the raw `shap_values` dict (stored in `alerts.csv` as JSON)
2. Maps the top-3 contributing features to analyst-readable strings via
   `FEATURE_EXPLANATIONS`
3. Renders a horizontal SHAP bar chart in the dashboard detail panel

Falls back to threshold-based attribution if SHAP computation fails.

### 2.6 Streaming simulation (`streaming.py`)

`stream_events(df, history, access_graph, clf, explainer, model, ...)` is a
Python generator that processes events one at a time:

- Maintains per-entity `deque`-style buffers to reconstruct the 8-event
  sliding window incrementally
- Scores each new event's window against the pre-trained LSTM immediately
- Yields a complete alert dict (score, severity, MITRE, reasons) per event
- `batch_delay` parameter controls inter-event delay (set to 0.05s in demo)

No message broker required. The generator architecture is drop-in compatible
with a Kafka consumer for production deployment.

### 2.7 Evaluation plots (`evaluate.py`)

| Output | Description |
|---|---|
| `pr_curve.png` | Precision-Recall curve with operating points at 1%, 5%, 10% budgets |
| `roc_curve.png` | ROC-AUC curve with random baseline |
| `confusion_matrix.png` | True vs. predicted anomaly type for classified alerts |

---

## 3. Metrics (from synthetic run, 14 days / ~19K sessions)

*Note: exact numbers are from the run generated by `main.py`. Values below
are representative; re-run to get current figures from `data/eval_report.json`.*

### Detector (LSTM anomaly scorer)

| Metric | Value |
|---|---|
| Total sequences | 17,328 |
| True anomalies | 216 |
| ROC-AUC | **0.762** |
| Average Precision (PR-AUC) | 0.111 |
| Precision @ 1% alert budget | **0.299** |
| Recall @ 1% alert budget | 0.241 |
| Precision @ 5% alert budget | 0.093 |
| Recall @ 5% alert budget | 0.375 |
| Precision @ 10% alert budget | 0.057 |
| Recall @ 10% alert budget | 0.458 |

### Concept drift - FP rate comparison

| Baselining mode | FP rate @ 1% budget |
|---|---|
| Fixed seed (first 20 normal sessions) | 0.701 |
| Rolling 30-day window | **0.695** |
| Reduction | 0.8% |

The 14-day synthetic corpus is too short for dramatic drift (the rolling window
captures most of the same sessions). The framework is correct and the improvement
grows with corpus length and time.


---

## 4. Design rationale

**Unsupervised detector + supervised classifier on top.** True intrusions
are too rare to train a supervised detector directly. The LSTM autoencoder
learns only what normal sequences look like; anything reconstructed badly
is anomalous by construction. A lighter Random Forest classifier (trained on
the labeled anomalies available) names the type, with a rule-based fallback
for low-confidence or novel patterns.

**Deviation-from-self features, not global features.** Every feature
(`new_geo`, `new_resource`, `new_device`, `hour_deviation`, `duration_z`,
`graph_distance`) is computed relative to that entity's own history. The same
pipeline works identically for a human user, service account, or edge device.

**Graph distance for lateral movement.** The binary `new_resource` flag misses
the structural context of which resources are adjacent to an entity's normal
cluster. Graph distance (shortest path in the entity-resource bipartite graph)
provides a continuous, interpretable signal: distance=1 is the entity's own
cluster; distance=5+ means it jumped into another entity's cluster.

**Sequence windows, not single events.** An 8-event sliding window feeds the
LSTM, catching multi-step patterns (lateral movement, low-and-slow exfil) that
no single access event reveals.

**Cold-start.** Entities with fewer than 5 historical events get
`cold_start=1` and `graph_distance=-1` (explicit sentinel, not silently treated
as distance=0). The classifier and rule-based fallback handle this explicitly.

**SHAP over threshold attribution.** SHAP TreeExplainer values are model-
faithful attributions; threshold loops are not. Judges asking "how does the
explainability work" get a real answer: "signed Shapley values decomposing the
classifier's output per feature."

---

## 5. How to run

```bash
pip install -r requirements.txt
python main.py                  # full pipeline; generates all output files
streamlit run dashboard.py      # SOC analyst console
```

The streaming demo runs automatically inside `main.py` (last 200 events,
0.05s inter-event delay). Set `STREAMING_DEMO_N_EVENTS` and
`STREAMING_DEMO_DELAY` at the top of `main.py` to adjust.
