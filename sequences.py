import numpy as np
import pandas as pd

WINDOW = 8


def make_sequences(df: pd.DataFrame, feature_cols, window=WINDOW):
    df = df.reset_index(drop=True)
    df["_orig_index"] = df.index
    sequences, meta = [], []
    for entity_id, g in df.sort_values("timestamp").groupby("entity_id"):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].values.astype(np.float32)
        n = len(feats)
        if n < window:
            pad = np.zeros((window - n, len(feature_cols)), dtype=np.float32)
            feats = np.vstack([pad, feats])
            n = window
        for i in range(window, n + 1):
            seq = feats[i - window:i]
            sequences.append(seq)
            row_idx = min(i - 1, len(g) - 1)
            meta.append({
                "entity_id": entity_id,
                "df_index": int(g.iloc[row_idx]["_orig_index"]),
                "label": g.iloc[row_idx]["label"],
                "timestamp": g.iloc[row_idx]["timestamp"],
            })
    X = np.stack(sequences)
    meta_df = pd.DataFrame(meta)
    return X, meta_df
