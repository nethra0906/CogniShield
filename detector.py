import numpy as np
import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=32, latent_size=16):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)

    def forward(self, x):
        seq_len = x.size(1)
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])
        h_dec = self.from_latent(z).unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.decoder(h_dec)
        return out


def train_autoencoder(X_train, n_features, epochs=15, batch_size=128, lr=1e-3, device="cpu"):
    model = LSTMAutoencoder(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32)

    n = len(X_t)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_t[idx].to(device)
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"[detector] epoch {epoch+1}/{epochs} loss={total_loss/n:.5f}")
    return model


def reconstruction_scores(model, X, device="cpu", batch_size=256):
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2)).cpu().numpy()
            scores.append(err)
    return np.concatenate(scores)
