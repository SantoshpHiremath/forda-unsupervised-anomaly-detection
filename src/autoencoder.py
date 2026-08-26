"""
Unsupervised anomaly detection via a PyTorch autoencoder.

The core idea, and why this is genuinely unsupervised: the autoencoder
is trained ONLY on signals from the normal class, and its training loop
never sees the label array at all -- it only ever sees X restricted to
normal rows. At evaluation time it is shown the full mixed dataset
(normal + abnormal) it was never trained on, and anomaly scoring is
purely a function of reconstruction error: a signal the model
reconstructs well is "normal-shaped," one it reconstructs poorly is
anomalous. Labels are used only afterward, to score how well this
worked -- never as a training signal. That is the actual distinction
between this and supervised classification (e.g. the logistic-regression
classifiers elsewhere in this portfolio, which train directly on labels).
"""

import numpy as np
import torch
import torch.nn as nn


class SignalAutoencoder(nn.Module):
    def __init__(self, n_timesteps=500, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_timesteps, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, n_timesteps),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def _standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std


def train_autoencoder(X_normal_train, n_timesteps=500, latent_dim=16,
                       epochs=60, lr=1e-3, batch_size=64, seed=0):
    """
    Train an autoencoder using ONLY normal-class signals.
    Returns (model, mean, std) -- mean/std are the training-set
    standardization stats, needed to normalize any data scored later.
    """
    torch.manual_seed(seed)
    X_norm, mean, std = _standardize(X_normal_train)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    model = SignalAutoencoder(n_timesteps=n_timesteps, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = X_tensor.shape[0]
    model.train()
    for _epoch in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            batch = X_tensor[batch_idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    return model, mean, std


def reconstruction_error(model, X, mean, std):
    """
    Per-sample reconstruction error (mean squared error) for X, using
    the SAME standardization stats the model was trained with. Higher
    error = less "normal-shaped" = more anomalous.
    """
    X_norm, _, _ = _standardize(X, mean=mean, std=std)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()
    return errors


def anomaly_scores_to_predictions(scores, threshold):
    """scores above threshold -> predicted anomalous (1), else normal (-1)."""
    return np.where(scores > threshold, 1, -1)
