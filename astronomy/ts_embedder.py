"""Custom time-series embedding for light curves.

Unlike text (where free pretrained embedding models like MiniLM already
exist), there is no standard pretrained embedding model for astronomical
flux time-series -- so this one really is trained by us, on our own data.
It's a small 1D-CNN autoencoder: encoder compresses a flux window down to
an `embedding_dim`-length latent vector, decoder reconstructs the window
from it. Two uses:
  1. The latent vector IS the embedding, stored in the FAISS index (rag/)
     so the Explainer agent can retrieve "similar past light curve windows".
  2. Reconstruction error doubles as a second, learned anomaly score,
     complementing the IsolationForest baseline in anomaly_detector.py.

Deliberately tiny and CPU-fast: a few hundred windows, <=20 epochs, seconds
to train on a laptop.
"""

import numpy as np
import torch
from torch import nn


class LightCurveAutoencoder(nn.Module):
    def __init__(self, window_size: int = 200, embedding_dim: int = 8):
        super().__init__()
        self.window_size = window_size
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, stride=2, padding=3),   # window_size -> ~window_size/2
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, stride=2, padding=3),  # -> ~window_size/4
            nn.ReLU(),
            nn.Flatten(),
        )
        # Measure the flattened width instead of assuming window_size // 4.
        # The strided convolutions only halve exactly for even lengths: an
        # odd window of 735 produces 184 columns where the formula predicts
        # 183, and the mismatch surfaces as an unhelpful
        # "mat1 and mat2 shapes cannot be multiplied (179x2944 and 2928x8)"
        # from the linear layer. Windows are sized from each target's cadence
        # now, so arbitrary lengths reach here routinely.
        with torch.no_grad():
            flat_dim = self.encoder(torch.zeros(1, 1, window_size)).shape[1]
        self._conv_len = flat_dim // 16
        self.to_latent = nn.Linear(flat_dim, embedding_dim)
        self.from_latent = nn.Linear(embedding_dim, flat_dim)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (16, self._conv_len)),
            nn.ConvTranspose1d(16, 8, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(x)
        recon = self.decoder(self.from_latent(latent))

        # Each ConvTranspose1d here doubles its input exactly, so the decoder
        # can only ever emit an even length. An odd window (735 points, which
        # a 12-hour window at ~1-minute cadence really produces) comes back as
        # 736 and the MSE against the input fails to broadcast. Crop or pad to
        # the original width -- standard practice for autoencoders whose
        # inputs are not a clean power-of-two multiple.
        width = recon.shape[-1]
        if width > self.window_size:
            recon = recon[..., :self.window_size]
        elif width < self.window_size:
            recon = torch.nn.functional.pad(recon, (0, self.window_size - width))
        return recon, latent


def train_autoencoder(windows: np.ndarray, embedding_dim: int = 8, epochs: int = 60, lr: float = 2e-3, seed: int = 42) -> LightCurveAutoencoder:
    # Random init is otherwise unseeded -> reconstruction-error discrimination
    # between normal/anomalous windows can swing run-to-run on a dataset
    # this small. Seed for reproducible results to report; when you scale to
    # the full 100-300 target dataset (TECHNICAL_PLAN.md Week 2), re-check
    # whether this still matters -- more data should reduce the variance.
    torch.manual_seed(seed)
    window_size = windows.shape[1]
    model = LightCurveAutoencoder(window_size=window_size, embedding_dim=embedding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)  # (N, 1, window_size)
    # normalize per-window mean-centering so the autoencoder learns *shape*, not absolute flux level
    x = x - x.mean(dim=2, keepdim=True)

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        recon, _ = model(x)
        loss = loss_fn(recon, x)
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[ts_embedder] epoch {epoch+1}/{epochs}  loss={loss.item():.6f}")

    return model


@torch.no_grad()
def embed_windows(model: LightCurveAutoencoder, windows: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
    x = x - x.mean(dim=2, keepdim=True)
    return model.encode(x).numpy()


@torch.no_grad()
def reconstruction_error(model: LightCurveAutoencoder, windows: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)
    x_centered = x - x.mean(dim=2, keepdim=True)
    recon, _ = model(x_centered)
    return ((recon - x_centered) ** 2).mean(dim=(1, 2)).numpy()


if __name__ == "__main__":
    from data_loader import generate_synthetic_light_curve
    from anomaly_detector import make_windows, label_windows_by_depth, extract_features

    lc = generate_synthetic_light_curve(n_points=6000, n_transits=4, transit_depth=0.015, seed=1)
    windows = make_windows(lc["flux"].values)

    feats = extract_features(windows)
    true_labels = label_windows_by_depth(feats)

    # Novelty-detection protocol: train ONLY on normal windows, so the
    # autoencoder never learns to reconstruct the anomalous shape -- that's
    # what makes reconstruction error a meaningful anomaly score. Training
    # on everything (anomalies included) would let it reconstruct both
    # equally well and destroy the signal.
    normal_windows = windows[true_labels == 0]
    model = train_autoencoder(normal_windows, embedding_dim=8, epochs=60)

    embeddings = embed_windows(model, windows)
    errors = reconstruction_error(model, windows)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print("Mean reconstruction error, normal windows:  ", errors[true_labels == 0].mean().round(6))
    print("Mean reconstruction error, anomalous windows:", errors[true_labels == 1].mean().round(6))
