from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(31)

X, _ = make_blobs(
    n_samples=[500, 500],
    centers=[(10, 0), (0, 10)],
    cluster_std=1,
    random_state=31
)

X_test = np.random.uniform(-10, 20, (1000, 2))

models = [
    ("IForest", IForest(contamination=0.02, random_state=31)),
    ("DIF [16, 8]", DIF(contamination=0.02, hidden_neurons=[16, 8], random_state=31)),
    ("DIF [64, 32]", DIF(contamination=0.02, hidden_neurons=[64, 32], random_state=31)),
    ("DIF [128, 64]", DIF(contamination=0.02, hidden_neurons=[128, 64], random_state=31)),
    ("LODA 5 bins", LODA(contamination=0.02, n_bins=5, random_state=31)),
    ("LODA 10 bins", LODA(contamination=0.02, n_bins=10, random_state=31)),
    ("LODA 20 bins", LODA(contamination=0.02, n_bins=20, random_state=31)),
    ("LODA 50 bins", LODA(contamination=0.02, n_bins=50, random_state=31))
]

shown = ["IForest", "DIF [64, 32]", "LODA 10 bins"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for name, model in models:
    model.fit(X)
    scores = model.decision_function(X_test)
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    if name in shown:
        ax = axes[shown.index(name)]
        scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis_r")
        ax.set_title(name)
        plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig("ex2_2d.png", dpi=300)

X, _ = make_blobs(
    n_samples=[500, 500],
    centers=[(0, 10, 0), (10, 0, 10)],
    cluster_std=1,
    random_state=31
)

X_test = np.random.uniform(-10, 20, (1000, 3))

fig = plt.figure(figsize=(15, 5))

for name, model in models:
    model.fit(X)
    scores = model.decision_function(X_test)
    print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    if name in shown:
        ax = fig.add_subplot(1, 3, shown.index(name) + 1, projection="3d")
        scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores, cmap="viridis_r")
        ax.set_title(name)
        plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig("ex2_3d.png", dpi=300)