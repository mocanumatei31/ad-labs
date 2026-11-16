from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

X, _ = make_blobs(
    n_samples=500,
    centers=[(0, 0)],
    cluster_std=1,
    random_state=31
)

np.random.seed(31)

projections = np.random.multivariate_normal([0, 0], cov=np.eye(2), size=5)
projections = projections / np.linalg.norm(projections, axis=1, keepdims=True)

X_test = np.random.uniform(-3, 3, (500, 2))
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, bins in zip(axes.flat, [5, 10, 20, 50]):
    scores = np.zeros(500)

    for v in projections:

        train_proj = X @ v
        test_proj = X_test @ v

        low = min(train_proj.min(), test_proj.min()) - 1
        high = max(train_proj.max(), test_proj.max()) + 1
        counts, edges = np.histogram(train_proj, bins=bins, range=(low, high))
        probabilities = counts / len(X)

        indices = np.digitize(test_proj, edges) - 1
        scores += probabilities[indices]

    scores /= 5

    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")
    ax.set_title(f"{bins} bins")
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.savefig("anomaly_scores.png", dpi=300)