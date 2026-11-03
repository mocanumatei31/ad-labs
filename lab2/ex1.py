import numpy as np
import matplotlib.pyplot as plt

# 1D
rng = np.random.default_rng(42)
n_regular, n_other = 80, 15
kind = np.repeat(np.arange(4), [n_regular, n_other, n_other, n_other])

x = np.concatenate([
    rng.normal(0, 1, n_regular),
    rng.normal(0, 6, n_other),
    rng.normal(0, 1, n_other),
    rng.normal(0, 6, n_other)
])
D = np.column_stack([np.ones(len(x)), x])
leverage = np.diag(D @ np.linalg.pinv(D))
top = np.argsort(leverage)[-8:]

names = ['Regular', 'High variance in X', 'High variance in Y', 'High variance in X and Y']
settings = [(0, 0.25), (0, 4), (2, 0.25), (2, 4)]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (mu, variance) in zip(axes.flat, settings):
    y = 2 * x + 1 + rng.normal(mu, np.sqrt(variance), len(x))
    y[kind >= 2] += rng.normal(0, 10, (kind >= 2).sum())
    for i in range(4):
        ax.scatter(x[kind == i], y[kind == i], label=names[i])
    ax.scatter(x[top], y[top], s=150, facecolors="none", edgecolors="black", linewidths=2, label="High leverage")
    ax.set_title(f"Mean={mu}, Variance={variance}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("leverage_1d.png")
plt.close()

# 2D
rng = np.random.default_rng(42)
X = np.vstack([
    rng.normal(0, 1, (n_regular, 2)),
    rng.normal(0, 6, (n_other, 2)),
    rng.normal(0, 1, (n_other, 2)),
    rng.normal(0, 6, (n_other, 2))
])
D = np.column_stack([np.ones(len(X)), X])
leverage = np.diag(D @ np.linalg.pinv(D))
top = np.argsort(leverage)[-8:]

fig = plt.figure(figsize=(12, 9))

for j, (mu, variance) in enumerate(settings):
    y = 2 * X[:, 0] - X[:, 1] + 1
    y += rng.normal(mu, np.sqrt(variance), len(X))
    y[kind >= 2] += rng.normal(0, 10, (kind >= 2).sum())
    ax = fig.add_subplot(2, 2, j + 1, projection="3d")
    for i in range(4):
        ax.scatter(X[kind == i, 0], X[kind == i, 1], y[kind == i], label=names[i])
    ax.scatter(X[top, 0], X[top, 1], y[top], s=150, facecolors="none", edgecolors="black", linewidths=2, label="High leverage")
    ax.set_title(f"Mean={mu}, Variance={variance}")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    if j == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("leverage_2d.png")
plt.close()