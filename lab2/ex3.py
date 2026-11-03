import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF

X, _ = make_blobs(
    n_samples=[200, 100],
    centers=[(-10, -10), (10, 10)],
    cluster_std=[2, 6],
    n_features=2,
    random_state=42
)

def plot_points(ax, data, labels, title):
    normal = labels == 0
    anomalies = labels == 1
    ax.scatter(data[normal, 0], data[normal, 1], color='blue', label='Normal')
    ax.scatter(data[anomalies, 0], data[anomalies, 1], color='red', label='Anomaly')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()


for k in [5, 20, 50]:
    knn = KNN(n_neighbors=k, contamination=0.07)
    lof = LOF(n_neighbors=k, contamination=0.07)
    knn.fit(X)
    lof.fit(X)

    knn_pred = knn.labels_
    lof_pred = lof.labels_
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    plot_points(axes[0], X, knn_pred, 'KNN: Predicted')
    plot_points(axes[1], X, lof_pred, 'LOF: Predicted')

    fig.suptitle(f'KNN vs LOF (k={k})')
    plt.tight_layout()
    plt.savefig(f'ex3_k_{k}.png', dpi=150)
    plt.close()