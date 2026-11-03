import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

X_train, X_test, y_train, y_test = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_clusters=2,
    n_features=2,
    contamination=0.1,
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

for k in [3, 10, 30, 50]:
    knn = KNN(n_neighbors=k, contamination=0.1)
    knn.fit(X_train)
    y_train_pred = knn.labels_
    y_test_pred = knn.predict(X_test)
    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    test_ba = balanced_accuracy_score(y_test, y_test_pred)
    print(
        f"k={k:2d} | "
        f"Train BA={train_ba:.3f} | "
        f"Test BA={test_ba:.3f}"
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    plot_points(axes[0, 0], X_train, y_train, 'Train: Actual')
    plot_points(axes[0, 1], X_train, y_train_pred, 'Train: Predicted')
    plot_points(axes[1, 0], X_test, y_test, 'Test: Actual')
    plot_points(axes[1, 1], X_test, y_test_pred, 'Test: Predicted')
    axes[0, 0].legend()

    fig.suptitle(f'KNN anomaly detection (k={k})')
    plt.tight_layout()
    plt.savefig(f'knn_k_{k}.png', dpi=150)
    plt.close()