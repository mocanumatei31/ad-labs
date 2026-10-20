from pyod.utils.data import generate_data
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score
)

# Ex 1

X_train, X_test, y_train, y_test = generate_data(
    n_train=400,
    n_test=100,
    contamination=0.1
)

plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="green", label="normal")
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="red", label="outlier")

plt.legend()
plt.savefig("plot.png")

# Ex 2

knn = KNN(contamination=0.1)
knn.fit(X_train)

y_pred_train = knn.labels_
y_scores_train = knn.decision_scores_

y_pred_test = knn.predict(X_test)
y_scores_test = knn.decision_function(X_test)

plt.figure()
plt.scatter(X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], color="green", label="normal")
plt.scatter(X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], color="red", label="outlier")

plt.legend()
plt.savefig("knn_results.png")

tn, fp, fn, tp = confusion_matrix(y_train, y_pred_train).ravel()
balanced_acc = balanced_accuracy_score(y_train, y_pred_train)

print("Training TN, FP, FN, TP:", tn, fp, fn, tp)
print("Training Balanced accuracy:", balanced_acc)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
balanced_acc = balanced_accuracy_score(y_test, y_pred_test)

print("TN, FP, FN, TP:", tn, fp, fn, tp)
print("Balanced accuracy:", balanced_acc)

fpr, tpr, thresholds = roc_curve(y_test, y_scores_test)
auc = roc_auc_score(y_test, y_scores_test)

print("ROC AUC:", auc)

plt.figure()
plt.plot(fpr, tpr, label=f"KNN (AUC = {auc:.3f})")

plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.title("KNN ROC Curve")
plt.legend()

plt.savefig("knn_roc.png")

for contamination in [0.05, 0.1, 0.2, 0.3]:
    knn = KNN(contamination=contamination)
    knn.fit(X_train)
    y_pred_test = knn.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    balanced_acc = balanced_accuracy_score(y_test, y_pred_test)
    y_scores_test = knn.decision_function(X_test)
    auc = roc_auc_score(y_test, y_scores_test)

    print(f"\nContamination: {contamination}")
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    print(f"Balanced accuracy: {balanced_acc:.3f}")
    print(f"ROC AUC: {auc:.3f}")

# Ex 3

X_train, X_test, y_train, y_test = generate_data(
    n_train=1000,
    n_test=0,
    contamination=0.1,
    n_features=1
)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

z_score = np.abs((X_train - mean) / std)
scores = z_score.ravel()

threshold = np.quantile(scores, 0.9)

y_pred = (scores >= threshold).astype(int)

print("Threshold:", threshold)
print("Detected anomalies:", np.sum(y_pred))

plt.figure()
plt.scatter(X_train[y_pred == 0, 0], np.zeros(np.sum(y_pred == 0)), color="green", label="Normal")
plt.scatter(X_train[y_pred == 1, 0], np.zeros(np.sum(y_pred == 1)), color="red", label="Anomaly")
plt.legend()
plt.savefig("zscore_anomalies.png")

balanced_acc = balanced_accuracy_score(y_train, y_pred)
print("Balanced accuracy:", balanced_acc)

# Ex 4

n_samples = 1000
contamination = 0.1
n_features = 2

mu = np.array([3, 5])
sigma = np.array([
    [4, 1.5],
    [1.5, 2]
])

X = np.random.randn(n_samples, n_features)
L = np.linalg.cholesky(sigma)
Y = (L @ X.T + mu[:, None]).T

n_outliers = int(n_samples * contamination)
Y[:n_outliers] += np.array([10, 10])
y_true = np.zeros(n_samples, dtype=int)
y_true[:n_outliers] = 1

indices = np.random.permutation(n_samples)
Y = Y[indices]
y_true = y_true[indices]

mean = np.mean(Y, axis=0)
cov = np.cov(Y, rowvar=False)
L = np.linalg.cholesky(cov)

diff = Y - mean
transformed = np.linalg.solve(L, diff.T).T
scores = np.linalg.norm(transformed, axis=1)

threshold = np.quantile(scores, 1 - contamination)
y_pred = (scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
balanced_acc = balanced_accuracy_score(y_true, y_pred)

print("Threshold:", threshold)
print("Detected anomalies:", np.sum(y_pred))
print("TN, FP, FN, TP:", tn, fp, fn, tp)
print("Balanced accuracy:", balanced_acc)

plt.figure()
plt.scatter(Y[y_pred == 0, 0], Y[y_pred == 0, 1], color="green", label="Normal")
plt.scatter(Y[y_pred == 1, 0], Y[y_pred == 1, 1], color="red", label="Anomaly")

plt.title("Mahalanobis Dist Anomaly Detection")
plt.legend()
plt.savefig("mahalanobis.png")