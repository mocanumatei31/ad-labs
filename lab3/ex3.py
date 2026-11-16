from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
import numpy as np

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

results = {"IForest": [], "DIF": [], "LODA": []}

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=31 + i, stratify=y)
    X_train, X_test = standardizer(X_train, X_test)
    contamination = y_train.mean()

    models = [
        ("IForest", IForest(contamination=contamination, random_state=31 + i)),
        ("DIF", DIF(contamination=contamination, hidden_neurons=[64, 32], n_ensemble=10, random_state=31 + i)),
        ("LODA", LODA(contamination=contamination, n_bins=10, random_state=31 + i))
    ]

    for name, model in models:
        model.fit(X_train)
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)
        ba = balanced_accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, scores)
        results[name].append((ba, auc))

for name, values in results.items():
    mean_ba, mean_auc = np.mean(values, axis=0)
    print(f"{name}: BA = {mean_ba:.4f}, ROC AUC = {mean_auc:.4f}")