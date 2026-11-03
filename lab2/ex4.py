import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.models.combination import average, maximization
from pyod.utils.utility import standardizer

cardio = loadmat('cardio.mat')
X = cardio['X']
y = cardio['y'].ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_train, X_test = standardizer(X_train, X_test)
contamination = np.mean(y_train)

neighbors = range(30, 121, 10)
train_scores = np.zeros((len(X_train), 10))
test_scores = np.zeros((len(X_test), 10))

for i, k in enumerate(neighbors):
    model = KNN(n_neighbors=k, contamination=contamination)
    model.fit(X_train)
    train_pred = model.labels_
    test_pred = model.predict(X_test)
    train_ba = balanced_accuracy_score(y_train, train_pred)
    test_ba = balanced_accuracy_score(y_test, test_pred)
    print(f'k={k:3d} | Train BA={train_ba:.3f} | Test BA={test_ba:.3f}')
    train_scores[:, i] = model.decision_scores_
    test_scores[:, i] = model.decision_function(X_test)

train_scores, test_scores = standardizer(train_scores, test_scores)

for name, combine in [('Average', average), ('Maximization', maximization)]:
    combined_train = combine(train_scores)
    combined_test = combine(test_scores)
    threshold = np.quantile(combined_train, 1 - contamination)
    train_pred = (combined_train > threshold).astype(int)
    test_pred = (combined_test > threshold).astype(int)

    train_ba = balanced_accuracy_score(y_train, train_pred)
    test_ba = balanced_accuracy_score(y_test, test_pred)
    print(f'{name:12s} | Train BA={train_ba:.3f} | Test BA={test_ba:.3f}')