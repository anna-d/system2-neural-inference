import numpy as np

def compute_centroids(X_train, y_train, num_classes=10):
    centroids = []
    for c in range(num_classes):
        samples = X_train[y_train == c]
        centroid = samples.mean(axis=0)
        centroids.append(centroid)
    return np.stack(centroids, axis=0)


def ncc_predict(X_test, centroids):
    # Υπολογισμός αποστάσεων
    X2 = np.sum(X_test**2, axis=1, keepdims=True)
    C2 = np.sum(centroids**2, axis=1, keepdims=True).T
    XC = X_test @ centroids.T

    dists = X2 - 2*XC + C2
    preds = np.argmin(dists, axis=1)
    return preds


def knn_predict(X_train, y_train, X_test, k=1):
    # Υπολογισμός αποστάσεων X_test -> X_train
    X2 = np.sum(X_test**2, axis=1, keepdims=True)
    T2 = np.sum(X_train**2, axis=1, keepdims=True).T
    XT = X_test @ X_train.T

    dists = X2 - 2*XT + T2

    # indices των k κοντινότερων neighbors
    idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]

    preds = []
    for neighbors in idx:
        labels, counts = np.unique(y_train[neighbors], return_counts=True)
        preds.append(labels[np.argmax(counts)])

    return np.array(preds)
