import time
import numpy as np
from src.utils.data import get_cifar10_loaders, flatten_dataset
from src.utils.knn import compute_centroids, ncc_predict, knn_predict

def main():
    print("Loading CIFAR-10 dataset...")

    _, _, train_dataset, test_dataset = get_cifar10_loaders(batch_size=128)

    N_train = 10000
    N_test = 2000

    print(f"Flattening {N_train} train samples and {N_test} test samples...")
    X_train, y_train = flatten_dataset(train_dataset, max_samples=N_train)
    X_test, y_test = flatten_dataset(test_dataset, max_samples=N_test)

    print("\n--- Nearest Class Centroid (NCC) ---")
    start = time.time()
    centroids = compute_centroids(X_train, y_train, num_classes=10)
    y_pred_ncc = ncc_predict(X_test, centroids)
    acc_ncc = (y_pred_ncc == y_test).mean()
    print(f"NCC Accuracy: {acc_ncc*100:.2f}%")
    print(f"Time: {time.time() - start:.2f} sec\n")

    print("--- 1-NN ---")
    start = time.time()
    y_pred_1nn = knn_predict(X_train, y_train, X_test, k=1)
    acc_1nn = (y_pred_1nn == y_test).mean()
    print(f"1-NN Accuracy: {acc_1nn*100:.2f}%")
    print(f"Time: {time.time() - start:.2f} sec\n")

    print("--- 3-NN ---")
    start = time.time()
    y_pred_3nn = knn_predict(X_train, y_train, X_test, k=3)
    acc_3nn = (y_pred_3nn == y_test).mean()
    print(f"3-NN Accuracy: {acc_3nn*100:.2f}%")
    print(f"Time: {time.time() - start:.2f} sec\n")

    print("Done.")

if __name__ == "__main__":
    main()
