import argparse
import time

from src.utils.data import SUPPORTED_DATASETS, flatten_dataset, get_data_loaders, get_dataset_spec
from src.utils.knn import compute_centroids, ncc_predict, knn_predict


def main():
    parser = argparse.ArgumentParser(description="Run NCC and kNN baselines on CIFAR-10, CIFAR-100, or SVHN")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=SUPPORTED_DATASETS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples", type=int, default=2000)
    args = parser.parse_args()

    spec = get_dataset_spec(args.dataset)
    print(f"Loading {spec.name} dataset...")

    _, _, train_dataset, test_dataset = get_data_loaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
    )

    print(f"Flattening {args.train_samples} train samples and {args.test_samples} test samples...")
    x_train, y_train = flatten_dataset(train_dataset, max_samples=args.train_samples)
    x_test, y_test = flatten_dataset(test_dataset, max_samples=args.test_samples)

    print("\n--- Nearest Class Centroid (NCC) ---")
    start = time.time()
    centroids = compute_centroids(x_train, y_train, num_classes=spec.num_classes)
    y_pred_ncc = ncc_predict(x_test, centroids)
    acc_ncc = (y_pred_ncc == y_test).mean()
    print(f"NCC Accuracy: {acc_ncc * 100:.2f}%")
    print(f"Time: {time.time() - start:.2f} sec\n")

    print("--- 1-NN ---")
    start = time.time()
    y_pred_1nn = knn_predict(x_train, y_train, x_test, k=1)
    acc_1nn = (y_pred_1nn == y_test).mean()
    print(f"1-NN Accuracy: {acc_1nn * 100:.2f}%")
    print(f"Time: {time.time() - start:.2f} sec\n")

    print("--- 3-NN ---")
    start = time.time()
    y_pred_3nn = knn_predict(x_train, y_train, x_test, k=3)
    acc_3nn = (y_pred_3nn == y_test).mean()
    print(f"3-NN Accuracy: {acc_3nn * 100:.2f}%")
    print(f"Time: {time.time() - start:.2f} sec\n")

    print("Done.")


if __name__ == "__main__":
    main()
