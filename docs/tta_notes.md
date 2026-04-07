# TTA notes

Added Test-Time Augmentation (TTA) support.

Files added:
- utils/tta.py: TTA prediction helper with average softmax aggregation.
- evaluate_tta.py: compares baseline CNN vs CNN+TTA on the test set.

Run locally after training:
python main_train.py
python evaluate_tta.py --weights cnn_cifar10.pth --hidden-dim 256
