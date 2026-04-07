import torch
import torch.nn.functional as F


def _shift_batch(images: torch.Tensor, shift_x: int = 0, shift_y: int = 0) -> torch.Tensor:
    """Shift a batch of images using zero padding and crop back to original size."""
    if shift_x == 0 and shift_y == 0:
        return images

    n, c, h, w = images.shape
    pad_left = max(shift_x, 0)
    pad_right = max(-shift_x, 0)
    pad_top = max(shift_y, 0)
    pad_bottom = max(-shift_y, 0)

    padded = F.pad(images, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

    start_x = max(-shift_x, 0)
    start_y = max(-shift_y, 0)
    return padded[:, :, start_y:start_y + h, start_x:start_x + w]


@torch.no_grad()
def predict_with_tta(model: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run test-time augmentation and average probabilities.

    Augmentations:
    1. original image
    2. horizontal flip
    3. shift right by 2 pixels
    4. shift left by 2 pixels
    5. shift down by 2 pixels
    """
    model.eval()

    augmented_batches = [
        images,
        torch.flip(images, dims=[3]),
        _shift_batch(images, shift_x=2, shift_y=0),
        _shift_batch(images, shift_x=-2, shift_y=0),
        _shift_batch(images, shift_x=0, shift_y=2),
    ]

    probs_sum = None
    for aug_images in augmented_batches:
        logits = model(aug_images)
        probs = torch.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs

    avg_probs = probs_sum / len(augmented_batches)
    preds = avg_probs.argmax(dim=1)
    return avg_probs, preds
