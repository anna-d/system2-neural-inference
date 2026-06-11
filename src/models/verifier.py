import torch
import torch.nn as nn

from .cnn import CNNClassifier


class Verifier(nn.Module):
    """Class-conditional binary verifier: V(x, y) -> logit for p(valid | x, y).

    The backbone is a CNNClassifier (same architecture as System 1) so its weights
    can be initialised from a trained baseline ("shared backbone"). The candidate
    class y enters through an embedding that is concatenated with the image features
    before a small verification head produces a single logit (use with BCEWithLogitsLoss).
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        input_channels: int = 3,
        embed_dim: int = 64,
        head_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = CNNClassifier(
            hidden_dim=hidden_dim, num_classes=num_classes, input_channels=input_channels
        )
        self.class_embedding = nn.Embedding(num_classes, embed_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(head_dim, 1),
        )

    def load_backbone(self, state_dict) -> None:
        """Initialise the backbone from a trained System 1 checkpoint."""
        self.backbone.load_state_dict(state_dict)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze (False) or fine-tune (True) the shared backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.extract_features(x)        # (B, hidden_dim)
        emb = self.class_embedding(y)                    # (B, embed_dim)
        h = torch.cat([feats, emb], dim=1)               # (B, hidden_dim + embed_dim)
        return self.head(h).squeeze(1)                   # (B,) raw logits
