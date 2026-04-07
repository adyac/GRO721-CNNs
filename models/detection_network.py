import torch
import torch.nn as nn

def build_detection_model():
    """
    Detection Architecture
    Input:  (1, 53, 53) - Grayscale images
    Output: (3, 7) - 3 detections × [confidence, x, y, size, score_circle, score_triangle, score_cross]

    Spatial progression:
      53 → MaxPool → 26 → MaxPool → 13 → stride-2 → 7 → stride-2 → 4
    Feature map after backbone: 64 × 4 × 4 = 1024 values

    Channels reduced (80→64) and dropout increased (0.1→0.3) to reduce overfitting.
    Previous run showed train mAP ~0.70 vs val mAP ~0.30 — classic overfitting.

    Parameters: ~265k (within 400k limit)
    """
    model = nn.Sequential(
        # 53×53 → 53×53
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1),
        nn.Dropout2d(0.2),

        # 53×53 → 26×26
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout2d(0.2),

        # 26×26 → 13×13
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout2d(0.3),

        # 13×13 → 7×7
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),
        nn.Dropout2d(0.3),

        # 7×7 → 4×4
        nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1),

        # Head: 64×4×4 = 1024 → 3×7
        nn.Flatten(),
        nn.Linear(1024, 256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.4),
        nn.Linear(256, 3 * 7),

        # Output: (batch, 3, 7)
        nn.Unflatten(1, (3, 7))
    )
    return model
