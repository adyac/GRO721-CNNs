import torch
import torch.nn as nn

def build_classification_model():

    return nn.Sequential(
        # Block 1
        nn.Conv2d(1, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(2, 2),
        
        # Block 2
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(2, 2),
        
        # Block 3
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(2, 2),
        
        # Block 4
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout2d(0.2),
        nn.MaxPool2d(2, 2),

        # FC layers with regularization
        nn.Flatten(),
        nn.Linear(128*3*3, 85),
        nn.BatchNorm1d(85),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(85, 3),
        nn.Sigmoid()
    )
