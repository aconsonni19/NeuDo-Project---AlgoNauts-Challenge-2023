import torch
import torch.nn as nn


import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, input_dim=100, num_vertices=19004):
        super(CNN1D, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        # Third Convolutional Layer
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # Fourth Convolutional Layer
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # Fifth Convolutional Layer
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # Sixth Convolutional Layer
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # Adaptive Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(512, num_vertices)

        # Dropout Layer (Regularization).
        self.dropout = nn.Dropout(0.3)

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolutions
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        # Apply Adaptive Average Pooling
        x = self.pool(x).squeeze(-1)  # Squeeze removes the last dimension
        # Apply dropout for regularization
        x = self.dropout(x)
        # Fully connected layer for final predictions
        x = self.fc(x)
        return x
