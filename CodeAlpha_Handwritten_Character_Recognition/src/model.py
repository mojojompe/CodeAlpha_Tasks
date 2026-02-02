import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(CharCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)
        
        # Fully Connected Layers
        # Image is 28x28
        # After pool1 (conv1) -> 14x14
        # After pool2 (conv2) -> 7x7
        # After pool3 (conv3) -> 3x3  (Wait, three pools might be too much for 28x28 if not padded carefully or if kernel size/stride differs)
        
        # Let's adjust:
        # Input: 28x28
        # Conv1 -> 28x28 -> Pool -> 14x14
        # Conv2 -> 14x14 -> Pool -> 7x7
        # Conv3 -> 7x7 -> Pool -> 3x3 (since 7/2 = 3.5, floor is 3)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
