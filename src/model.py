import torch
import torch.nn as nn

class StarClassifier(nn.Module):
  def __init__(self, input_dim=6, num_classes=6):
    super().__init__()

    self.fc1 = nn.Linear(input_dim, 16)
    self.fc2 = nn.Linear(16, 16)
    self.fc3 = nn.Linear(16, num_classes)

  def forward(self, x):
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return (x)