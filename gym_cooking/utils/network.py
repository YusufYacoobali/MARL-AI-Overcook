import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    This class is a neural network for policy estimation. It has two fully connected layers
    with ReLU activation and a softmax activation function to get subtask approximations.
    """
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        """Forward pass through the neural network."""
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
