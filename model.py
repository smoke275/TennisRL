import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer: nn.Linear) -> Tuple[float, float]:
    """Calculate initialization bounds for hidden layers using Xavier initialization."""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model with original MADDPG architecture."""

    def __init__(self, state_size: int, action_size: int, seed: int, 
                 fc1_units: int = 100, fc2_units: int = 75, fc3_units: int = 75):
        """Initialize parameters and build model.
        
        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
            seed: Random seed for reproducibility
            fc1_units: Number of nodes in first hidden layer
            fc2_units: Number of nodes in second hidden layer
            fc3_units: Number of nodes in third hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Network layers (original sizes)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        
        # Batch normalization (proven effective for first layer)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize network weights using proper schemes."""
        # Hidden layers use Xavier initialization
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        
        # Output layer uses small uniform initialization for stable initial actions
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        
        # Initialize biases to zero for symmetric learning
        self.fc1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()
        self.fc4.bias.data.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Build an actor (policy) network that maps states -> actions."""
        # Handle single state input
        state = state.unsqueeze(0) if state.dim() == 1 else state
        
        # Forward pass with original architecture
        x = self.bn1(F.relu(self.fc1(state)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # Output with tanh activation to bound actions between -1 and 1
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model with original MADDPG architecture."""

    def __init__(self, state_size: int, action_size: int, seed: int, 
                 fcs1_units: int = 100, fc2_units: int = 100):
        """Initialize parameters and build model.
        
        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
            seed: Random seed for reproducibility
            fcs1_units: Number of nodes in the first hidden layer
            fc2_units: Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Original architecture
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        
        # Batch normalization for state processing
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize network weights using proper schemes."""
        # Hidden layers use Xavier initialization
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        
        # Output layer uses small uniform initialization
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
        # Initialize biases to zero
        self.fcs1.bias.data.zero_()
        self.fc2.bias.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Process state through first layer (original DDPG approach)
        xs = self.bn1(F.relu(self.fcs1(state)))
        
        # Concatenate processed state with action (late fusion)
        x = torch.cat((xs, action), dim=1)
        
        # Process combined features
        x = F.relu(self.fc2(x))
        
        # Output Q-value (no activation)
        return self.fc3(x)