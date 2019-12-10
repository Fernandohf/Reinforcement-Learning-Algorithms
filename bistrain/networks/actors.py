"""
Networks architectures for Actors/Policy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCActorDiscrete(nn.Module):
    """
    Fully connected policy/actor network model for discrete action spaces
    """

    def __init__(self, state_size, action_size, seed,
                 hidden_sizes=(128, 64), hidden_activation='relu',
                 output_actions='tanh', output_scale=2):
        """
        Initialize parameters and build model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        seed: int
            Random seed
        hidden_units: iterable
            Iterable with the hidden units dimensions
        TODO
        """
        super().__init__()
        # Set seed
        self.seed = torch.manual_seed(seed)

        # Model
        self.layers = nn.ModuleList()
        layers_sizes = [state_size] + list(hidden_sizes) + [action_size]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))

        # Gaussian distribution std
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        """
        Build an actor network that maps states to actions
        """
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))


class FCActorContinuous(nn.Module):
    """
    Fully connected policy/actor network model for continuos actions spaces.
    """

    def __init__(self, state_size, action_size,
                 hidden_sizes=(128, 64), seed=42, hidden_activation='relu',
                 output_actions='tanh', output_scale=2):
        """
        Initialize parameters and build model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        seed: int
            Random seed
        hidden_units: iterable
            Iterable with the hidden units dimensions
        TODO
        """
        super().__init__()
        # Set seed
        self.seed = torch.manual_seed(seed)

        # Model
        self.layers = nn.ModuleList()
        layers_sizes = [state_size] + list(hidden_sizes) + [action_size]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))

        # Gaussian distribution std
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        """
        Build an actor network that maps states to actions
        """
        x = state
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(self.fc3(x))
