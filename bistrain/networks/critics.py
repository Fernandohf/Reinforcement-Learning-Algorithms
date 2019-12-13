"""
Networks architectures for critic/value function.
"""
import torch
import torch.nn as nn


class FCCritic(nn.Module):
    """
    Fully connected policy/actor network model
    """
    def __init__(self, state_size, action_size,
                 hidden_sizes=(128, 64), seed=42, hidden_activation='relu'):
        """
        Initialize parameters and build model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        hidden_size: iterable
            Iterable with the hidden units dimensions
        seed: int
            Random seed
        hidden_activation: str
            Hidden units activation function
        """
        super().__init__()
        # Set seed
        self.seed = torch.manual_seed(seed)

        # Model
        self.layers = nn.ModuleList()
        layers_sizes = [state_size + action_size] + list(hidden_sizes) + [1]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))

        # Activation hidden
        self.hidden_activation = getattr(torch, hidden_activation)

    def forward(self, state, action):
        """
        Propagates critic network that maps
        states and actions pairs to Q-values
        """
        x = torch.cat((state, action), dim=1)
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        # No activatioin in last layer
        q = self.layers[-1](x)
        return q


class LSTMCritic(nn.Module):
    """
    Basic critic network with LSTM architecture.

    Receives a sequence of states/actions to output q-value function.
    """
    def __init__(self, state_size, action_size, num_layers=2,
                 hidden_size=128, seed=42, dropout=0):
        """
        Initialize parameters and build model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        num_layers: int
            Number os stacked LSTM layers
        hidden_size: iterable
            Iterable with the hidden units dimensions
        seed: int
            Random seed
        dropout: float
            Probability of Dropout layer between LSTMs
        """
        super().__init__()
        # Set seed
        self.seed = torch.manual_seed(seed)

        # Model
        self.lstm = nn.LSTM(state_size + action_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, state, action):
        """
        Propagates critic network that maps
        states and actions pairs to Q-values
        """
        x = torch.cat((state, action), dim=1)
        # TODO

        return x
