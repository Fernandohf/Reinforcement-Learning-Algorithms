"""
Networks architectures for Actors/Policy.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal, Categorical


class FCActorDiscrete(nn.Module):
    """
    Fully connected policy/actor network model for discrete action spaces
    """

    def __init__(self, state_size, action_size, hidden_sizes=(128, 64),
                 seed=42, hidden_activation='relu',
                 output_activation='log_sofmax',):
        """
        Initialize parameters and build model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        hidden_sizes: iterable
            Iterable with the hidden units dimensions
        seed: int
            Random seed
        hidden_activation: str
            Hidden units activation function
        output_activation: str
            Ouput activation function
        """
        super().__init__()
        # Set seed
        self.seed = torch.manual_seed(seed)

        # Model
        self.layers = nn.ModuleList()
        layers_sizes = [state_size] + list(hidden_sizes) + [action_size]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        
        # Activation hidden
        self.hidden_activation = getattr(F, hidden_activation)

        # Ouput activation
        self.output_activation = getattr(F, output_activation)

    def forward(self, state):
        """
        Build an actor network that maps states to actions probabilities
        """
        x = state
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        # Logsoftmax for numerical stability
        logits = self.output_activation(self.layers[-1](x))
        
        # Distribution
        dist = Categorical(logits=logits)

        return dist.sample() 


class FCActorContinuous(nn.Module):
    """
    Fully connected policy/actor network model for continuos actions spaces.
    """

    def __init__(self, state_size, action_size,
                 hidden_sizes=(128, 64), seed=42, hidden_activation='relu',
                 output_loc_activation='tanh', output_scale_activation='relu',
                 output_loc_scaler=2):
        """
        Initialize parameters and build model.

        Parameters
        ----------
        state_size: int
            Dimension of each state
        action_size: int
            Dimension of each action
        hidden_sizes: iterable
            Iterable with the hidden units dimensions
        seed: int
            Random seed
        hidden_activation: str
            Hidden units activation function
        output_activation: str
            Ouput activation function
        """
        super().__init__()
        # Set seed
        self.seed = torch.manual_seed(seed)

        # Model
        self.layers = nn.ModuleList()
        layers_sizes = [state_size] + list(hidden_sizes) + [action_size]
        for i in range(len(layers_sizes) - 1):
            self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            # duplicate last layer
            if i == len(layers_sizes) - 1:
                self.layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))    

        # Activation hidden
        self.hidden_activation = getattr(F, hidden_activation)

        # Ouput activation
        self.output_loc_activation = getattr(F, output_loc_activation)
        self.output_scale_activation = getattr(F, output_scale_activation)
        
        self.output_loc_scaler = output_loc_scaler

    def forward(self, state):
        """
        Build an actor network that maps states to actions
        """
        x = state
        for layer in self.layers[-2]:
            x = self.hidden_activation(layer(x))
        
        # Distribution
        loc = self.output_loc_activation(x) * self.output_loc_scaler
        scale = self.output_scale_activation(x)
        
        dist = Normal(loc=loc, scale=scale)
        return dist.sample()
