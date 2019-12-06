"""
Networks architectures for critic/value function.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Fully connected policy/actor network model
    """

    def __init__(self, state_size, action_size, n_agents=1,
                 fcs1_units=256, fc2_units=128, fcs3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.fcs1 = nn.Linear(state_size * n_agents, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + (action_size * n_agents), fc2_units)
        self.fc3 = nn.Linear(fc2_units, fcs3_units)
        self.fc4 = nn.Linear(fcs3_units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
