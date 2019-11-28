import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from model import FCActor, Critic, LSTMActor
from utils import n_step_boostrap
# In case of being imported on notebook
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm

# Hyperparameters
GAMMA = 0.999             # Discount factor
LR_ACTOR = 1e-4           # Learning rate of the actor
LR_CRITIC = 1e-3          # Learning rate of the critic
ACTION_MIN = -2           # Min value in continuous action
ACTION_MAX = 2            # Max value in continuous action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class A2CAgent():
    """Interacts and learns from the environment."""

    def __init__(self, state_size, action_size, actor_hidden_size=(32),
                 critic_hidden_size=(32, 16), random_seed=42, noise='gaussian'):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            actor_hidden_size (tuple): Dimension of hidden units for actor network
            critic_hidden_size (tuple): Dimension of hidden units for critic network
            random_seed (int): random seed
            noise (str): Weather use 'gaussian' or 'ornstein-uhlenbeck' noise.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network
        # self.actor = Actor(state_size, action_size, actor_hidden_size, random_seed).to(device)
        self.actor = FCActor(state_size, action_size, (128,), random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic = Critic(state_size, action_size, critic_hidden_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Noise process
        if noise == 'gaussian':
            self.noise = GaussianNoise(action_size, random_seed)
        elif noise == 'ornstein-uhlenbeck':
            self.noise = OUNoise(action_size, random_seed)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        # Forwards pass on policy
        self.actor.eval()
        with torch.no_grad():
            action, _ = self.actor(state)
            action = action.cpu().data.numpy()
        self.actor.train()
        # Noise
        if add_noise:
            action += self.noise.sample()
        # Clipped action
        return np.clip(action, ACTION_MIN, ACTION_MAX)

    def reset(self):
        self.noise.reset()

    def learn(self, states, actions, rewards, next_states, dones, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        # Check consistency
        assert(np.array_equal(states[0, 1, :], next_states[0, 0, :]))
        # Current state, actions and next_states
        n_bootstrap = next_states.shape[1]
        curr_states = torch.from_numpy(next_states[:, 0, :]).float().to(device)
        curr_actions = torch.from_numpy(actions[:, 0, :]).float().to(device)

        last_boot_next_state = torch.from_numpy(next_states[:, -1, :]).float().to(device)
        # actions_boot = torch.from_numpy(actions[:, :, 0]).float().to(device)
        last_dones_boot = torch.from_numpy(dones[:, -1, :]).float().to(device)

        # ---------------------------- Update Critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_n_next, _ = self.actor(last_boot_next_state)
        discount = gamma ** np.arange(n_bootstrap).reshape(1, -1, 1)
        rewards = (rewards * discount).sum(axis=1)

        rewards_boot = torch.from_numpy(rewards).float().to(device)
        Q_targets_next = self.critic(last_boot_next_state, actions_n_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_boot + ((gamma ** n_bootstrap) * Q_targets_next * (1. - last_dones_boot))

        # Compute critic loss
        Q_expected = self.critic(curr_states, curr_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- Update Actor ---------------------------- #
        # Compute advantage - actor loss
        _, log_action_next = (self.actor(curr_states))
        advantages = (Q_targets.detach() - Q_expected.detach())
        actor_loss = -(log_action_next * advantages).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


class GaussianNoise():
    """
    Simple scaled gaussian noise
    """
    def __init__(self, size, seed, mu=0., sigma=0.2, sigma_factor=.95):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.size = size
        self.sigma_init = sigma
        self.sigma = sigma
        self.sigma_factor = sigma_factor
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal sigma to initial sigma"""
        self.sigma = self.sigma_init

    def sample(self):
        """Update internal state and return it as a noise sample."""
        self.sigma *= self.sigma_factor
        sample = np.random.normal(self.mu, self.sigma, size=self.size)
        return sample


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


def train_a2c(mp_envs, agent, episodes=2000, n_step=5, print_every=10, max_steps=300):
    """
    Train the given agent on the parallel environments provided.

    Parameters
    ----------
    mp_envs: SubprocVecEnv
        Parallels environments
    agent: Agent
        Agent exploring the environments
    episodes: int
        Number of episodes to train
    print_every: int
        Frequency to display training metrics
    max_steps: int
        Maximum number of steps
    """
    # Saving metrics
    avg_scores_deque = deque(maxlen=print_every)
    avg_scores = []
    scores_envs = []
    # Keep track of progress
    pbar = tqdm(range(1, episodes + 1), ncols=800)
    for i_episode in pbar:
        # Reset env
        initial_states = mp_envs.reset()
        # Reset agent noise (exploration)
        agent.reset()
        score = []
        gamma = GAMMA
        for i in range(max_steps):
            # Collect trajectories
            S, A, R, Sp, dones = n_step_boostrap(mp_envs, agent,
                                                 initial_states,
                                                 n_step)
            agent.learn(S, A, R, Sp, dones, gamma)
            # Start from the next state
            initial_states = Sp[:, 0, :]
            # Collect scores from all parallel envs
            score.append(R[:, 0])
            # Update initial gamma
            gamma *= GAMMA
            if dones[:, -1].any():
                break
        # Save scores
        episode_score = np.asarray(score).sum(axis=0)
        mean_score = episode_score.mean()
        avg_scores_deque.append(mean_score)
        avg_scores.append(mean_score)
        scores_envs.append(episode_score)
        # Display some progress
        if (i_episode) % print_every == 0:
            text = '\rEpisode {}/{}\tAverage Scores: {:.2f}'.format(i_episode,
                                                                    episodes,
                                                                    np.mean(avg_scores_deque))
            pbar.set_description(text)

        # Solve environment
        # if mean(avg_scores_deque) >=

    return np.asarray(scores_envs)
