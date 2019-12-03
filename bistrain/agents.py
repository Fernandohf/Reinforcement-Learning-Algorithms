import numpy as np
import random
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from .networks.actors import FCActor
from .base import BaseAgent, BaseNoise
from utils import n_step_boostrap
# In case of being imported on notebook
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


class A2CAgent(BaseAgent):
    """
    Advantage Actor Critic (A2C) Agent that interacts and learns from the environment
    """
    def __init__(self, config_file):
        """
        Initialize an Advantage Actor Critic (A2C) Agent object.

        Parameters
        ----------
        config: str
            Configuration file
        """
        # Base class
        super().__init__(config_file)

        # Quick access
        self.state_size = self.config.STATE_SIZE
        self.action_size = self.config.ACTION_SIZE
        self.seed = random.seed(self.config.SEED)

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



import numpy as np
import random
import copy
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR


BUFFER_SIZE = int(1e5)      # Replay buffer size
BATCH_SIZE = 128            # Minibatch size
GAMMA = 0.99                # Discount factor
TAU = 1e-3                  # Soft update of target parameters
LR_ACTOR = 1e-3             # Learning rate of the actor
LR_CRITIC = 1e-3            # Learning rate of the critic
WEIGHT_DECAY = .000         # L2 weight decay
UPDATE_EVERY_N_STEPS = 5    # Number of step wait before update
UPDATE_N_TIMES = 10         # Number of updates
GRADIENT_CLIP_VALUE = 2     # Max gradient modulus for clipping
# LR_STEP_SIZE = 30         # LR step size
# LR_GAMMA = .2             # LR gamma multiplier
OU_THETA = .15              # OU noise parameters
OU_SIGMA = .1               # OU noise parameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)
        # self.actor_lr_scheduler = StepLR(self.actor_optimizer,
        #                                  step_size=LR_STEP_SIZE,
        #                                  gamma=LR_GAMMA)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)
        # self.critic_lr_scheduler = StepLR(self.critic_optimizer,
        #                                   step_size=LR_STEP_SIZE,
        #                                   gamma=LR_GAMMA)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self._step_count = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self._step_count += 1
        # self.update_every = self._step_count

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and self._step_count >= UPDATE_EVERY_N_STEPS:
            self._step_count = 0
            # Multiple updates
            for i in range(UPDATE_N_TIMES):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
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
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), GRADIENT_CLIP_VALUE)
        self.critic_optimizer.step()
        # self.critic_lr_scheduler.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), GRADIENT_CLIP_VALUE)
        self.actor_optimizer.step()
        # self.actor_lr_scheduler.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
