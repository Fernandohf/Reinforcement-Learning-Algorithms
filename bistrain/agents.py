import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from .base.base_agent import BaseAgent
from .networks.actors import FCActorContinuous
from .networks.critics import FCCritic
from .utils.bootstrap import n_step_boostrap

# In case of being imported on notebook
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm



class A2CAgent(BaseAgent):
    """
    Advantage Actor Critic (A2C) Agent
    """

    def __init__(self, config_file):
        """
        Initialize an Advantage Actor Critic (A2C) Agent object.

        Parameters
        ----------
        config_file: str
            Path to configuration file
        """
        # Base class
        super().__init__(config_file)

        # Actor Network
        self.actor = self._set_policy()
        self.actor_optimizer = self._set_optimizer(self.actor.parameters())

        # Critic Network
        self.actor = self._set_val_func()
        self.critic_optimizer = self._set_optimizer(self.critic.parameters())

        # Noise process
        self.noise = self._set_noise()

        # Reset current status
        self.reset()

    def act(self, state, explore=True):
        """
        Returns actions for given state and current policy.

        Parameters
        ----------
        state: torch.Tensor
            Current states
        explore: bool
            Weather or not add noise to the actions
        """
        state = torch.from_numpy(state).float().to(self.config.DEVICE)

        # Forwards pass on policy
        self.actor.eval()

        # Continuous actions
        if self.config.ACTION_SPACE == "continuous":
            with torch.no_grad():
                action, _ = self.actor(state)
                action = action.cpu().data.numpy()
            self.actor.train()
            # Noise
            if explore:
                action += self.noise.sample()
            # Clipped action
            action = np.clip(action,
                             *self.config.ACTION_RANGE)

        # Discrete Actions
        elif self.config.ACTION_SPACE == "discrete":
            with torch.no_grad():
                action, _ = self.actor(state)
                action = action.cpu().data.numpy()
            self.actor.train()
            # Noise
            if explore:
                action = self.noise.sample(action)

        return action

    def reset(self):
        """
        Reset the current learning episode
        """
        # Noise scalling
        self.noise.reset()
        # Episode parameters
        self._gamma = self.config.GAMMA
        self._initial_states = None
        # Activate training section
        self.config.activate_subsection("TRAINING")

    def step(self, envs):
        """
        Records experiences (S, A, R, S', dones) and learns from them.

        Parameters
        ----------
        envs: Gym.Environment
            Open ai compatible GYM environment

        Returns
        -------
        done: bool
            Return wether the episode is done or not
        scores: array
            Rewards for each parallel environment
        """
        # If first step
        if self._initial_states is None:
            self._initial_states = envs.reset()
        # Unroll trajectories of parallel envs
        s, a, r, sp, dones = n_step_boostrap(envs, self,
                                             self._initial_states,
                                             n_step=self.config.N_STEP_BS)
        self._learn(s, a, r, sp, dones, self._gamma)
        # Start from the next state
        self._initial_states = sp[:, 0, :]
        # Collect scores from all parallel envs and if any has finished
        scores = r[:, 0]
        done = dones[:, -1].any()
        # Update initial gamma
        self._gamma *= self.config.GAMMA

        return scores, done

    def _learn(self, states, actions, rewards, next_states, dones, gamma):
        """
        Update policy and value parameters using given batch of trajectories.

        Parameters
        ----------
        states: array
            States across bootstraps environments
        actions: array
            Actions across bootstraps and environments
        rewards: array
            Rewards across bootstraps and environments
        next_states: array
            Next states across bootstrap and environments
        dones: array
            Boolean array masking finished trajectories
        gamma: float
            Current discount factor
        """
        # Check consistency
        assert(np.array_equal(states[0, 1, :], next_states[0, 0, :]))

        # Current state, actions and next_states
        curr_states = torch.from_numpy(states[:, 0, :]).float()
        curr_states = curr_states.to(self.config.DEVICE)
        curr_actions = torch.from_numpy(actions[:, 0, :]).float()
        curr_actions = curr_actions.to(self.config.DEVICE)

        # n step bootstrap
        n_step_bs = next_states.shape[1]

        # Last bootstrapped state
        last_bs_nstate = torch.from_numpy(next_states[:, -1, :])
        last_bs_nstate = last_bs_nstate.float().to(self.config.DEVICE)

        # If they are ending acions or not
        last_bs_dones = torch.from_numpy(dones[:, -1, :])
        last_bs_dones = last_bs_dones.float().to(self.config.DEVICE)

        # ----------------------- Update Critic ----------------------- #
        # Get predicted actions from last bootstrapped next-state
        actions_bs_next, _ = self.actor(last_bs_nstate)
        # Discounted bootstraped rewards
        discount = gamma ** np.arange(n_step_bs).reshape(1, -1, 1)
        rewards_bs = torch.from_numpy((rewards * discount).sum(axis=1))
        rewards_bs = rewards_bs.float().to(self.config.DEVICE)
        # Predicted value function
        Q_next = self.critic(last_bs_nstate, actions_bs_next)
        # Compute Q targets for current states (y_i)
        Q_target = rewards_bs + ((gamma ** n_step_bs) * Q_next *
                                 (1. - last_bs_dones))
        # Compute Critic loss
        Q_expected = self.critic(curr_states, curr_actions)
        critic_loss = F.mse_loss(Q_expected, Q_target)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping
        if self.config.GRADIENT_CLIP != 0:
            clip_grad_norm_(self.critic.parameters(),
                            self.config.GRADIENT_CLIP)
        self.critic_optimizer.step()

        # ----------------------- Update Actor ----------------------- #
        # Compute advantage - actor loss
        _, log_action = self.actor(curr_states)
        advantages = (Q_target.detach() - Q_expected.detach())
        actor_loss = -(log_action * advantages).mean()
        # Minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Gradient clipping
        if self.config.GRADIENT_CLIP != 0:
            clip_grad_norm_(self.critic.parameters(),
                            self.config.GRADIENT_CLIP)
        self.actor_optimizer.step()


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



class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.

        Params
        ======
            self.config.state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.config.state_size = state_size
        self.self.config.action_size = self.config.action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, self.config.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)
        # self.actor_lr_scheduler = StepLR(self.actor_optimizer,
        #                                  step_size=LR_STEP_SIZE,
        #                                  gamma=LR_GAMMA)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size, self.config.action_size, random_seed).to(device)
        self.critic_target = Critic(
            state_size, self.config.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)
        # self.critic_lr_scheduler = StepLR(self.critic_optimizer,
        #                                   step_size=LR_STEP_SIZE,
        #                                   gamma=LR_GAMMA)

        # Noise process
        self.noise = OUNoise(self.config.action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(
            self.config.action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self._step_count = 0

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
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), GRADIENT_CLIP_VALUE)
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
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)
