import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from .base.base_agent import BaseAgent
from .utils.experience import n_step_boostrap, soft_updates


class A2CAgent(BaseAgent):
    """
    Advantage Actor Critic (A2C) Agent
    """

    def __init__(self, config_file, noise):
        """
        Initialize an Advantage Actor Critic (A2C) Agent object.

        Parameters
        ----------
        config_file: str, LocalConfig or BisTrainConfiguration
            Path to configuration file or configuration object
        noise: utils.noise
            Noise object used in the agent
        """
        # Base class
        super().__init__(config_file)

        # Actor Network
        self.actor = self._set_policy()

        # Critic Network
        self.critic = self._set_val_func()

        # Noise process
        self.noise = noise

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
        # Episode parameter
        self._initial_states = None

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
            Rewardsactor_configach parallel environment
        """
        # To facilitate access
        config = self.config.TRAINING

        # First step
        if self._initial_states is None:
            self._initial_states = envs.reset()
            self._gamma = config.GAMMA
        # Unroll trajectories of parallel envs
        s, a, r, sp, dones = n_step_boostrap(envs, self,
                                             self._initial_states,
                                             n_step=config.N_STEP_BS)
        self._learn(s, a, r, sp, dones, self._gamma)
        # Start from the next state
        self._initial_states = sp[:, 0, :]
        # Collect scores from all parallel envs and if any has finished
        scores = r[:, 0]
        done = dones[:, -1].any()
        # Update initial gamma
        self._gamma *= config.GAMMA

        return scores, done

    def _learn(self, states, actions, rewards, next_states, dones, gamma):
        """
        Update policy and value parameters using given batch of expericences.

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
        self.critic.optimizer.zero_grad()
        critic_loss.backward()

        # Gradient clipping
        if self.config.TRAINING.GRADIENT_CLIP != 0:
            clip_grad_norm_(self.critic.parameters(),
                            self.config.TRAINING.GRADIENT_CLIP)
        self.critic.optimizer.step()

        # ----------------------- Update Actor ----------------------- #
        # Compute advantage - actor loss
        _, log_action = self.actor(curr_states)
        advantages = (Q_target.detach() - Q_expected.detach())
        actor_loss = -(log_action * advantages).mean()
        # Minimize loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()

        # Gradient clipping
        if self.config.TRAINING.GRADIENT_CLIP != 0:
            clip_grad_norm_(self.critic.parameters(),
                            self.config.TRAINING.GRADIENT_CLIP)
        self.actor.optimizer.step()


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, config_file, noise, replay_buffer):
        """
        Initialize an Advantage Actor Critic (A2C) Agent object.

        Parameters
        ----------
        config_file: str, LocalConfig or BisTrainConfiguration
            Path to configuration file or configuration object
        noise: utils.noise
            Noise object used in the agent
        replay_buffer: utils.replay
            Buffer that stores the experiences
        """
        # Base class
        super().__init__(config_file)

        # Actor Network
        self.actor_local = self._set_policy()
        self.actor_target = self._set_policy(optimizer=False)

        # Critic Network
        self.critic_local = self._set_val_func()
        self.critic_target = self._set_val_func()

        # Noise process
        self.noise = noise

        # Replay buffer
        self.memory = replay_buffer

        # Reset current status
        self.reset()

    def step(self, env):
        """
        Records experiences

        Parameters
        ----------
        env: Gym.Environment
            Open ai compatible GYM environment

        Returns
        -------
        done: bool
            Return wether the episode is done or not
        scores: array
            Rewards of each parallel environment
        """
        # Propagates envrironment
        env.step()
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
        self._step_count = 0
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
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), GRADIENT_CLIP_VALUE)
        self.critic.optimizer.step()
        # self.critic_lr_scheduler.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), GRADIENT_CLIP_VALUE)
        self.actor.optimizer.step()
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
