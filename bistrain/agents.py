import torch
import torch.nn.functional as F
import numpy as np

from .base.base_agent import BaseAgent
from .utils.experience import n_step_boostrap, soft_update


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
        super().__init__(config_file, noise)

        # Actor Network
        self.actor = self._set_policy()

        # Critic Network
        self.critic = self._set_val_func()

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
        with torch.no_grad():
            action, _ = self.actor(state)
            action = action.cpu().detach().numpy()
        self.actor.train()

        # Add action noise
        if explore:
            action = self._add_action_noise(action)

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
                                             n_bootstrap=config.N_STEP_BS)
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
        self._clip_gradient(self.critic)

        # Run optimizer
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
        self._clip_gradient(self.actor)

        self.actor.optimizer.step()


class DDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""

    def __init__(self, config_file, noise):
        """
        Initialize an Deep Deterministic Policy Gradient Agent object.

        Parameters
        ----------
        config_file: str, LocalConfig or BisTrainConfiguration
            Path to configuration file or configuration object
        noise: utils.noise object
            Noise object used in the agent
        """
        # Base class
        super().__init__(config_file, noise)

        # Actor Network
        self.actor_local = self._set_policy()
        self.actor_target = self._set_policy(optimizer=False)

        # Critic Network
        self.critic_local = self._set_val_func()
        self.critic_target = self._set_val_func(optimizer=False)

        # Replay buffer
        self.memory = self._set_buffer()

        # Reset current status
        self.reset()

    def reset(self):
        # Learning parameters
        self._step_count = 0
        self._initial_states = None
        self.noise.reset()

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
        # Shortcut
        config = self.config.TRAINING

        # Check if first step
        if self._initial_states is None:
            states = env.reset()
        else:
            states = self._initial_states

        # Get action
        action = self.act(states, add_noise=True)

        # Propagates environment
        next_states, reward, done, _ = env.step(action)

        # Save experience / reward
        self.memory.add(states, action, reward, next_states, done)
        self._step_count += 1

        # Update for next step
        self._initial_states = next_states

        # Learn, if enough samples are available in memory
        if (len(self.memory) > config.BATCH_SIZE and
           self._step_count >= config.UPDATE_EVERY_N_STEPS):
            self._step_count = 0
            # Multiple updates
            for i in range(config.UPDATE_N_TIMES):
                experiences = self.memory.sample()
                self._learn(experiences, config.GAMMA)

        scores = reward[:, 0].mean()
        done = done[:, -1].any()

        return scores, done

    def act(self, state, explore=True):
        """
        Returns actions for given state as per current policy.
        """
        state = (torch.from_numpy(state).float()
                 .to(self.config.DEVICE))
        # Action
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().detach().numpy()
        self.actor_local.train()

        # Exploration
        if explore:
            action = self._add_action_noise(action)

        return action

    def _learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch
        of experience tuples.

        Q_targets = r + γ * critic_target(next_state,
                                          actor_target(next_state))

        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
            experiences: Tuple[torch.Tensor]
                Tuple of (s, a, r, s', done) tensors
            gamma: float
                Discount factor
        """
        # Shortcut
        config = self.config.TRAINING

        # Unpack
        states, actions, rewards, next_states, dones = experiences
        # -------------------- Update Critic -------------------- #
        # Predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_local.optimizer.zero_grad()
        critic_loss.backward()

        # Clip gradients
        self._clip_gradient(self.critic_local)

        self.critic_local.optimizer.step()

        # -------------------- Update Actor -------------------- #

        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_local.optimizer.zero_grad()
        actor_loss.backward()

        # Clip values
        self._clip_gradient(self.actor_local)

        self.actor_local.optimizer.step()

        # ------------- Update Target Networks ------------- #
        soft_update(self.critic_local, self.critic_target,
                    config.TAU)
        soft_update(self.actor_local, self.actor_target,
                    config.TAU)
