"""
Utilities functions for bootstraping experiences
"""
import gym
import numpy as np
from gym.vector import SyncVectorEnv


def make_multi_envs(n_envs, env_name, seed):
    """
    Create multiple function GYM environments

    Parameters
    ----------
    n_env: int
        Number of environments
    env_name: string
        Name of the GYM environment
    seed: int
        Seed for the first environment

    Return
    ------
    mp_envs: SubprocVecEnv
        Parallel environments
    """

    def make_env(env_name, seed):
        """
        Auxiliary function used to create the GYM environment
        with the given seed

        Parameters
        ----------
        env_name: string
            Name of the GYM environment
        seed: int
            Initial seed for the first environment

        Return
        ------
        _init: func
            function that creates env
        """

        def _init():
            env = gym.make(env_name)
            env.seed(seed)
            return env
        return _init

    # Create envs
    env_fns = [make_env(env_name, seed + i) for i in range(n_envs)]

    # Multiprocessing Environments
    mp_envs = SyncVectorEnv(env_fns)

    return mp_envs


def n_step_boostrap(envs, agent, previous_states, n_bootstrap=5):
    """
    Perform n_step bootstrap on the list of parallel environments.

    Parameters:
    -----------
    envs: SyncVectorEnv
        List of environments running on parallel
    agent: Agent with act function
        Policy being used to get the S.A.R.S'. sequences
    previous_states: tuple
        Previous states of all environments
    n_bootstrap: int
        Number of steps used for bootstrapping

    Returns:
    -------
    trajectories: tuple
        Bootstrapped experiences for each parallel environment

        With the shape:
        (n_envs, n_bootstrap, states, actions, rewards, next_states)
    """

    # Initialize returning lists
    state_list = []
    reward_list = []
    action_list = []
    states_next_list = []
    dones_list = []

    for t in range(n_bootstrap):

        # Actions from states for each env
        actions_env = agent.act(previous_states, explore=True)

        # Advance the environment
        states_next_env, rewards_env, done_envs, _ = envs.step(actions_env)

        # Store the result
        state_list.append(previous_states)
        reward_list.append(rewards_env.reshape(-1, 1))
        action_list.append(actions_env)
        states_next_list.append(states_next_env)
        dones_list.append(done_envs.reshape(-1, 1))

        previous_states = states_next_env

        # Stop if any of the trajectories is done, ensures ret. lists
        if done_envs.any():
            break

    # Return states, actions, rewards, states_next
    trajectories = (np.stack(state_list, axis=1),
                    np.stack(action_list, axis=1),
                    np.stack(reward_list, axis=1),
                    np.stack(states_next_list, axis=1),
                    np.stack(dones_list, axis=1))
    return trajectories


def soft_update(self, local_model, target_model, tau):
    """
    Soft update model parameters, perform changes in-place.
    θ_target = τ * θ_local + (1 - τ) * θ_target

    Parameters
    ----------
    local_model: torch.Module
        PyTorch model (weights will be copied from)
    target_model: torch.Module
        PyTorch model (weights will be copied to)
    tau: float
        Interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(),
                                         local_model.parameters()):
        target_param.data.copy_(tau * local_param.data +
                                (1.0 - tau) * target_param.data)
