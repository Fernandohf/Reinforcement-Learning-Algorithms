import os
import random
from gym.core import Env
import gym
from gym import make
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

# In case of being imported on notebook
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


class Trainer():
    """
    Common trainer for all agents.
    """
    # ALL GYM environments
    def __init__(self, config):
        super().__init__()
        # Load global config
        self.config = config
        # Load environment
        if os.path.isfile(self.config.ENVIRONMENT):
            # Check if the env is a local file
            self.env = UnityEnvironmentWrapper(self.config.ENVIRONMENT,
                                               self.config.N_BRAINS)
        else:
            # Gym environment
            self.env = env = gym.make(self.config.ENVIRONMENT)
            self.env.seed(self.config.SEED)

    def run(self):
        """
        Run the trainer episodes
        TODO
        """

        scores_deque = deque(maxlen=self.config.DEQUE_SCORES_LEN)
        mean_scores_per_agent = []
        scores = []
        max_score = -np.Inf
        for i_episode in range(1, self.config.N_EPISODES + 1):
            env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
            states = env_info.vector_observations               # get the current states
            scores_per_episode = []
            agent.reset()
            for t in range(max_t):
                actions = agent.act(states)
                env_info = env.step(actions)[brain_name]                 # perform actions
                next_states = env_info.vector_observations               # get the next states
                rewards = np.array(env_info.rewards).reshape(-1, 1)      # get the rewards
                dones = np.array(env_info.local_done).reshape(-1, 1)     # see if episode has finished
                agent.step(states, actions, rewards, next_states, dones)
                states = next_states
                scores_per_episode.append(rewards)
                if dones.any():
                    break
            # Scores
            scores_per_episode = np.concatenate(scores_per_episode, axis=1)
            mean_scores_per_agent.append(scores_per_episode.sum(axis=1).reshape(-1, 1))
            score = np.mean(scores_per_episode.sum(axis=1).reshape(-1, 1))
            scores_deque.append(score)
            scores.append(score)
    #         import pdb; pdb.set_trace()

            mean_deque_score = np.mean(scores_deque)
            print('\rEpisode {}/{}\tAverage Score (100): {:.2f}\t Last score: {:.2f}'.format(i_episode,
                                                                                n_episodes,
                                                                                mean_deque_score,
                                                                                score), end="")
            if i_episode % print_every == 0:
                reversed_deque = copy(scores_deque)
                reversed_deque.reverse()
                mean_last_10 = np.mean(list(islice(reversed_deque, 0, print_every)))
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                print('\rEpisode {}/{}\tAverage Score (last {}): {:.2f}\t Last score: {:.2f}'.format(i_episode,
                                                                                    n_episodes,
                                                                                    print_every,
                                                                                    mean_last_10,
                                                                                    score))
            if mean_deque_score >= success_score:
                torch.save(agent.actor_local.state_dict(), 'solved_checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'solved_checkpoint_critic.pth')
                print('\rSolved on Episode {}/{}\tAverage Score (100): {:.2f}'.format(i_episode,
                                                                                    n_episodes,
                                                                                    mean_deque_score))
                break
            return scores


class UnityEnvironmentWrapper(Env):
    """
    A wrapper over unity environments so it behaves equivalent
    to gym environments.
    """
    def __init__(self, unity_env_file, n_brains=1):
        """
        Initialize the name

        Parameters
        ----------
        unity_env_file: str
            Path to unity environment file
        n_brains: int
            Number of brains
        """
        self.env =
        super().__init__()

    def __str__(self):
        return self.env.__str__()


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

