"""
Trainer code
"""
import os
from collections import deque

import gym
import numpy as np
from gym.core import Env

from .noise import OUNoise, GaussianNoise
from .agents import DDPGAgent, A2CAgent
from .config.configuration import BisTrainConfiguration, LocalConfig
from .utils import make_multi_envs
from .config import get_specfile

# In case of being imported on notebook
try:
    get_ipython
    from tqdm import tqdm_notebook as tqdm
    from halo import HaloNotebook as Halo
    N_COLS = 800
except NameError:
    N_COLS = 150
    from tqdm import tqdm
    from halo import Halo as Halo


class Trainer():
    """
    Common trainer for all agents.
    """
    # ALL GYM environments
    def __init__(self, config, env=None, agent=None):
        """
        Initialize the trainer with its configurations.
        If initialized with the BisTrain configuration it creates all
        the auxiliary classes classes, if only with local configuration it
        should receive the created classes.

        Parameters
        ----------
        config: BisTrainConfig or str
            Configuration file with all hyperparameters

        env: object
            Environment Object

        agent: agent object
            Agent object from agents file

        """
        # Halo configuration
        self.spinner = Halo(text="Loading configuration", spinner='dots')
        self.spinner.start()

        # Load global/local config
        if isinstance(config, str):
            c = BisTrainConfiguration(config)
            config = BisTrainConfiguration(config,
                                           get_specfile(c["AGENT"].upper()))
        elif isinstance(config, BisTrainConfiguration):
            # Validate with new configspec
            config.set_configspec(get_specfile(config["AGENT"].upper()))
            config._validate()

        else:
            raise ValueError("Configuration file is invalid!")

        self.global_config = config
        self.config = LocalConfig(self.global_config["TRAINER"])
        self.spinner.succeed("Configuration loaded successfully!")

        self.spinner.start("Loading environment")
        # Load environment
        self.env = self.load_environment(env)
        self.spinner.succeed("Environment loaded successfully!")

        # Load agent
        self.spinner.start("Loading agent")
        self.agent = agent or self.load_agent(self.env)
        self.spinner.succeed("Agent loaded successfully!")

        # Final message
        self.spinner.start("")
        self.spinner.stop_and_persist(symbol="✔".encode('utf-8'), text="Trainer ready!")

    def load_environment(self, env=None):
        """
        Return environment

        Parameters
        ----------
        env:
            Gym compatible environment

        Return
        ------
        env: Env
            Environment object
        """
        env_config = LocalConfig(self.global_config["ENVIRONMENT"])
        if env is None:
            # Load environment(s)
            if os.path.isfile(env_config.NAME):
                # Check if the env is a local file
                env = UnityEnvironmentWrapper(env_config.NAME,
                                              env_config.N_ENVS)
            else:
                if env_config.N_ENVS == 1:
                    # Gym environment
                    env = gym.make(env_config.NAME)
                    env.seed(self.config.SEED)
                else:
                    # Gym environments
                    env = make_multi_envs(env_config.N_ENVS,
                                          env_config.NAME,
                                          self.config.SEED)
        # Add configuration
        env.config = env_config
        return env

    def load_noise(self):
        """
        Load the noise defined in the BisTrainConfig

        Returns
        -------
        noise: Noise
            Noise with all hyperparameters

        """
        config = LocalConfig(self.global_config["EXPLORATION"])
        if config.TYPE == 'ou':
            noise = OUNoise(config)
        elif config.TYPE == 'normal' or config.TYPE == 'gaussian':
            noise = GaussianNoise(config)
        else:
            msg = f"Noise type {config.TYPE} not implemented yet."
            raise NotImplementedError(msg)

        return noise

    def load_agent(self, env):
        """
        Load the agent defined in the BisTrainConfig

        Parameters
        ---------
        env: Environment object
            Environment object to train the agent
        Returns
        -------
        agent: Agent
            Agent with all hyperparameters
        """
        # Agent type
        agent_type = self.global_config["AGENT"].upper()
        agent_config = LocalConfig(self.global_config[agent_type])

        # Noise
        noise = self.load_noise()

        # Agent
        if agent_type == 'DDPG':
            agent = DDPGAgent(agent_config, noise, env)
        elif agent_type == 'A2C':
            agent = A2CAgent(agent_config, noise, env)
        else:
            msg = f"Agent type {agent_type} not implemented yet."
            raise NotImplementedError(msg)

        return agent

    def training_stats(self):
        """
        Display the training stats
        TODO
        """
        pass

    def run(self, save=False):
        """
        Run the trainer through episodes

        Parameters
        ----------
        save: bool
            Wether save the the modelsprogress
        """
        # Saving metrics
        scores_deque = deque(maxlen=self.config.PRINT_EVERY)
        avg_scores = []

        # Keep track of progress
        pbar = tqdm(range(1, self.config.EPISODES + 1), ncols=N_COLS, desc="Training")
        for i_episode in pbar:
            # Reset agent noise (exploration)
            self.agent.reset()

            for i in range(self.config.MAX_STEPS):
                scores, done = self.agent.step(self.env)
                scores_deque.append(scores.mean())
                avg_scores.append(np.mean(scores_deque))
                if done:
                    break

            # Display some progress
            if (i_episode) % self.config.PRINT_EVERY == 0:
                text = '\rEpisode {}/{}\tAverage Scores(Last {}):{:5.2f}'.format(i_episode,
                                                                                 self.config.EPISODES,
                                                                                 self.config.PRINT_EVERY,
                                                                                 np.mean(scores_deque))
                pbar.set_description(text)

            # TODO - verify if the environment was solved!

        # Saves training values
        self.scores = avg_scores
        if save:
            # TODO Add save function to each agent
            # agent.save('checkpoint_final.pth')
            pass


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
        # TODO
        self.env = 0
        super().__init__()

    def __str__(self):
        return self.env.__str__()
