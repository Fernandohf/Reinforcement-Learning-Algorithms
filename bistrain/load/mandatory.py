"""
Constants expected values for configurations
"""

import numpy as np

MANDATORY = {"AGENT": {"action_size": {"MAX": np.inf, "MIN": 1},
                       "state_size": {"MAX": np.inf, "MIN": 1},
                       "actor_lr": {"MAX": np.inf, "MIN": np.nextafter(0, 1)},
                       "critic_lr": {"MAX": np.inf, "MIN": np.nextafter(0, 1)},
                       "gamma": {"MAX":  np.nextafter(1, 0), "MIN": np.nextafter(0, 1)},
                       "tau": {"MAX": np.nextafter(1, 0), "MIN": np.nextafter(0, 1)},
                       "seed": {"MAX": np.inf, "MIN": -np.inf}},
             "TRAINING": {"agent": ["ddpg", "a2c", "ppo"],
                          "n_agents": {"MAX": np.inf, "MIN": 1},
                          "max_t": {"MAX": np.inf, "MIN": 1},
                          "buffer_size": {"MAX": np.inf, "MIN": 1},
                          "episodes": {"MAX": np.inf, "MIN": 1},
                          "print_every": {"MAX": np.inf, "MIN": 1},
                          "seed": {"MAX": np.inf, "MIN": 1},
                          "update_every": {"MAX": np.inf, "MIN": 1},
                          "batch_size": {"MAX": np.inf, "MIN": 1},
                          "device": ["cuda", "cpu", "cuda:0", "cuda:1"]},
             "EXPLORATION": {"noise": ["gaussian", "ou"]}}
