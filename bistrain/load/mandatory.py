"""
Constants expected values for configurations
"""

import numpy as np
MANDATORY = {"NUMERIC": {"ACTION_SIZE": {"MAX": np.inf, "MIN": 1},
                         "STATE_SIZE": {"MAX": np.inf, "MIN": 1},
                         "ACTOR_LR": {"MAX": np.inf, "MIN": np.nextafter(0, 1)},
                         "BATCH_SIZE": {"MAX": np.inf, "MIN": 1},
                         "BUFFER_SIZE": {"MAX": np.inf, "MIN": 1},
                         "CRITIC_LR": {"MAX": np.inf, "MIN": np.nextafter(0, 1)},
                         "GAMMA": {"MAX":  np.nextafter(1, 0), "MIN": np.nextafter(0, 1)},
                         "MAX_T": {"MAX": np.inf, "MIN": 1},
                         "N_AGENTS": {"MAX": np.inf, "MIN": 1},
                         "N_EPISODES": {"MAX": np.inf, "MIN": 1},
                         "PRINT_EVERY": {"MAX": np.inf, "MIN": 1},
                         "SEED": {"MAX": np.inf, "MIN": 1},
                         "SUCCESS_SCORE": {"MAX": np.inf, "MIN": 1},
                         "TAU": {"MAX": np.nextafter(1, 0), "MIN": np.nextafter(0, 1)},
                         "UPDATE_EVERY": {"MAX": np.inf, "MIN": 1}},
             "CATEGORICAL": {"AGENT": ["ddpg", "a2c", "ppo"],
                             "DEVICE": ["cuda", "cpu", "cuda:0", "cuda:1"],
                             "NOISE_TYPE": ["gaussian", "ou"]}}
