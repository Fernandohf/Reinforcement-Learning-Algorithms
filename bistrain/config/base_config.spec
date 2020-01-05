# Configuration specifications of different algorithms

[GLOBAL]                                                          # *COMMON CONFIGURATION*
SEED = integer(default=42)                                        # Random seed
DEVICE = option("cuda", "cpu", default="cuda")                    # Device used to train
AGENT = option("a2c", "ddpg", "ppo")                              # Agent algorithm being used

[ENVIRONMENT]
ACTION_SIZE = integer(min=1, default=1)                           # Action dimensions
ACTION_SPACE = option("continuous", "discrete")                   # Actions space type
ACTION_RANGE = float_list(default=list(0, 1))                     # Actions values allowed range
STATE_SIZE = integer(min=1)                                       # State dimensions
NAME = string()                                                   # Environment name
N_ENVS = integer(min=1, default=1)                                # Number of parallel environments


