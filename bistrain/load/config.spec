keyword = int_list(max=6)

[DEFAULT]
SEED = integer(default=42)
DEVICE= option("cuda", "cpu", default="cuda")

[TRAINING]

BATCH_SIZE = integer(0, 1e5, default=256)
BUFFER_SIZE = integer(1e3, 1e12, default=1e7)
MAX_T = integer(1, 1e5)
N_AGENTS = int(1, 1e3)
EPISODES = int(1, 1e5)
PRINT_EVERY = int(1, 1e3, default=100)
SUCCESS_SCORE = float()
UPDATE_EVERY = integer(1, 1e5, default=1)
WANDB = boolean(default=False)

[EXPLORATION]
TYPE = option("gaussian", "ou", "e-greedy", default="gaussian")
EPS_BETA = float(min=0, default=0.1)
EPS_MIN = float(min=0, default=0.01)
MEAN = float(default=0)
SIGMA = float(min=0, default=0.05)
THETA = float(min=0, default=0.05)

[AGENT]
TYPE = options("a2c", "ddpg", "ppo", default="ddpg")
ACTION_SIZE = integer(min=1)
STATE_SIZE = integer(min=1)
GAMMA = float(.1, 1, default=.999)
TAU = float(min=0, default=.05)
OPTIMIZER = option("adam", "adamw", "sgd", default="adam")

  [[ACTOR]]
  GRADIENT_CLIP_VALUE = float(min=0)
  LR = float(min=0)
  WEIGHT_DECAY = float(min=0)
  OPTIMIZER = option("adam", "adamw", "sgd", default="adam")

  [[CRITIC]]
  GRADIENT_CLIP_VALUE = float(min=0, default=1e32)
  LR = float(min=0, default=.01)
  WEIGHT_DECAY = float(min=0, default=0.)
  OPTIMIZER = option("adam", "adamw", "sgd", default="adam")
