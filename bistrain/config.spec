[TRAINING]

SEED = integer(default=42)
DEVICE = option("cuda", "cpu", default="cuda")
BATCH_SIZE = integer(0, 100000, default=256)
BUFFER_SIZE = integer( 1000, 10000000000000, default=1000000)
MAX_T = integer(1, 100000, default=200)
N_AGENTS =  integer(1, 1000, default=1)
EPISODES =  integer(1, 100000, default=500)
PRINT_EVERY =  integer(1, 1000, default=100)
SUCCESS_SCORE = float()
UPDATE_EVERY = integer(1, 100000, default=1)
WANDB = boolean(default=False)

[AGENT]
TYPE = option("a2c", "ddpg", "ppo", default="ddpg")
ACTION_SIZE = integer(min=1)
STATE_SIZE = integer(min=1)
GAMMA = float(.1, 1, default=.999)
TAU = float(min=0, default=.05)
OPTIMIZER = option("adam", "adamw", "sgd", default="adam")
SEED = integer(default=42)

  [[ACTOR]]
  HIDDEN_SIZE = int_list()
  GRADIENT_CLIP_VALUE = float(min=0)
  LR = float(min=0)
  WEIGHT_DECAY = float(min=0)
  OPTIMIZER = option("adam", "adamw", "sgd", default="adam")

  [[CRITIC]]
  HIDDEN_SIZE = int_list()
  GRADIENT_CLIP_VALUE = float(min=0, default=10000000000)
  LR = float(min=0, default=0.01)
  WEIGHT_DECAY = float(min=0, default=0.0)
  OPTIMIZER = option("adam", "adamw"  , "sgd", default="adam")

[EXPLORATION]
TYPE = option("gaussian", "ou", "e-greedy", default="gaussian")
SIZE = integer(min=1)
EPS_BETA = float(min=0, default=0.1)
EPS_MIN = float(min=0, default=0.01)
MEAN = float(default=0)
SIGMA = float(min=0, default=0.05)
THETA = float(min=0, default=0.05)