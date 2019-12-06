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

TODO
EPS_BETA= 0.1
EPS_MIN= 0.01
MEAN= 0.0
SIGMA= 0.4
THETA= 0.01

[AGENT]
TYPE= a2c
ACTION_SIZE= 2
STATE_SIZE= 24
GAMMA= 0.99
TAU= 0.005

  [[ACTOR]]
  GRADIENT_CLIP_VALUE= 5
  LR= 0.001
  WEIGHT_DECAY= 0.0
  OPTIMIZER= adam

  [[CRITIC]]
  GRADIENT_CLIP_VALUE= 1
  LR= 0.001
  WEIGHT_DECAY= 1.0e-05
  OPTIMIZER= adam
