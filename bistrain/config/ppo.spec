# Configuration specifications of differet algorithms

[GLOBAL]                                                          # *COMMON CONFIGURATION*
SEED = integer(default=42)                                        # Random seed
DEVICE = option("cuda", "cpu", default="cuda")                    # Device used to train
AGENT = option("a2c", "ddpg", "ppo")                              # Agent algorithm being used
ACTION_SIZE = integer(min=1, default=1)                           # Action dimensions
ACTION_SPACE = option("continuous", "discrete")                   # Actions space type
ACTION_RANGE = float_list(default=list(0, 1))                     # Actions values allowed range
STATE_SIZE = integer(min=1)                                       # State dimensions
ENVIRONMENT = string()                                            # Environment name


[PPO]                                                             # *PPO AGENT CONFIGURATIONS*
  [[ACTOR]]                                                       # *ACTOR/POLICY CONFIGURATION*
  ARCHITECTURE = option("fc", "lstm", default="fc")               # TODO
  HIDDEN_SIZE = int_list(default=list(256, 128))                  #
  LR = float(min=0, default=0.01)                                 #
  WEIGHT_DECAY = float(min=0, default=0.0)                        #
  OPTIMIZER = option("adam", "adamw", "sgd", default="adam")      #

  [[CRITIC]]                                                      #
  ARCHITECTURE = option("fc", "lstm", default="fc")               #
  HIDDEN_SIZE = int_list(default=list(256, 128))                  #
  LR = float(min=0, default=0.01)                                 #
  WEIGHT_DECAY = float(min=0, default=0.0)                        #
  OPTIMIZER = option("adam", "adamw"  , "sgd", default="adam")    #

  [[TRAINING]]                                                    #
  BATCH_SIZE = integer(0, 100000, default=256)                    #
  BUFFER_SIZE = integer(1000, 10000000000, default=10000000)      #
  GAMMA = float(.1, 1, default=.999)                              #
  TAU = float(min=0, max=1, default=0.05)                         #
  N_STEP_BS = integer(min=1, default=4)                           #
  LAMBDA = float(min=0, max=1, default=0.05)                      #


[EXPLORATION]                                                 #
TYPE = option("gaussian", "ou", "e-greedy", default="gaussian") #
EPS_BETA = float(min=0, default=0.1)                            #
EPS_MIN = float(min=0, default=0.01)                            #
MEAN = float(default=0)                                         #
SIGMA = float(min=0, default=0.05)                              #
THETA = float(min=0, default=0.05)                              #


[TRAINER]
MAX_STEPS = integer(1, 100000, default=200)                       #
N_ENVS =  integer(1, 1000, default=1)                             #
EPISODES =  integer(1, 100000, default=500)                       #
PRINT_EVERY =  integer(1, 1000, default=100)                      #
UPDATE_EVERY = integer(1, 100000, default=1)                      #
WANDB = boolean(default=False)                                    #