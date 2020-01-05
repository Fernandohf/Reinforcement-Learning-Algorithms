import os


# Global files
LOCAL_FOLDER = os.path.dirname(__file__)
VALID_FILE = os.path.join(LOCAL_FOLDER, 'test_valid_config.yaml')
VALID_FILE_A2C = os.path.join(LOCAL_FOLDER, 'test_valid_a2c.yaml')
VALID_FILE_DDPG = os.path.join(LOCAL_FOLDER, 'test_valid_ddpg.yaml')
CONFIG_SPEC = os.path.join('bistrain', 'config', 'base_config.spec')
INVALID_FILE_1 = os.path.join(LOCAL_FOLDER, 'test_invalid_config_1.yaml')
INVALID_FILE_2 = os.path.join(LOCAL_FOLDER, 'test_invalid_config_2.yaml')
CONFIG_A2C = os.path.join('bistrain', 'config', 'a2c.spec')
CONFIG_PPO = os.path.join('bistrain', 'config', 'ppo.spec')
CONFIG_DDPG = os.path.join('bistrain', 'config', 'ddpg.spec')
