import os


# CONFIG SPECIFICATIONS
CONFIG_SPEC = os.path.join(os.path.dirname(__file__), "config.spec")
CONFIGSPEC_A2C = os.path.join(os.path.dirname(__file__), "a2c.spec")
CONFIGSPEC_DDPG = os.path.join(os.path.dirname(__file__), "ddpg.spec")
CONFIGSPEC_PPO = os.path.join(os.path.dirname(__file__), "ppo.spec")


def get_specfile(agent_type):
    if agent_type == "DDPG":
        return CONFIGSPEC_DDPG
    elif agent_type == "PPO":
        return CONFIGSPEC_PPO
    elif agent_type == "A2C":
        return CONFIGSPEC_A2C
    else:
        return CONFIG_SPEC

