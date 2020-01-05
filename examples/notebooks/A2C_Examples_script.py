#!/usr/bin/env python
# coding: utf-8

# # A2C Example
#
# This example shows how to use TODO (A2C) Agent to solve an environment.
import os
from bistrain.trainer import Trainer
from bistrain.config.configuration import BisTrainConfiguration
from bistrain.config import CONFIGSPEC_A2C


# Define configuration file or Object
config_file = os.path.join(os.path.dirname(__file__), 'config_a2c.yaml')
config = BisTrainConfiguration(config_file, configspec=CONFIGSPEC_A2C)
print(config)

# confobj = ConfigObj(config_file, configspec=CONFIG_SPEC)
# validator = Validator()
# print(confobj.validate(validator, preserve_errors=True))
# Trainer
# print(confobj)
trainer = Trainer(config)
trainer.run()
