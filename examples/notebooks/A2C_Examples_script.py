#!/usr/bin/env python
# coding: utf-8

# # A2C Example
#
# This example shows how to use the Asyncronous TODO (A2C) Agent to solve an environment.

from bistrain.trainer import Trainer

# Trainer
trainr = Trainer('config_a2c.yaml')

trainr.run()
