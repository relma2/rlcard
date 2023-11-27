
import os
import argparse
from random import randrange

import torch

import rlcard
from agents.dqn_rule_agent import DQNAgent
from agents.nfsp_rule_agent import NFSPAgent
from agents.dmc_rule_trainer import DMCTrainer
from agents.rule_filter import gin_rummy_rule_filter
from utils.logger import (
    Logger,
    plot_curve
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize
)
from rlcard.rlcard.envs.gin_rummy import GinRummyEnv
from rlcard.rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.rlcard.agents import RandomAgent

if __name__ == '__main__':

    # Prepare testing.
    seed = randrange(10000)

    set_seed(seed)

    config = {
        'allow_step_back': False,
        'seed': seed,
    }
    env = GinRummyEnv(config)

    checkpoint_path = "cs534/models/dmc/checkpoint/"

    agent = DMCTrainer(
            env=env,
            savedir=checkpoint_path,
            rule=gin_rummy_rule_filter
        )
    agent.start()