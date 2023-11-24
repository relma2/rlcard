''' Evaluation/playing rlcard.  
'''
import os
import argparse
from random import randrange

import rlcard
from agents.dqn_rule_agent import DQNAgent
from agents.nfsp_rule_agent import NFSPAgent
from rlcard.agents import RandomAgent
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)
from rlcard.envs.gin_rummy import GinRummyEnv
from rlcard.agents.human_agents.gin_rummy_human_agent.gin_rummy_human_agent import HumanAgent
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.agents.human_agents.gin_rummy_human_agent.gui_gin_rummy.game_app import GameApp
from train import available_agents


class StoredAgent:
    def __init__(self, env):
        self.name = ""
        self.version = ""
        self.agent = RandomAgent(env.num_actions)

    def __str__(self) -> str:
        if self.name == "":
            return "invalid_agent"
        if self.name in available_agents:
            return f"{self.name}_{self.version}"
        
        return self.name


available_simple_agents = ["random", "rule"]
available_agents_message = "Potential agents are 'human', '" + "', '".join(available_agents + available_simple_agents) + "'"

def load_model(stored_agent, env, position, device):
    if (stored_agent.name == "human"):
        return HumanAgent(env.num_actions)
    elif (stored_agent.name == "random"):
        return RandomAgent(env.num_actions)
    elif (stored_agent.name == "rule"):
        return GinRummyNoviceRuleAgent()

    model_path = f"cs534/models/{stored_agent.name}/{stored_agent.__str__()}.pth"

    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

# Get agents
def get_agents(env, device):
    agents = [StoredAgent(env), StoredAgent(env)]
    for i in range(2):
        print(available_agents_message)
        print(f"Please select agent {i}: ", end="")
        while True:
            agents[i].name = input().lower()
            print()
            if agents[i].name not in available_agents and agents[i].name not in available_simple_agents and agents[i].name != "human":
                print("No such agent " + agents[i].name)
                print(available_agents_message)
                print("Please try again: ", end="")
            elif i == 0 and agents[i].name == "human":
                print("Selecting a human first causes GUI problems.")
                print(available_agents_message)
                print("Please try again: ", end="")
            else:
                if agents[i].name in available_agents:
                    print("Please enter the amount of training the agent has received.")
                    print("This should be between 100 and 5000: ", end="")
                    agents[i].version = int(input())
                    print()
                break

        agents[i].agent = load_model(agents[i], env, i, device)

    print(f"Selected {agents[0].__str__()} vs {agents[1].__str__()}\n")
    return agents

if __name__ == '__main__':
    print()

    # Prepare testing.
    seed = randrange(10000)

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(seed)

    # Make the environment with seed
    config = {
        'allow_step_back': False,
        'seed': seed,
    }
    env = GinRummyEnv(config)

    # Get players
    print()
    agents = get_agents(env, device)
    env.set_agents([agent.agent for agent in agents])

    if "human" in [agent.name for agent in agents]:
        GameApp(lambda: env)
    else:
        steps = 1000
        payoff = [0,0]
        for i in range(steps):
            _, _payoff = env.run()
            payoff[0] += _payoff[0]
            payoff[1] += _payoff[1]

        payoff[0] /= steps
        payoff[1] /= steps

        for i in range(2):
            print(f"Agent {i} - {agents[i].__str__()} - payoff: {payoff[i]}")
    print()
    




