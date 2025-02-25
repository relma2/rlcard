import os
import argparse
from random import randrange

import torch

from agents.dqn_rule_agent import DQNAgent
from agents.nfsp_rule_agent import NFSPAgent
from agents.cfr_rule_agent import CFRAgent
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
from rlcard.envs.gin_rummy import GinRummyEnv
from rlcard.models.gin_rummy_rule_models import GinRummyNoviceRuleAgent
from rlcard.agents import RandomAgent

available_agents = ["dqn", "nfsp", "cfr"]
available_agents_message = "Potential agents are '" + "', '".join(available_agents) + "'"

def get_agent_name(args):
    if (args.agent != ""):
        if args.agent in available_agents:
            return args.agent
        else:
            raise Exception("No such agent.")

    print("What would you like to train?")
    agent = ""
    print(available_agents_message)
    print("Please select agent: ", end="")
    while True:
        agent = input().lower()
        print()
        if agent not in available_agents:
            print("No such agent " + agent)
            print(available_agents_message)
            print("Please try again: ", end="")
        else:
            break

    return agent

def configure_agent_training(agent_name, env, args):
    start_episodes = 0
    num_episodes = 0
    if agent_name == "dqn":
        checkpoint_path = "cs534/models/dqn/checkpoint/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if os.path.exists(checkpoint_path + "checkpoint_dqn.pt"):
            agent = DQNAgent.from_checkpoint(checkpoint=torch.load(checkpoint_path + "checkpoint_dqn.pt"))
            agent.rule = gin_rummy_rule_filter
            with open(checkpoint_path + "checkpoint.txt") as f:
                start_episodes = int(next(f))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64,64],
                device=device,
                save_path=checkpoint_path,
                rule=gin_rummy_rule_filter
        )
    elif agent_name == "nfsp":
        checkpoint_path = "cs534/models/nfsp/checkpoint/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if os.path.exists(checkpoint_path + "checkpoint_nfsp.pt"):
            agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(checkpoint_path + "checkpoint_nfsp.pt"))
            agent.rule = gin_rummy_rule_filter
            with open(checkpoint_path + "checkpoint.txt") as f:
                start_episodes = int(next(f))
        else:
            agent = NFSPAgent(
                    num_actions=env.num_actions,
                    state_shape=env.state_shape[0],
                    hidden_layers_sizes=[64,64],
                    q_mlp_layers=[64,64],
                    device=device,
                    save_path=checkpoint_path,
                    rule=gin_rummy_rule_filter
                )
    elif agent_name == "cfr":
        env.allow_step_back = True
        env.game.allow_step_back = True
        checkpoint_path = "cs534/models/cfr/checkpoint/"
        agent = CFRAgent(env, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        if os.path.exists(checkpoint_path + "checkpoint_cfr.pt"):
            CFRAgent.load()
            agent.rule = gin_rummy_rule_filter
            with open(checkpoint_path + "checkpoint.txt") as f:
                start_episodes = int(next(f))
        
    if (args.episodes > 0):
        num_episodes = args.episodes
    else:
        if start_episodes != 0:
            print(f"This model has already been trained for {start_episodes} episodes.")
        print("How many more episodes should this model train for?")
        print("This should be be a number divisible by 100: ", end="")
        num_episodes = int(input())

    return (agent, start_episodes, num_episodes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("RLCard Training")
    parser.add_argument(
        '--agent',
        type=str,
        default='',
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=0,
    )
    args = parser.parse_args()

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
    rand_env = GinRummyEnv(config)

    # Get agent
    agent_name = get_agent_name(args)

    # Get actual agent
    agent, start_episodes, num_episodes = configure_agent_training(agent_name, env, args)
    opponent = GinRummyNoviceRuleAgent()
    env.set_agents([agent, opponent])

    random_agent = RandomAgent(rand_env.num_actions)
    rand_env.set_agents([agent, random_agent])

    # Train
    model_dir = f"cs534/models/{agent_name}/"
    checkpoint_path = os.path.join(model_dir, "checkpoint/")
    checkpoint_txt_path = os.path.join(checkpoint_path, "checkpoint.txt")
    eval_interval = 2000
    if agent_name == 'cfr':
        eval_interval = 2
    save_interval = eval_interval * 5
    with Logger(model_dir + "log/") as logger:
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

        for episode in range(start_episodes + 1, num_episodes + start_episodes + 1):
            if agent_name == 'cfr':
                print('\rIteration {}'.format(episode), end='')
                agent.train()
            else:
                if agent_name == 'nfsp':
                    agent.sample_episode_policy()

                # Generate data from the environment
                trajectories, payoffs = env.run(is_training=True)

                # Reorganaize the data to be state, action, reward, next_state, done
                trajectories = reorganize(trajectories, payoffs)

                # Feed transitions into agent memory, and train the agent
                # Here, we assume that DQN always plays the first position
                for ts in trajectories[0]:
                    agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % eval_interval == 0 or episode == 1:
                print("\n*****************************************************")
                print(f"Running episode {episode} evaluation tournament.")
                logger.store_performance(
                    episode,
                    tournament(
                        env,
                        5000,
                    )[0],
                    tournament(
                        rand_env,
                        5000,
                    )[0]
                )
                print(f"\nFinished episode {episode} evaluation tournament.")
                print("*****************************************************\n")

                if episode % save_interval == 0:
                    logger.flush_performance()
                    save_path = os.path.join(model_dir, f"{agent_name}_{episode}.pth")
                    torch.save(agent, save_path)
                    agent.save_checkpoint(checkpoint_path)
                    f = open(checkpoint_txt_path, "w")
                    f.write(str(episode))
                    f.close()

                    # Plot the learning curve
                    plot_curve(csv_path, fig_path, agent_name)


