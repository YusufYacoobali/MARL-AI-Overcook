#from environment import OvercookedEnvironment
#from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.map import BaseMap
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from utils.utils import agent_settings
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import random
import argparse
from collections import namedtuple

import gym


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--dish", type=str, default="SimpleTomato", help="The dish to make")
    parser.add_argument("--num-agents", type=int, required=True, default=2, help="The number of agents wanted")
    parser.add_argument("--grid-size", type=str, default=4, help="The size of the map wanted")
    parser.add_argument("--grid-type", type=str, default="o", help="The type of map to generate")
    parser.add_argument("--eps", type=int, default=2, help="Number of training episodes to run")
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--record", action="store_true", default=False, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    # `pg` = Policy Gradient; `ql` = Q-Learning
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, greedy, pg or ql)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, greedy, pg or ql)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, greedy, pg or ql)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, greedy, pg or ql)")

    return parser.parse_args()

def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
def initialize_agents(arglist):
    real_agents = []

    with open('utils/levels/{}.txt'.format(arglist.level), 'r') as f:
        phase = 1
        recipes = []
        for line in f:
            line = line.strip('\n')
            if line == '':
                phase += 1
            # phase 2: read in recipe list
            elif phase == 2:
                recipes.append(globals()[line]())

            # phase 3: read in agent locations (up to num_agents)
            elif phase == 3:
                if len(real_agents) < arglist.num_agents:
                    loc = line.split(' ')
                    real_agent = RealAgent(
                            arglist=arglist,
                            name='agent-'+str(len(real_agents)+1),
                            id_color=COLORS[len(real_agents)],
                            recipes=recipes)
                    real_agents.append(real_agent)
    return real_agents

def main_loop(arglist):

    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    # Create map for game
    map = BaseMap(file_path='utils/levels/map.txt', arglist=arglist)
    map.start()
    arglist.level = "map"
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    #game = GameVisualize(env)
    real_agents = initialize_agents(arglist=arglist)
    rl_agents = []
    # Change max steps per training episode, the higher the better for training
    max_steps_per_episode = int(arglist.grid_size) * 5

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    for agent in real_agents:
        if agent.is_using_reinforcement_learning:
            rl_agents.append(agent)

    # Training loop for RL agents
    if rl_agents:
        episode_rewards = []
        epsilon_start = 0.9
        epsilon_end = 0.1  
        epsilon_decay = 0.85
        num_episodes = int(arglist.eps)

        for agent in rl_agents:
            agent.in_training = True

        for episode in range(num_episodes):
            # Reset the environment for a new episode
            print("Episode: ", episode)
            episode_reward = 0
            epsilon = epsilon_start * (epsilon_decay ** episode)
            epsilon = max(epsilon, epsilon_end)
            obs = env.reset()

            for step in range(max_steps_per_episode):
                action_dict = {}
                # Action selection for RL agents
                for agent in rl_agents:
                    action = agent.select_action(obs=obs, episode=episode, max_steps=max_steps_per_episode, epsilon=epsilon)
                    action_dict[agent.name] = action
                    print(f"Agent {agent.name} selects action {action}")
                # Take one step in the environment
                obs, reward, done, info = env.step(action_dict=action_dict)
                print(f"Step {step}: Reward = {reward}, Done = {done}")

                # If agents completed a task, they can use the reward
                for agent in rl_agents:
                    agent.refresh_subtasks(world=env.world, reward=(max_steps_per_episode + 1) - step)

                if done or step == max_steps_per_episode - 1:
                    # Collect total rewards for the episode and train for policy gradient
                    for agent in rl_agents:
                        episode_reward += sum(agent.rewards)
                        if agent.model_type == "pg":
                            agent.train()
                    break

            episode_rewards.append(episode_reward)
        print("Training for RL agents has finished")

    # Info bag for saving pkl files
    bag = Bag(arglist=arglist, filename=env.filename)
    bag.set_recipe(recipe_subtasks=env.all_subtasks)

    # Turn RL agents training to be done
    for agent in rl_agents:
        agent.in_training = False
    obs = env.reset()

    while not env.done():
        action_dict = {}
        for agent in real_agents:
            action = agent.select_action(obs=obs, episode=0, max_steps=max_steps_per_episode, epsilon=0)
            action_dict[agent.name] = action

        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        for agent in real_agents:
            agent.refresh_subtasks(world=env.world, reward=0)

        # Saving info
        bag.add_status(cur_time=info['t'], real_agents=real_agents)
    # Saving final information before saving pkl file
    bag.set_collisions(collisions=env.collisions)
    bag.set_termination(termination_info=env.termination_info,
            successful=env.successful)

if __name__ == '__main__':
    arglist = parse_arguments()
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_execute()
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)