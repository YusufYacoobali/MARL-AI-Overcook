# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard
from utils.utils import agent_settings

# Reinforcement learning modules
from utils.network import PolicyNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']

class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        # Q-learning parameters
        self.q_values = {}
        self.learning_rate = 0.1  
        self.gamma = 0.8
        self.in_training = False
        self.is_using_reinforcement_learning = False

        # Policy Gradient parameters
        self.states = []
        self.actions = []
        self.rewards = []
        self.task_length = 0

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        elif self.model_type == "ql" or self.model_type == "pg":
            self.is_using_reinforcement_learning = True
            self.priors = 'spatial'
        else:
            self.priors = 'spatial'

        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)
        
    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def select_action(self, obs, episode, max_steps, epsilon):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs, episode=episode)
        self.update_subtasks(env=obs)
        self.task_length += 1

        print("\nAgent Is Training: ", self.in_training)

        if self.is_using_reinforcement_learning and self.task_length <= max_steps:
            self.new_subtask_agent_names = [self.name]
            # If agent is using RL and needs inference, then use appropriate model type solution
            if not self.in_training:
                if self.model_type == "ql":
                    state = tuple(self.task_completion_status.values())
                    actions = tuple(self.task_completion_status.keys())
                    # Find the action with the highest Q-value for the current state.
                    self.new_subtask = max(actions, key=lambda act: self.q_values.get((state, act), 0), default=None)

                elif self.model_type == "pg":
                    completion_status_list = [status for status in self.task_completion_status.values()]
                    state_tensor = torch.tensor(completion_status_list, dtype=torch.float32)
                    # Gradient calculation not needed for inference
                    with torch.no_grad():  
                        logits = self.policy_network(state_tensor)
                        action_probs = F.softmax(logits, dim=-1).numpy()  
                        
                    # Select the action with the highest probability
                    best_action_index = np.argmax(action_probs)
                    task = list(self.task_completion_status.keys())[best_action_index]
                    if task in self.incomplete_subtasks:
                        self.new_subtask = task
                    else:
                        # If new task is already complete, find next task with next highest probability
                        for i in range(best_action_index + 1, len(action_probs)):
                            next_task = list(self.task_completion_status.keys())[i]
                            if next_task in self.incomplete_subtasks:
                                self.new_subtask = next_task
                                break
            # Epsilon greedy during training, uses the same code for picking the best actions as above
            else:
                if np.random.rand() < epsilon:
                    # Choose a random action
                    self.new_subtask = np.random.choice(self.incomplete_subtasks)
                    
                elif self.model_type == "ql": 
                    state = tuple(status for status in self.task_completion_status.values())
                    actions = tuple(status for status in self.task_completion_status.keys())
                    # Find the action with the highest Q-value for the current state.
                    self.new_subtask = max(actions, key=lambda act: self.q_values.get((state, act), 0), default=None)

                elif self.model_type == "pg": 
                    completion_status_list = [status for status in self.task_completion_status.values()]
                    state_tensor = torch.tensor(completion_status_list, dtype=torch.float32)
                    # Gradient calculation not needed for inference
                    with torch.no_grad():  
                        logits = self.policy_network(state_tensor)
                        action_probs = F.softmax(logits, dim=-1).numpy()  
                        
                    # Select the action with the highest probability
                    best_action_index = np.argmax(action_probs)
                    task = list(self.task_completion_status.keys())[best_action_index]
                    if task in self.incomplete_subtasks:
                        self.new_subtask = task
                    else:
                        # If new task is already complete, find next task with next highest probability
                        for i in range(best_action_index + 1, len(action_probs)):
                            next_task = list(self.task_completion_status.keys())[i]
                            if next_task in self.incomplete_subtasks:
                                self.new_subtask = next_task
                                break
        # Use original delegator if RL method is not chosen.
        elif not self.is_using_reinforcement_learning:
            self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(agent_name=self.name)
        # If an RL agent has had the same task for a while and is unable to do it, it chooses a random task
        else:
            self.new_subtask_agent_names = [self.name]
            self.new_subtask = np.random.choice(self.incomplete_subtasks)
            self.task_length = 0

        print(f"Chosen action: {self.new_subtask}")
        self.plan(copy.copy(obs))
        return self.action

    def get_subtasks(self, world):
        """Return different subtask permutations for recipes."""
        self.sw = STRIPSWorld(world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]

        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        return all_subtasks

    def setup_subtasks(self, env, episode):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        # Lists are used for collecting experiences in episodes for pg
        self.states = []
        self.actions = []
        self.rewards = []
        self.task_length = 0
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
        self.task_completion_status = {task: 0 for task in self.incomplete_subtasks}
        # Set up important variables in the first training episode
        if self.is_using_reinforcement_learning and self.in_training and episode == 0:
            if self.model_type == "pg":
                size = len(self.incomplete_subtasks)
                # Create the policy network along with its optimizer using number of subtasks as its size
                self.policy_network = PolicyNetwork(input_size=size, output_size=size)  
                self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        self.delegator = BayesianDelegator(
                agent_name=self.name,
                all_agent_names=env.get_agent_names(),
                model_type=self.model_type,
                planner=self.planner,
                none_action_prob=self.none_action_prob)

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world, reward):
        """Refresh subtasks---relevant for Bayesian Delegation."""
        # Check whether subtask is complete.
        self.subtask_complete = False
        if self.subtask is None or len(self.subtask_agent_names) == 0:
            print("{} has no subtask".format(color(self.name, self.color)))
            return
        self.subtask_complete = self.is_subtask_complete(world)
        print("{} done with {} according to planner: {}\nplanner has subtask {} with subtask object {}".format(
            color(self.name, self.color),
            self.subtask, self.is_subtask_complete(world),
            self.planner.subtask, self.planner.goal_obj))
        
        # Collect data for RL training or train it directly
        if self.is_using_reinforcement_learning and self.subtask_complete and self.in_training:
            if self.model_type == "ql":
                self.update_q_values(reward)
            elif self.model_type == "pg":
                self.collect_experience()

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            self.rewards.append(reward)
            self.task_completion_status[self.subtask] = 1            
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
                self.task_length = 0
        print('{} incomplete subtasks:'.format(
            color(self.name, self.color)),
            ', '.join(str(t) for t in self.incomplete_subtasks))

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if ((self.subtask is not None and self.subtask not in self.incomplete_subtasks)
                or (self.delegator.should_reset_priors(obs=copy.copy(env),
                            incomplete_subtasks=self.incomplete_subtasks))):
            self.reset_subtasks()
            self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
            else:
                self.delegator.bayes_update(
                        obs_tm1=copy.copy(env.obs_tm1),
                        actions_tm1=env.agent_actions,
                        beta=self.beta)

    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        print('right before planning, {} had old subtask {}, new subtask {}, subtask complete {}'.format(self.name, self.subtask, self.new_subtask, self.subtask_complete))

        # Check whether this subtask is done.
        if self.new_subtask is not None:
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.
        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0-self.none_action_prob)/(len(actions)-1))
            total_prob = sum(probs)
            if total_prob != 1.0:
                probs = [prob / total_prob for prob in probs]
                probs[-1] += 1.0 - sum(probs)
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            if self.model_type == 'greedy' or initializing_priors:
                other_agent_planners = {}
            else:
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                backup_subtask = self.new_subtask if self.new_subtask is not None else self.subtask
                other_agent_planners = self.delegator.get_other_agent_planners(
                        obs=copy.copy(env), backup_subtask=backup_subtask)

            print("[ {} Planning ] Task: {}, Task Agents: {}".format(
                self.name, self.new_subtask, self.new_subtask_agent_names))

            action = self.planner.get_next_action(
                    env=env, subtask=self.new_subtask,
                    subtask_agent_names=self.new_subtask_agent_names,
                    other_agent_planners=other_agent_planners)

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0]
            else:
                self.action = action[self.new_subtask_agent_names.index(self.name)] if self.planner.is_joint else action

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_subtask_agent_names = []

        print('{} proposed action: {}\n'.format(self.name, self.action))

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask=self.new_subtask)
        self.subtask_action_object = nav_utils.get_subtask_action_obj(subtask=self.new_subtask)

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
        if isinstance(self.new_subtask, Deliver):
            self.cur_obj_count = len(list(
                filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)),
                env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                    len(list(filter(lambda o: o in
                set(env.world.get_all_object_locs(obj=self.subtask_action_object)),
                w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        # Otherwise, for other subtasks, check based on # of objects.
        else:
            # Current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count

    def update_q_values(self, reward):
        """Update Q-value for the given subtask based on observed reward."""
        state = tuple([status for status in self.task_completion_status.values()])
        old_q_val = self.q_values.get((state, self.subtask), 0)
        # Create the next game state
        next_task_completion_status = dict(self.task_completion_status)
        next_task_completion_status[self.subtask] = 1
        next_state = tuple([status for status in next_task_completion_status.values()])
        # Get the maximum Q-value for the next state
        next_max_q_val = max(self.q_values.get((next_state, action), 0) for action in self.task_completion_status.keys())

        # Update Q-value using the Q-learning update rule
        self.q_values[(state, self.subtask)] = (1 - self.learning_rate) * old_q_val + self.learning_rate * (reward + self.gamma * next_max_q_val)

    def collect_experience(self):
        """Collect experience from the current completed subtask to be used in training later on."""
        state = [status for status in self.task_completion_status.values()]
        self.states.append(state)
        self.actions.append(self.subtask)

    def train(self):
        """
        Train the agent using collected experiences. Experiences are turned into tensors so the policy can use them
        The losses are computed and the neural network's weights and bias are then adjusted accordingly
        """
        # Retrieve training data 
        states, actions, rewards = self.states, self.actions, self.rewards
        # Convert to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        action_indices = [list(self.task_completion_status.keys()).index(action) for action in actions]
        actions = torch.tensor(action_indices, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute loss and optimize network
        try:
            loss = self.compute_loss(states, actions, rewards)
            # Clears previous gradients for fresh new data
            self.optimizer.zero_grad()
            # Backpropagation, it calculates new gradients
            loss.backward()
            # Uses gradients to update the neural network's weights to get closer to optimal policy
            self.optimizer.step()
        except RuntimeError as e:
            print("There was no good data to update on")
        print("Policy Network updated.")

    def compute_loss(self, states, actions, rewards):
        """
        Compute the loss for training the policy network. 
        Policy gives probabilities which are adjusted according to actions and rewards at those states
        """
        # Retrieve predicted values from neural network to compare to actual rewards
        logits = self.policy_network(states)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_for_actions = log_probs.gather(1, actions.unsqueeze(1))
        # Minimizing a negative loss is the same as maximizing the reward function
        policy_loss = -(log_probs_for_actions * rewards).mean()
        return policy_loss

class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        print("{} currently at {}, action {}, holding {}".format(
                color(self.name, self.color),
                self.location,
                self.action,
                self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location