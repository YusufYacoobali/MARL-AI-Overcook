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

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)
    
    # def update_output_size(self, new_output_size):
    #     # Update the output size of the last layer
    #     self.fc2 = nn.Linear(64, new_output_size)
    #     #self.fc1 = nn.Linear(new_output_size, 64)

class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        # Define policy network
        # self.policy_network = PolicyNetwork(input_size=..., output_size=...)  # Define input and output size

        # Define optimizer for policy network
        #self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.is_using_reinforcement_learning = False

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        elif self.model_type == "rl":
            self.is_using_reinforcement_learning = True
            self.priors = 'spatial'
        else:
            self.priors = 'spatial'
        print("MODEL TYPE " ,self.model_type)
        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)
        
        # Q-learning parameters
        self.q_values = ()
        self.learning_rate = 0.1  
        self.in_training = True

        self.log_prob = 0
        
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

    def select_action(self, obs, episode):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs, episode=episode)

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)

        if self.is_using_reinforcement_learning:
            print("AGENT TRAINING STATUS ", self.in_training)
            #if rl agent is training, learn from subtasks of delegator, else pick best from learnt tasks
            if self.in_training:
                self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
                        agent_name=self.name)
            else: 
                # for subtask, q_value in self.q_values.items():
                #     print(f"PICKING BEST Subtask: {subtask}, Q-value: {q_value}")
                
                #NEXT PART TO FIXXXXX
                completion_status_list = [status for status in self.task_completion_status.values()]
                # Convert completion_status_list to a float tensor with requires_grad=True
                input_tensor = torch.tensor([completion_status_list], dtype=torch.float32, requires_grad=True)

                print("INPUT Tensor:", input_tensor)
                subtask_probs = self.policy_network(torch.FloatTensor(input_tensor))
                for task, completion_status in self.task_completion_status.items():
                    print("Task:", task, "| Completion Status:", completion_status)
                print("Subtask Probabilities:", subtask_probs)
                subtask_index = torch.argmax(subtask_probs).item()
                print("Selected Subtask Index:", subtask_index)
                self.new_subtask = self.incomplete_subtasks[subtask_index]
                self.new_subtask_agent_names = [self.name]
                print(f"Chosen action: {self.new_subtask}")
                #raise Exception("STOPPP")
        #non rl is original way
        else: 
             self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
                agent_name=self.name)
             
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
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
        self.task_completion_status = {task: 0 for task in self.incomplete_subtasks}
        #set up q values in first episode
        if self.is_using_reinforcement_learning and self.in_training and episode == 0:
            #self.q_values = {subtask: 0.0 for subtask in self.incomplete_subtasks}
            self.all_tasks = sorted(self.incomplete_subtasks,  key=str)
            self.size = len(self.incomplete_subtasks)
            self.policy_network = PolicyNetwork(input_size=self.size, output_size=self.size)  # Define input and output size
            print("POLICY NETWORK MADE")
            # Define optimizer for policy network
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)
            print("optimiser MADE")

        
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

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
        print('{} incomplete subtasks:'.format(
            color(self.name, self.color)),
            ', '.join(str(t) for t in self.incomplete_subtasks))
        
        if self.is_using_reinforcement_learning:
            if self.is_subtask_complete(world) and self.in_training == True:
    
                self.task_completion_status[self.subtask] = 1
                print("PERFORMING PPO UPDATE")
                self.perform_ppo_update(reward=reward)
            elif self.is_subtask_complete(world) and self.in_training == False:
                # # Remove the selected subtask from the Q-table
                # if self.subtask is not None:
                #     del self.q_values[self.subtask]
                #     print("OLD SUBTASK DELETED")
                pass

    def perform_ppo_update(self, reward):
        # Convert completion_status_list to a tensor
        completion_status_list = [status for status in self.task_completion_status.values()]
        input_tensor = torch.tensor([completion_status_list], dtype=torch.float32, requires_grad=True)
        input_tensor = input_tensor.clone().detach()

        for task, completion_status in self.task_completion_status.items():
             print("Task:", task, "| Completion Status:", completion_status)
        print("INPUT Tensor:", input_tensor)
        
        # Calculate new action probabilities
        new_action_probs = self.policy_network(input_tensor)
        task_names = list(self.task_completion_status.keys())
        subtask_index = task_names.index(self.subtask)
        # Calculate new log probability for the current subtask
        new_log_prob = torch.log(new_action_probs[0, subtask_index])
        
        # Calculate ratio of new and old policy probabilities
        with torch.no_grad():  # Ensure that the computation is not tracked for gradient calculation
            ratio = torch.exp(new_log_prob - self.log_prob)
        # Now, ratio is a leaf tensor with requires_grad=True
        ratio.requires_grad = True
        
        # Calculate surrogate loss
        epsilon = 0.2
        surrogate1 = ratio * reward
        # Detach ratio tensor before applying clamp
        clamped_ratio = torch.clamp(ratio.detach(), 1 - epsilon, 1 + epsilon)
        surrogate2 = clamped_ratio * reward
        #surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * reward
        # Clone tensors to avoid inplace operations
        surrogate1_clone = surrogate1.clone().detach()
        surrogate2_clone = surrogate2.clone().detach()

        surrogate1_clone.requires_grad = True
        surrogate2_clone.requires_grad = True

        # Calculate surrogate loss
        surrogate_loss = -torch.min(surrogate1_clone, surrogate2_clone)
        #surrogate_loss = -torch.min(surrogate1, surrogate2)

        print("New Log Probability:", new_log_prob)
        print("Old Log Probability:", self.log_prob)
        self.log_prob = new_log_prob
        
        # Update policy network
        self.optimizer.zero_grad()
        surrogate_loss.backward(retain_graph=True)
        self.optimizer.step()

        return surrogate_loss.item()
        
    # def perform_ppo_update(self, reward):
    #     # Calculate advantage (e.g., using Generalized Advantage Estimation)
    #     if self.is_subtask_complete:
    #         advantage = reward  # If subtask is complete, advantage is equal to the reward
    #     else:
    #         advantage = 0  # If subtask is not complete, advantage is 0 (no advantage)

    #     print("ADVANTYAGE:", advantage)

    #     for task, completion_status in self.task_completion_status.items():
    #         print("Task:", task, "| Completion Status:", completion_status)

    #     completion_status_list = [status for status in self.task_completion_status.values()]
    #     # Convert completion_status_list to a float tensor with requires_grad=True
    #     input_tensor = torch.tensor([completion_status_list], dtype=torch.float32, requires_grad=True)

    #     print("INPUT Tensor:", input_tensor)
    #     #new_action_probs = (input_tensor) 
    #     new_action_probs = self.policy_network(input_tensor)

    #     task_names = list(self.task_completion_status.keys())
    #     # Find the index of self.subtask within task_names
    #     subtask_index = task_names.index(self.subtask)
    #     # Access the completion status using subtask_index
    #     new_log_prob = torch.log(new_action_probs[0, subtask_index])
        # print("New Log Probability:", new_log_prob)
        # print("Old Log Probability:", self.log_prob)
    #     ratio = torch.exp(new_log_prob - self.log_prob)
    #     print("Ratio:", ratio)
    #     self.log_prob = new_log_prob

    #     # Calculate surrogate loss (e.g., using clipped surrogate objective)
    #     epsilon = 0.2  # Hyperparameter for clipped surrogate objective
    #     surrogate1 = ratio * advantage
    #     surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
    #     surrogate_loss = -torch.min(surrogate1.clone(), surrogate2.clone())
    #     print("Surrogate Loss:", surrogate_loss)

    #     # Clone tensors to avoid inplace operations
    #     surrogate1 = surrogate1.clone()
    #     surrogate2 = surrogate2.clone()

    #     # Update policy network
    #     self.optimizer.zero_grad()
    #     surrogate_loss.backward(retain_graph=True)
    #     self.optimizer.step()


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
                    #probs dont add up to 1 error here
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

    # def update_q_values(self, reward):
    #     """Update Q-value for the given subtask based on observed reward."""
    #     # Update Q-value using Q-learning update rule
    #     print("AGENT ", self.name)
    #     print("subtask GIVEN and updating, new q values of ",self.subtask)
    #     self.q_values[self.subtask] += self.learning_rate * (reward - self.q_values[self.subtask])
    #     for subtask, q_value in self.q_values.items():
    #         print(f"Subtask: {subtask}, Q-value: {q_value}")


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