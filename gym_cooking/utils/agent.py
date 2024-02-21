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

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        else:
            self.priors = 'spatial'
        print("MODEL TYPE " ,self.model_type)
        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)
        
        self.good_tasks = []

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

    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs)

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)
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

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
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

    def refresh_subtasks(self, world):
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
        
        if self.is_subtask_complete(world):
            print("------------------------\n-----------------")
            print("TASK COMPLETED AND ADDED TO GOOD TASKS", self.subtask)
            print("------------------------\n-----------------")
            self.good_tasks.append(self.subtask)

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
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









class QLearningAgent:

    """Q-learning Agent object that performs task inference and plans."""
    
    def __init__(self, arglist, name, id_color, recipes, num_actions):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes
        self.num_actions = num_actions
        
        # Q-learning parameters
        self.q_values = np.zeros((num_actions,))
        self.learning_rate = arglist.learning_rate
        self.discount_factor = arglist.discount_factor
        self.epsilon = arglist.epsilon

           # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        else:
            self.priors = 'spatial'
        
        # Navigation planner
        self.planner = E2E_BRTDP(
            alpha=arglist.alpha,
            tau=arglist.tau,
            cap=arglist.cap,
            main_cap=arglist.main_cap
        )
        
        self.good_tasks = []
        
    def __str__(self):
        return color(self.name[-1], self.color)
    
    def select_action(self, obs):
        """Return action for this agent given observations."""
        if np.random.rand() < self.epsilon:
            # Explore: select random action
            self.action = np.random.choice(self.num_actions)
        else:
            # Exploit: select action with highest Q-value
            self.action = np.argmax(self.q_values)
        return self.action
    
    def update_q_values(self, action, reward):
        """Update Q-values based on observed reward."""
        # Update Q-value using Q-learning update rule
        max_q_value_next = np.max(self.q_values)
        td_target = reward + self.discount_factor * max_q_value_next
        td_error = td_target - self.q_values[action]
        self.q_values[action] += self.learning_rate * td_error

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

    # def select_action(self, obs):
    #     """Return best next action for this agent given observations."""
    #     sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
    #     self.location = sim_agent.location
    #     self.holding = sim_agent.holding
    #     self.action = sim_agent.action

    #     if obs.t == 0:
    #         self.setup_subtasks(env=obs)

    #     # Select subtask based on Bayesian Delegation.
    #     self.update_subtasks(env=obs)
    #     self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
    #             agent_name=self.name)
    #     self.plan(copy.copy(obs))
    #     return self.action

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

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
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

    def refresh_subtasks(self, world):
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
        
        if self.is_subtask_complete(world):
            print("------------------------\n-----------------")
            print("TASK COMPLETED AND ADDED TO GOOD TASKS", self.subtask)
            print("------------------------\n-----------------")
            self.good_tasks.append(self.subtask)

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
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





            

# class RealAgentQLearning:
#     """Real Agent object that performs task inference and plans using Q-learning."""

#     def __init__(self, arglist, name, id_color, recipes, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99, exploration_min=0.01):
#         self.arglist = arglist
#         self.name = name
#         self.color = id_color
#         self.recipes = recipes
#         self.state_size = state_size
#         self.action_size = action_size
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate
#         self.exploration_decay = exploration_decay
#         self.exploration_min = exploration_min

#         # Initialize Q-table with zeros
#         self.q_table = np.zeros((state_size, action_size))

#     def select_action(self, state):
#         """Select action using epsilon-greedy policy."""
#         if np.random.rand() < self.exploration_rate:
#             # Exploration: Choose a random action
#             return np.random.choice(self.action_size)
#         else:
#             # Exploitation: Choose action with maximum Q-value
#             return np.argmax(self.q_table[state])

#     def update_q_table(self, state, action, reward, next_state):
#         """Update Q-value based on Q-learning update rule."""
#         # Q-learning update rule
#         target = reward + self.discount_factor * np.max(self.q_table[next_state])
#         self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

#     def decay_exploration(self):
#         """Decay exploration rate over time."""
#         if self.exploration_rate > self.exploration_min:
#             self.exploration_rate *= self.exploration_decay
