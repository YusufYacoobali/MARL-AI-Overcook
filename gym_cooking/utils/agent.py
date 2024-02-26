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

         # DQN parameters
        self.q_network = None  # Initialize Q-network
        self.target_network = None  # Initialize target network
        self.batch_size = 1  # Mini-batch size for training
        self.replay_buffer = []  # Experience replay buffer
        self.replay_buffer_capacity = 10000  # Capacity of the replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial epsilon for epsilon-greedy exploration
        self.epsilon_min = 0.01  # Minimum epsilon
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.update_target_network_freq = 1000  # Frequency to update the target network
        
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
                for subtask, q_value in self.q_values.items():
                    print(f"PICKING BEST Subtask: {subtask}, Q-value: {q_value}")

                q_values = self.q_network.predict(np.zeros((1, 1)))  # Placeholder input
                self.new_subtask = self.incomplete_subtasks[np.argmax(q_values)]
                self.new_subtask_agent_names = [self.name]
                print(f"NEW Chosen action: {self.new_subtask}")

                # max_q_value = float('-inf')
                # self.new_subtask = None
                # for subtask, q_value in self.q_values.items():
                #     if q_value > max_q_value:
                #         max_q_value = q_value
                #         self.new_subtask = subtask
                # print(f"Chosen action: {self.new_subtask}")
                # self.new_subtask_agent_names = [self.name]
        #non rl is original way
        else: 
             self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
                agent_name=self.name)
             
        # Decay epsilon for epsilon-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
             
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
        #set up q values in first episode
        if self.is_using_reinforcement_learning and self.in_training and episode == 0:
            self.q_values = {subtask: 0.0 for subtask in self.incomplete_subtasks}

        # print("Q-values:")
        # for i, value in enumerate(self.q_values):
        #     print(f"Action {i}: {value}")

        for subtask, q_value in self.q_values.items():
                    print(f"Q VALUES Subtask: {subtask}, Q-value: {q_value}")

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
        
        # if self.is_using_reinforcement_learning:
        #     if self.is_subtask_complete(world) and self.in_training == True:
        #         self.update_q_values(reward)
        #     elif self.is_subtask_complete(world) and self.in_training == False:
        #         # Remove the selected subtask from the Q-table
        #         if self.subtask is not None:
        #             del self.q_values[self.subtask]
        #             print("OLD SUBTASK DELETED")

          # Update Q-values using DQN
        if self.is_using_reinforcement_learning:
            if self.subtask_complete and self.in_training:
                print("UPDATING Q-values:")
                for subtask, q_value in self.q_values.items():
                    print(f"Q VALUES Subtask: {subtask}, Q-value: {q_value}")

                self.update_replay_buffer(reward)
                self.train_dqn()

                print("NEW UPDATED Q-values:")
                for subtask, q_value in self.q_values.items():
                    print(f"Q VALUES Subtask: {subtask}, Q-value: {q_value}")
            elif self.subtask_complete and self.in_training == False:
                #delete old subtask?
                pass


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

    def update_q_values(self, reward):
        """Update Q-value for the given subtask based on observed reward."""
        # Update Q-value using Q-learning update rule
        print("AGENT ", self.name)
        print("subtask GIVEN and updating, new q values of ",self.subtask)
        self.q_values[self.subtask] += self.learning_rate * (reward - self.q_values[self.subtask])
        for subtask, q_value in self.q_values.items():
            print(f"Subtask: {subtask}, Q-value: {q_value}")

    def update_replay_buffer(self, reward):
        # Store experience in replay buffer
        experience = {
            #'state': np.zeros((1, 1)),  # Placeholder state
            'action': self.subtask,
            'reward': reward,
            'next_subtask': self.new_subtask
        }
        self.replay_buffer.append(experience)

        # Limit replay buffer size
        if len(self.replay_buffer) > self.replay_buffer_capacity:
            self.replay_buffer.pop(0)

    def train_dqn(self):
        # Sample mini-batch from replay buffer
        mini_batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)

        # Extract actions, rewards, and next subtasks from mini-batch
        actions = np.array([experience['action'] for experience in mini_batch])
        rewards = np.array([experience['reward'] for experience in mini_batch])
        next_subtasks = np.array([experience['next_subtask'] for experience in mini_batch])

        # Compute target Q-values using target network
        target_q_values = rewards + self.gamma * np.amax(self.predict_target_q_values(next_subtasks), axis=1)

        # Update Q-values by minimizing loss
        for subtask, target_q_value in zip(actions, target_q_values):
            self.fit_q_values(subtask, target_q_value)

        # # Periodically update target network
        # if episode % self.update_target_network_freq == 0:
        #     self.update_target_network()

    def predict_q_values(self, subtask):
        """Predict Q-values for a given subtask using the Q-network."""
        return self.q_values[subtask]  # Assuming q_values is a dictionary mapping subtasks to Q-values

    def predict_target_q_values(self, next_subtasks):
        """Predict target Q-values for given next subtasks using the target network."""
        return np.array([self.predict_q_values(next_subtask) for next_subtask in next_subtasks])

    def fit_q_values(self, subtask, target_q_value):
        """Update Q-value for a given subtask based on observed target Q-value."""
        self.q_values[subtask] += self.learning_rate * (target_q_value - self.q_values[subtask])

    def update_target_network(self):
        """Update the weights of the target network with the weights of the Q-network."""
        self.target_q_values = self.q_values.copy()  # Assuming q_values is the Q-network


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