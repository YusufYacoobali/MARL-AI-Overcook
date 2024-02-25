from collections import namedtuple
import numpy as np
import scipy as sp
import random
from utils.utils import agent_settings

import torch
import torch.nn as nn
import torch.optim as optim

class SubtaskAllocDistribution():
    """Represents a distribution over subtask allocations."""

    def __init__(self, subtask_allocs):
        # subtask_allocs are a list of tuples of (subtask, subtask_agents).
        self.probs = {}
        if len(subtask_allocs) == 0:
            return
        prior = 1./(len(subtask_allocs))
        print('set prior', prior)

        for subtask_alloc in subtask_allocs:
            self.probs[tuple(subtask_alloc)] = prior

    def __str__(self):
        s = ''
        for subtask_alloc, p in self.probs.items():
            s += str(subtask_alloc) + ': ' + str(p) + '\n'
        return s

    def enumerate_subtask_allocs(self):
        return list(self.probs.keys())

    def get_list(self):
        return list(self.probs.items())

    def get(self, subtask_alloc):
        return self.probs[tuple(subtask_alloc)]

    def get_max(self):
        if len(self.probs) > 0:
            max_prob = max(self.probs.values())
            max_subtask_allocs = [subtask_alloc for subtask_alloc, p in self.probs.items() if p == max_prob]
            return random.choice(max_subtask_allocs)
        return None

    def get_max_bucketed(self):
        subtasks = []
        probs = []
        for subtask_alloc, p in self.probs.items():
            for t in subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    # If already accounted for, then add probability.
                    if t in subtasks:
                        probs[subtasks.index(t)] += p
                    # Otherwise, make a new element in the distribution.
                    else:
                        subtasks.append(t)
                        probs.append(p)
        best_subtask = subtasks[np.argmax(probs)]
        return self.probs.get_best_containing(best_subtask)

    def get_best_containing(self, subtask):
        """Return max likelihood subtask_alloc that contains the given subtask."""
        valid_subtask_allocs = []
        valid_p = []
        for subtask_alloc, p in self.probs.items():
            if subtask in subtask_alloc:
                valid_subtask_allocs.append(subtask)
                valid_p.append(p)
        return valid_subtask_allocs[np.argmax(valid_p)]

    def set(self, subtask_alloc, value):
        self.probs[tuple(subtask_alloc)] = value

    def update(self, subtask_alloc, factor):
        self.probs[tuple(subtask_alloc)] *= factor

    def delete(self, subtask_alloc):
        try:
            del self.probs[tuple(subtask_alloc)]
        except:
            print('subtask_alloc {} not found in probsdict'.format(subtask_alloc))

    def normalize(self):
        total = sum(self.probs.values())
        for subtask_alloc in self.probs.keys():
            if total == 0:
                self.probs[subtask_alloc] = 1./len(self.probs)
            else:
                self.probs[subtask_alloc] *= 1./total
        return self.probs





class QLearningAgent:
    def __init__(self, agent_name, subtasks, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64):
        self.agent_name = agent_name
        self.subtasks = subtasks
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory = []
        self.gamma = gamma
        self.model = QNetwork(len(subtasks))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def remember(self, action, reward, next_action, done):
        self.memory.append((action, reward, next_action, done))

    def choose_action(self):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(len(self.subtasks))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(np.eye(len(self.subtasks))[self.agent_name]).unsqueeze(0)
                q_values = self.model(state)
                return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for action, reward, next_action, done in batch:
            state = torch.FloatTensor(np.eye(len(self.subtasks))[action]).unsqueeze(0)
            next_state = torch.FloatTensor(np.eye(len(self.subtasks))[next_action]).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(state).clone()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.loss_function(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class QNetwork(nn.Module):
    def __init__(self, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_actions, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x




# class DQN(nn.Module):
#     def __init__(self, num_actions):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(num_actions, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, num_actions)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class RLAgent:
#     def __init__(self, agent_name, actions, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64):
#         self.agent_name = agent_name
#         self.actions = actions
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min
#         self.batch_size = batch_size
#         self.memory = []
#         self.gamma = gamma
#         self.model = DQN(len(actions))
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.loss_function = nn.MSELoss()

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def choose_action(self, state):
#         if np.random.rand() <= self.epsilon:
#             return np.random.choice(self.actions)
#         else:
#             with torch.no_grad():
#                 state = torch.FloatTensor(state).unsqueeze(0)
#                 q_values = self.model(state)
#                 return torch.argmax(q_values).item()

#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         batch = random.sample(self.memory, self.batch_size)
#         for state, action, reward, next_state, done in batch:
#             state = torch.FloatTensor(state).unsqueeze(0)
#             next_state = torch.FloatTensor(next_state).unsqueeze(0)
#             target = reward
#             if not done:
#                 target = reward + self.gamma * torch.max(self.model(next_state).detach())
#             target_f = self.model(state).clone()
#             target_f[0][action] = target
#             self.optimizer.zero_grad()
#             loss = self.loss_function(self.model(state), target_f)
#             loss.backward()
#             self.optimizer.step()
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay