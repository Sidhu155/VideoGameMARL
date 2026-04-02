import pettingzoo
from collections import defaultdict
import numpy as np
from .agent import Agent
import random

class QLearnAgent(Agent):

    def __init__(
        self,
        learning_rate: float = 0.2,
        initial_epsilon: float = 0.2,
        epsilon_decay: float = 0.000001,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95
    ):
        """Initialize a Q-Learning agent.

        Args:
            learning_rate: How quickly to update Q-values
            initial_epsilon: Starting exploration rate
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate
            discount_factor: How much to value future rewards
        """

        super().__init__()
        self.lr = learning_rate
        self.discount_factor = discount_factor 
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

        self.prevObs = None
        self.prevAction = None

    def get_action(self, obs: tuple[int, int, bool], mask) -> int:
        if (self.learning) and (np.random.random() < self.epsilon):
            return self.action_space.sample(mask)
        else:
            max = None
            maxActions = []
            for action in range(len(mask)):
                if mask[action] == 0:
                    continue
                else:
                    q = self.get_q_value(obs, action)
                    if max is None or max < q:
                        max = q
                        maxActions = [action]
                    elif max == q:
                        maxActions.append(action)
            return random.choice(maxActions)

    def update(self, reward: float, obs: tuple[int, int, bool], action: int):
        if self.learning and (self.prevObs is not None):
            if obs is None:
                next_q = 0
            else:
                next_q = np.max(self.q_values[obs])

            target = reward + (self.discount_factor * next_q)
            temporal_difference = target - self.q_values[self.prevObs][self.prevAction]
            self.q_values[self.prevObs][self.prevAction] += (self.lr * temporal_difference)

            self.training_error.append(temporal_difference)
        
        self.prevObs = obs
        self.prevAction = action

    def get_q_value(self, obs, action):
        return self.q_values[obs][action]

    def update_q_value(self, obs, action, new_q):
        self.q_values[obs][action]= new_q

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def set_up(self, action_space):
        super().set_up(action_space)
        self.numActions = action_space.n
        self.q_values = defaultdict(self.getDefaultVals)

    def getDefaultVals(self):
        return np.array(list(0 for _ in range(self.numActions)))

    def final(self, reward):
        super().final(reward)
        self.decay_epsilon()
