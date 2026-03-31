import pettingzoo
from collections import defaultdict
import numpy as np
from .agent import Agent

class QLearnAgent(Agent):

    def __init__(
        self,
        learning_rate: float = 0.2,
        initial_epsilon: float = 0.1,
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
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        
        if (self.learning) and (np.random.random() < self.epsilon):
            return self.action_space.sample(mask)
        else:
            max = None
            maxIdx = -1
            q_value = self.q_values[obs]
            for idx in range(len(mask)):
                if mask[idx] == 0:
                    continue
                elif max is None or max < q_value[idx]:
                    max = q_value[idx]
                    maxIdx = idx
            
            return int(maxIdx)

    def update(self, reward: float, obs: tuple[int, int, bool], action: int):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """

        if self.learning and (self.prevObs is not None):
            if obs is None:
                next_q = 0
            else:
                next_q = np.max(self.q_values[obs])
            

            # What should the Q-value be? (Bellman equation)
            target = reward + self.discount_factor * next_q

            # How wrong was our current estimate?
            temporal_difference = target - self.q_values[self.prevObs][self.prevAction]

            # Update our estimate in the direction of the error
            # Learning rate controls how big steps we take
            self.q_values[self.prevObs][self.prevAction] = (
                self.q_values[self.prevObs][self.prevAction] + self.lr * temporal_difference
            )

            # Track learning progress (useful for debugging)
            self.training_error.append(temporal_difference)
        
        self.prevObs = obs
        self.prevAction = action

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def set_up(self, action_space):
        super().set_up(action_space)
        self.numActions = action_space.n
        self.q_values = defaultdict(self.getDefaultVals)

    def getDefaultVals(self):
        return np.zeros(self.numActions)

    def final(self, reward):
        super().final(reward)
        self.decay_epsilon()
