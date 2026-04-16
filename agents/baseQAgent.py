from collections import defaultdict
from gymnasium.spaces import Space
import numpy as np
import random
from .agent import Agent, assert_agent_set_up

class BaseQValAgent(Agent):
    def __init__(
        self,
        learning_rate: float = 0.2,
        epsilon: float = 0.2,
        learning_decay: float = 0.0,
        epsilon_decay: float = 0.000002,
        final_learning_rate: float = 0.2,
        final_epsilon: float = 0.05,
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
        self.epsilon = epsilon
        self.lr_decay = learning_decay
        self.epsilon_decay = epsilon_decay
        self.final_lr = final_learning_rate
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor 

        self.training_error = []
        self.prevObs = None
        self.prevAction = None

        self.num_updates = 0

    def set_up(self,
               action_space: Space,
               observation_space: Space | None = None,
               seed=None):
        super().set_up(action_space, seed=seed)
        self.numActions = action_space.n

    @assert_agent_set_up
    def get_action(self, obs: tuple, mask) -> int:
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

    @assert_agent_set_up
    def update(self, reward: float, obs: tuple, action: int):
        if self.learning and (self.prevObs is not None):
            next_q = self.get_next_q(obs, action)
            curr_q = self.get_q_value(self.prevObs, self.prevAction)
            
            target = reward + (self.discount_factor * next_q)
            temporal_difference = target - curr_q
            new_q = curr_q + (self.lr * temporal_difference)
            
            self.update_q_value(self.prevObs, self.prevAction, new_q)
            self.training_error.append(temporal_difference)
        
        self.prevObs = obs
        self.prevAction = action

    @assert_agent_set_up
    def final(self, reward):
        super().final(reward)
        self.decay()

    @assert_agent_set_up
    def get_q_value(self, obs, action) -> float:
        pass

    @assert_agent_set_up
    def get_max_q_value(self, obs) -> float:
        pass

    @assert_agent_set_up
    def update_q_value(self, obs, action, new_q: float):
        self.num_updates += 1

    @assert_agent_set_up
    def get_next_q(self, obs, action):
        pass

    def decay(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        self.lr = max(self.final_lr, self.lr - self.lr_decay)

    @assert_agent_set_up
    def get_next_q(self, obs, action):
        pass
        

class FuncApprox:
    pass

class Tabular(BaseQValAgent):
    """
    Tabular q-value based agent
    Converts observations to bytes and adds to a dictionary with action utilities
    """

    def set_up(self,
               action_space: Space,
               observation_space: Space | None = None,
               seed=None):
        super().set_up(action_space, seed=seed)
        self.q_values = defaultdict(self.getDefaultVals)

    @assert_agent_set_up
    def get_q_value(self, obs, action) -> float:
        return self.q_values[obs.tobytes()][action]

    @assert_agent_set_up
    def get_max_q_value(self, obs) -> float:
        return np.max(self.q_values[obs.tobytes()])
    
    @assert_agent_set_up
    def update_q_value(self, obs, action, new_q: float):
        super().update_q_value(obs, action, new_q)
        self.q_values[obs.tobytes()][action]= new_q

    @assert_agent_set_up
    def getDefaultVals(self):
        return np.array(list(0.0 for _ in range(self.numActions)), dtype=np.float16)