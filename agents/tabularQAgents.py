from collections import defaultdict
from gymnasium.spaces import Space
import numpy as np
import random
from .baseQAgent import BaseQValAgent
from utils import assert_agent_set_up, time_func
from .qMixins import QLearnMixin, SARSAMixin

class Tabular(BaseQValAgent):
    """
    Tabular q-value based agent
    Converts observations to bytes and adds to a dictionary with action utilities
    """
    
    def __init__(
        self,
        learning_rate: float = 2e-1,
        epsilon: float = 2e-1,
        learning_decay: float = 2e-7,
        epsilon_decay: float = 2e-7,
        final_learning_rate: float = 2e-2,
        final_epsilon: float = 2e-2,
        discount_factor: float = 0.95,
        obs_abstraction: bool = False,
        action_abstraction: bool = False
    ):

        super().__init__(
            learning_rate, epsilon, learning_decay, epsilon_decay,
            final_learning_rate, final_epsilon, discount_factor, 
            obs_abstraction, action_abstraction
        )
    
    def set_up(self,
               action_space: Space,
               observation_space: Space | None = None,
               seed=None) -> None:
        super().set_up(action_space, observation_space, seed=seed)
        self.q_values = defaultdict(self.getDefaultVals)

    @time_func("get_action")
    @assert_agent_set_up
    def get_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        if (self.learning) and (np.random.random() < self.epsilon):
            return self.action_space.sample(mask)
        else:
            #Get Q-values from Q-table
            q_vals = self.q_values[obs.tobytes()]
            max = None
            maxActions = []

            #Iterate over all actions. If action is legal, check if q-val is greater or equal to max
            for action in range(self.numActions):
                if mask[action] == 1:
                    if max is None or q_vals[action] > max:
                        max = q_vals[action]
                        maxActions = [action]
                    elif q_vals[action] == max:
                        maxActions.append(action)

            self.next_max_q = max
            return random.choice(maxActions)
        
    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        super().get_q_value(obs, action)
        return self.q_values[obs.tobytes()][action]
    
    @assert_agent_set_up
    def get_max_q_value(self, obs: np.ndarray) -> float:
        return np.max(self.q_values[obs.tobytes()])
    
    @assert_agent_set_up
    def update_q_value(self, obs: np.ndarray, action: int, curr_q: float, temporal_difference: float) -> None:
        super().update_q_value(obs, action, curr_q, temporal_difference)
        new_q = curr_q + (self.lr * temporal_difference)
        self.q_values[obs.tobytes()][action]= new_q

    @assert_agent_set_up
    def getDefaultVals(self) -> np.ndarray:
        """
        Return:
            Default initial Q-Values for each observation in the Q-table

        Q-Values are initialised to 0.
        """

        return np.zeros(self.numActions, dtype=np.float16)
    
class QTabAgent(QLearnMixin, Tabular):
    pass

class SARSATabAgent(SARSAMixin, Tabular):
    pass