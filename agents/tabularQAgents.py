from collections import defaultdict
from gymnasium.spaces import Space
import numpy as np
from .baseQAgent import BaseQValAgent
from .decorators import assert_agent_set_up, time_func
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
        learning_decay: float = 2e-6,
        epsilon_decay: float = 2e-6,
        final_learning_rate: float = 2e-2,
        final_epsilon: float = 2e-2,
        discount_factor: float = 0.95
    ):

        super().__init__(
            learning_rate, epsilon, learning_decay, epsilon_decay,
            final_learning_rate, final_epsilon, discount_factor
        )
    
    def set_up(self,
               action_space: Space,
               observation_space: Space | None = None,
               seed=None) -> None:
        super().set_up(action_space, observation_space, seed=seed)
        self.q_values = defaultdict(self.getDefaultVals)

    @time_func("get_q_value")
    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        return self.q_values[obs.tobytes()][action]
    
    @time_func("update_q_value")
    @assert_agent_set_up
    def update_q_value(self, curr_q: float, temporal_difference: float) -> None:
        super().update_q_value(curr_q, temporal_difference)
        new_q = curr_q + (self.lr * temporal_difference)
        self.q_values[self.prevObs.tobytes()][self.prevAction]= new_q

    @assert_agent_set_up
    def getDefaultVals(self) -> np.ndarray:
        """
        Return:
            Default initial Q-Values for each observation in the Q-table

        Q-Values are initialised to 0.
        """

        return np.array(list(0.0 for _ in range(self.numActions)), dtype=np.float16)
    
class QTabAgent(QLearnMixin, Tabular):
    pass

class SARSATabAgent(SARSAMixin, Tabular):
    pass