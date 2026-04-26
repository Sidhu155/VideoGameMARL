from gymnasium.spaces import Space
import numpy as np
from .baseQAgent import BaseQValAgent
from utils import assert_agent_set_up
from .qMixins import QLearnMixin, SARSAMixin

class FuncApprox(BaseQValAgent):
    """
    Base class for function approximation based q-value agents.
    Uses dimension scaling trick for q-function in order to estimate observation-action q-values.
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        epsilon: float = 1e-1,
        learning_decay: float = 1e-8,
        epsilon_decay: float = 1e-6,
        final_learning_rate: float = 1e-4,
        final_epsilon: float = 1e-2,
        discount_factor: float = 0.95
    ):

        super().__init__(
            learning_rate, epsilon, learning_decay, epsilon_decay,
            final_learning_rate, final_epsilon, discount_factor
        )
        

    def set_up(self, 
               action_space: Space,
               observation_space: Space,
               seed=None) -> None:
        super().set_up(action_space, observation_space, seed=seed)
        self.numFeatures = np.prod(observation_space.shape)
        self.q_function = self.getDefaultFunc()

    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        super().get_q_value(obs, action)
        vector = self.obs_to_feature_vector(obs, action)
        return vector @ self.q_function
    
    @assert_agent_set_up
    def update_q_value(self, curr_q: float, temporal_difference: float) -> None:
        super().update_q_value(curr_q, temporal_difference)
        obs_vector = self.obs_to_feature_vector(self.prevObs, self.prevAction)
        self.q_function += (obs_vector * (self.lr * temporal_difference))

    @assert_agent_set_up
    def get_max_q_value(self, obs):
        if obs.size != self.numFeatures:
            raise ValueError(f"Observation has size {obs.size} but size {self.numFeatures} is required!")
        
        obs_vector = np.ravel(obs)
        matrix = np.zeros((self.numActions, self.q_function.size))
        for action in range(self.numActions):
            matrix[action][action * self.numFeatures:(action + 1) * self.numFeatures] += obs_vector
            matrix[action][-1] += 1

        return np.max(matrix @ self.q_function)

    def obs_to_feature_vector(self, obs: np.ndarray, action: int) -> np.ndarray:
        """
        Args:
            obs: An observation in the form an ndarray
            action: An action
            
        Returns:
            A feature vector (ndarray) representing the action and the observation
            
        This method converts an observation and an action into a combined vector to be
        used when calculating the q_value of said observation-action pair.
        """

        if obs.size != self.numFeatures:
            raise ValueError(f"Observation has size {obs.size} but size {self.numFeatures} is required!")
        
        obs_vector = np.ravel(obs)  #Flatten observation array
        vector = np.zeros((self.q_function.size))   #Create vector of zeros in same size as q_function
        
        #Convert a section of indices in the vector to the obs vector based on the action taken.
        vector[action * self.numFeatures:(action + 1) * self.numFeatures] += obs_vector
        #Add 1 to the last index in the vector as this represents the intercept variable.
        vector[-1] += 1.0

        return vector

    def getDefaultFunc(self) -> np.ndarray:
        """
        Returns:
            A default q-function based on the number of actions and the number of features
            within the action_space and observation_space respectively.
        """

        size = self.numActions * self.numFeatures
        return np.zeros(size + 1)
    
class QFuncApproxAgent(QLearnMixin, FuncApprox):
    pass

class SARSAFuncApproxAgent(SARSAMixin, FuncApprox):
    pass