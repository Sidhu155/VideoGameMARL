from collections import defaultdict
from gymnasium.spaces import Space
import numpy as np
import random
from .agent import Agent, assert_agent_set_up

class BaseQValAgent(Agent):
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        learning_decay: float,
        epsilon_decay: float,
        final_learning_rate: float,
        final_epsilon: float,
        discount_factor: float
    ):
        """Initialize a Q-Value based agent.

        Args:
            learning_rate: How quickly to update Q-values
            epsilon: Starting exploration rate
            learning_decay: How much to decay the learning rate
            epsilon_decay: How much to reduce epsilon each episode
            final_learning_rate: Minimum learning rate
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
               seed=None) -> None:
        super().set_up(action_space, seed=seed)
        self.numActions = action_space.n

    @assert_agent_set_up
    def get_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
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
    def update(self, reward: float, obs: np.ndarray, action: int) -> None:
        if self.learning and (self.prevObs is not None):
            next_q = self.get_next_q(obs, action)
            curr_q = self.get_q_value(self.prevObs, self.prevAction)
            
            target = reward + (self.discount_factor * next_q)
            temporal_difference = target - curr_q
            self.update_q_value(curr_q, temporal_difference)

            self.training_error.append(temporal_difference)
        
        self.prevObs = obs
        self.prevAction = action

    @assert_agent_set_up
    def final(self, reward: float) -> None:
        super().final(reward)
        self.decay()

    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        pass

    @assert_agent_set_up
    def get_max_q_value(self, obs: np.ndarray) -> float:
        pass

    @assert_agent_set_up
    def update_q_value(self, curr_q: float, temporal_difference: float) -> None:
        self.num_updates += 1

    @assert_agent_set_up
    def get_next_q(self, obs: np.ndarray, action: int) -> float:
        pass

    def decay(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        self.lr = max(self.final_lr, self.lr - self.lr_decay)      

class FuncApprox(BaseQValAgent):

    def __init__(
        self,
        learning_rate: float = 1e-3,
        epsilon: float = 1e-1,
        learning_decay: float = 1e-8,
        epsilon_decay: float = 1e-5,
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
        super().set_up(action_space, seed=seed)
        self.observation_space = observation_space
        self.numFeatures = np.prod(observation_space.shape)
        self.q_function = self.getDefaultFunc()

    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        vector = self.obs_to_feature_vector(obs, action)
        return vector @ self.q_function

    @assert_agent_set_up
    def get_max_q_value(self, obs: np.ndarray) -> float:
        return max(self.get_q_value(obs, action) for action in range(self.numActions))
    
    @assert_agent_set_up
    def update_q_value(self, curr_q: float, temporal_difference: float) -> None:
        super().update_q_value(curr_q, temporal_difference)
        obs_vector = self.obs_to_feature_vector(self.prevObs, self.prevAction)
        self.q_function += (obs_vector * (self.lr * temporal_difference))

    def obs_to_feature_vector(self, obs: np.ndarray, action: int) -> np.ndarray:
        obs_vector = np.ravel(obs)
        vector = np.zeros((self.q_function.size))
        vector[action * self.numFeatures:(action + 1) * self.numFeatures] += obs_vector
        vector[-1] += 1.0
        return vector

    def getDefaultFunc(self) -> np.ndarray:
        size = self.numActions * self.numFeatures
        return np.zeros(size + 1)

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
        super().set_up(action_space, seed=seed)
        self.q_values = defaultdict(self.getDefaultVals)

    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        return self.q_values[obs.tobytes()][action]

    @assert_agent_set_up
    def get_max_q_value(self, obs: np.ndarray) -> float:
        return np.max(self.q_values[obs.tobytes()])
    
    @assert_agent_set_up
    def update_q_value(self, curr_q: float, temporal_difference: float) -> None:
        super().update_q_value(curr_q, temporal_difference)
        new_q = curr_q + (self.lr * temporal_difference)
        self.q_values[self.prevObs.tobytes()][self.prevAction]= new_q

    @assert_agent_set_up
    def getDefaultVals(self) -> np.ndarray:
        return np.array(list(0.0 for _ in range(self.numActions)), dtype=np.float16)