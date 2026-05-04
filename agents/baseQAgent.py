from gymnasium.spaces import Space
import numpy as np
import random
from .agent import Agent
from utils import assert_agent_set_up, time_func

class BaseQValAgent(Agent):
    """
    Base Q-Value based agent class. Inherited by tabular and func-approximation agents.
    """
    
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        learning_decay: float,
        epsilon_decay: float,
        final_learning_rate: float,
        final_epsilon: float,
        discount_factor: float,
        obs_abstraction: bool = False,
        action_abstraction: bool = False
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

        super().__init__(obs_abstraction, action_abstraction)
        self.lr = learning_rate
        self.epsilon = epsilon
        self.lr_decay = learning_decay
        self.epsilon_decay = epsilon_decay
        self.final_lr = final_learning_rate
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor 

        self.prevObs = None
        self.prevAction = None
        self.next_max_q = None

    def set_up(self,
               action_space: Space,
               observation_space: Space | None = None,
               seed=None) -> None:
        super().set_up(action_space, observation_space, seed=seed)
        self.numActions = action_space.n

    @time_func("get_action")
    @assert_agent_set_up
    def get_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Args:
            obs: The current observation
            mask: A list of bools certifying which actions are legal
        
        Returns:
            A legal action for the agent to take

        The Q-Value agent uses epsilon-based exploration in order to choose an action.
        """

        if (self.learning) and (np.random.random() < self.epsilon):
            return self.action_space.sample(mask)
        else:
            max = None
            maxActions = []

            #Iterate over all actions in action space.
            for action in range(len(mask)):
                if mask[action] == 0:
                    #If action is illegal, continue
                    continue
                else:
                    #If action is legal compare q against max
                    q = self.get_q_value(obs, action)
                    if max is None or max < q:
                        max = q
                        maxActions = [action]
                    elif max == q:
                        maxActions.append(action)
            
            #Set the maximum q_value for the observation to avoid recalculation in update and get_next_q
            self.next_max_q = max
            #Randomly choose from list of maxActions. This avoids always picking actions with lower index
            return random.choice(maxActions)

    @time_func("update")
    @assert_agent_set_up
    def update(self, reward: float, obs: np.ndarray, action: int) -> None:
        if self.learning and (self.prevObs is not None):
            next_q = self.get_next_q(obs, action)   #If obs is none, next_q returns 0, else returns q_val for obs
            curr_q = self.get_q_value(self.prevObs, self.prevAction)
            
            target = reward + (self.discount_factor * next_q)
            temporal_difference = target - curr_q
            self.update_q_value(self.prevObs, self.prevAction, curr_q, temporal_difference)

            self.logger.updateLogs("training_error", temporal_difference)
        
        self.prevObs = obs
        self.prevAction = action

    @assert_agent_set_up
    def final(self, reward: float) -> None:
        super().final(reward)
        self.decay()

    @assert_agent_set_up
    def get_q_value(self, obs: np.ndarray, action: int) -> float:
        """
        Args:
            obs: The current observation for the agent
            action: An action that could be taken

        Returns:
            A Q-Value for the observation-action combination
        """

        pass

    @assert_agent_set_up
    def get_max_q_value(self, obs: np.ndarray) -> float:
        """
        Args:
            obs: The current observation
            
        Returns:
            The maximum attainable Q-Value for an observation
            
        Find and return the maximum possible Q-Value by checking q-values of all actions
        under an observation
        """

        return max(self.get_q_value(obs, action) for action in range(self.numActions))

    @assert_agent_set_up
    def update_q_value(self, obs: np.ndarray, action: int, curr_q: float, temporal_difference: float) -> None:
        """
        Args:
            curr_q: The current q-value of the previous observation
            temporal_difference: The difference between the current q-value and the target
        
        Use learning rate and temporal-difference to update future q-value estimates
        """

        pass

    @assert_agent_set_up
    def get_next_q(self, obs: np.ndarray, action: int) -> float:
        """
        Args:
            obs: The current observation
            action: The action taken under the observation

        Returns:
            A q-value

        This function is implemented in the Q-Learning and SARSA mixins and is used to
        find the next q-value during the update step.
        """

        pass

    def decay(self) -> None:
        """
        Decay epsilon and learning rate linearly based on the parameters epsilon_decay 
        and lr_decay provided to the agent.
        """

        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        self.lr = max(self.final_lr, self.lr - self.lr_decay)