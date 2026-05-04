from logger import Logger
from gymnasium.spaces import Space
import numpy as np
from utils import assert_agent_set_up

class Agent:
    """
    Base Agent class for interacting with an environment
    """

    def __init__(self, obs_abstraction: bool = False, action_abstraction: bool = False):
        """
        Initialise Agent. Enable learning and create logger.
        """

        self.obs_abstraction = obs_abstraction
        self.action_abstraction = action_abstraction
        self.learning = True
        self.set_up_bool = False
        self.logger = Logger()
    
    def set_up(self,
               action_space: Space, 
               observation_space: Space | None = None,
               seed: int | None = None) -> None:
        """
        Args:
            action_space: The action space that actions can be chosen from by the agent
            observation_space: The observation space of the environment for an agent.
            seed: This defines the seed for random sampling from the action_space.
        
        Performs any setting up required before the agent begins interaction with the environment.
        Raises exception if agent has already been set up.
        """

        if self.set_up_bool:
            raise Exception("Agent has already been set up!")
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_space.seed(seed)
        self.set_up_bool = True

    @assert_agent_set_up
    def get_action(self, obs: np.ndarray, mask: np.ndarray) -> int:
        """
        Args:
            obs: The current observation
            mask: A list of bools certifying which actions are legal
        
        Returns:
            A legal action for the agent to take
        """

        pass
    
    @assert_agent_set_up
    def update(self, reward: float, obs: np.ndarray, action: int) -> None:
        """
        Args:
            reward: The reward received for taking the previous action and observation
            obs: The current observation
            action: The action taken under the current observation
        
        The agent updates itself based on the previous observations and actions, 
        the reward and the current observation.
        """

        pass

    @assert_agent_set_up
    def final(self, reward: float) -> None:
        """
        Args:
            reward: The cumulative reward for the episode
        
        Record cumulative reward achieved throughout environment episode.
        Also perform any end of episode processes
        """
        try:
            reward = float(reward)
        except ValueError:
            raise ValueError(f"Final episode reward not a float value. Actual Type: {type(reward)}. Reward: {reward}")
        else:
            self.logger.updateLogs("record", reward)

    def enableLearning(self) -> None:
        """
        Enable Learning.
        This allows the agent to update itself based on information from the environment
        """

        self.learning = True

    def disableLearning(self) -> None:
        """
        Disable Learning.
        Agents no longer update themselves based on information received from the environment.
        """

        self.learning = False