from collections import defaultdict
from gymnasium.spaces import Space
import numpy as np
import random

class Agent:
    """
    Base Agent class for interacting with an environment
    """

    def __init__(self):
        """
        Initialise Agent. Enable learning and create record of wins.
        """

        self.record: list[float] = []
        self.learning = True

    def set_up(self, action_space: Space, seed: int | None = None):
        """
        Args:
            action_space: The action_space that actions can be chosen from by the agent
            seed: This defines the seed for random sampling from the action_space.
        
        Performs any setting up required before the agent begins interaction with the environment.
        Seeds action space if seed is provided.
        """

        self.action_space = action_space
        self.action_space.seed(seed)

    def get_action(self, obs: tuple, mask) -> int:
        """
        Args:
            obs: The current observation
            mask: A list of bools certifying which actions are legal
        
        Returns:
            A legal action for the agent to take
        """

        pass

    def update(self, reward, obs, action):
        """
        Args:
            reward: The reward received for taking the previous action and observation
            obs: The current observation
            action: The action taken under the current observation
        
        The agent updates itself based on the previous observations and actions, 
        the reward and the current observation.
        """

        pass

    def final(self, reward: float):
        """
        Args:
            reward: The cumulative reward for the episode
        
        Record cumulative reward achieved throughout environment episode.
        Also perform any end of episode processes
        """
        try:
            reward = float(reward)
        except ValueError:
            pass
        else:
            self.record.append(float(reward))

    def enableLearning(self):
        """
        Enable Learning.
        This allows the agent to update itself based on information from the environment
        """

        self.learning = True

    def disableLearning(self):
        """
        Disable Learning.
        Agents no longer update themselves based on information received from the environment.
        RL based agents may only pick best known actions instead of explorative actions.
        """

        self.learning = False


class BaseQValAgent(Agent):
    pass

class FuncApprox:
    pass