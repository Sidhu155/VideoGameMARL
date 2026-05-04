from logger import Logger
from agents.agent import Agent
from tqdm import tqdm
from utils import time_func, log_memory_func
import numpy as np

class Environment:
    """
    Base Environment Class
    Used as a wrapper for pettingzoo and custom environments
    """

    def __init__(self):
        """
        Initialise environment
        """

        self.agent_names = [
            "player_0",
            "player_1"
        ]
        self.logger = Logger()
        self.create_env()

    @log_memory_func("mem_run")
    @time_func("run")
    def run(self, agent_list: tuple[Agent]):
        """
        Args:
            agent_list: List of agents that interact with the game

        Runs one episode of the environment
        At each step, gets actions from the agent by providing current observation.
        Updates agent with current reward and obs.
        """
        
        num_iterations = 0
        rewards = list(0 for _ in range(len(agent_list)))
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            
            agent_idx = self.agent_names.index(agent)
            currAgent = agent_list[agent_idx]
            rewards[agent_idx] += reward

            obs = observation["observation"]
            if termination or truncation:
                obs = None
                action = None
                currAgent.final(rewards[agent_idx])
                currAgent.update(reward, obs, action)
            else:
                if currAgent.obs_abstraction:
                    agent_obs = self.get_abstract_obs(agent_idx, obs)
                else:
                    agent_obs = obs

                mask = observation["action_mask"]
                if currAgent.action_abstraction:
                    agent_mask = self.get_abstract_mask(agent_idx, mask)
                    agent_action = currAgent.get_action(agent_obs, agent_mask)
                    currAgent.update(reward, agent_obs, agent_action)
                    action = self.convert_abstract_action(agent_idx, obs, mask, agent_action)
                else:
                    action = currAgent.get_action(agent_obs, mask)
                    currAgent.update(reward, agent_obs, action)

            self.env.step(action)
            num_iterations += 1

        self.logger.updateLogs("num_iterations", num_iterations)
        self.env.reset()

    def runNumGames(self, agent_list: tuple[Agent], numGames: int):
        """
        Args:
            agent0: The player Agent
            agent1: The adversary agent
            numGames: Number of Games to play
        
        Runs a given number of episodes within the environment.
        """

        if len(agent_list) != len(self.agent_names):
            raise Exception("Incorrect number of agents provided")
        
        for _ in tqdm(range(numGames)):
            self.run(agent_list)

    def get_abstract_obs(self, agent_idx, obs: np.ndarray) -> np.ndarray:
        return obs

    def convert_abstract_action(self, agent_idx, obs, mask, abstracted_action):
        return abstracted_action
    
    def get_abstract_mask(self, agent_idx, mask):
        return mask

    def get_action_space(self, idx: int, abstract: bool) -> list:
        """
        Args:
            idx: Representing agent index

        Returns:
            Action space for agent
        """

        return self.env.action_space(self.agent_names[idx])

    def get_observation_space(self, idx: int, abstract: bool) -> list:
        """
        Args:
            idx: Representing agent index

        Returns:
            Observation space for agent.
        """

        return self.env.observation_space(self.agent_names[idx])["observation"]
    
    def enable_rendering(self):
        """
        Renders environment in human_mode
        """

        self.env.close()
        self.create_env("human")

    def disable_rendering(self):
        """
        Disables human rendering of environment
        """

        self.env.close()
        self.create_env()

    def create_env(self, render_type=None):
        """
        Create environment
        """

        pass

    def tear_down(self):
        """
        Destroy environment
        """

        self.env.close()