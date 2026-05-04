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

    def __init__(self, obs_abstraction: bool = True, action_abstraction: bool = True):
        """
        Initialise environment
        """

        self.obs_abstraction = obs_abstraction
        self.action_abstraction = action_abstraction
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
                abstracted_obs = self.get_obs_abstraction(obs)
                mask = observation["action_mask"]
                abstracted_mask = self.get_mask_abstraction(agent, mask)
                
                abstracted_action = currAgent.get_action(abstracted_obs, abstracted_mask)
                currAgent.update(reward, abstracted_obs, abstracted_action)

                action = self.convert_abstracted_action(agent, obs, mask, abstracted_action)

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

    def get_obs_abstraction(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def convert_abstracted_action(self, agent_name, obs, mask, abstracted_action):
        return abstracted_action
    
    def get_mask_abstraction(self, agent_name, mask):
        return mask

    def get_action_spaces(self) -> list:
        """
        Returns:
            List of Action Spaces for each player
        """

        return list(self.env.action_space(name)
                    for name in self.agent_names)

    def get_observation_spaces(self) -> list:
        """
        Returns:
            List of observation spaces for each player
        """

        return list(self.env.observation_space(name)["observation"]
                    for name in self.agent_names)
    
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