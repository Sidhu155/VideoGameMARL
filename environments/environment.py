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

        #Names of agents used by self.env
        self.agent_names = [
            "player_0",
            "player_1"
        ]
        self.logger = Logger()
        self.create_env()
        self.set_observations: set = set()      #Record unique states visited by environment object

    @log_memory_func("mem_run")
    @time_func("run")
    def run(self, agent_list: tuple[Agent]):
        """
        Args:
            agent_list: List of agents that interact with the game

        Runs one episode of the environment.
        At each step, gets actions from the agent by providing current observation.
        Updates agent with current reward and obs.
        """
        
        num_iterations = 0
        #Initalise list of cumulative rewards for each agent
        rewards = list(0 for _ in range(len(agent_list)))

        #Carry out an episode by iterating over agents.
        for agent_name in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            
            agent_idx = self.agent_names.index(agent_name)   #Use agent_name to find index of agent
            agent = agent_list[agent_idx]
            rewards[agent_idx] += reward

            #Observation is a dict of observation and action-mask. Get obs from observation
            obs = observation["observation"]
            self.set_observations.add(obs.tobytes())

            #Check if episode is finished.
            if termination or truncation:
                obs = None
                action = None
                agent.final(rewards[agent_idx])         #Update agent logs with cumulative rewards
                agent.update(reward, obs, action)
            else:
                if agent.obs_abstraction:
                    agent_obs = self.get_abstract_obs(agent_idx, obs)
                else:
                    agent_obs = obs

                mask = observation["action_mask"]

                #If agent uses action abstraction, get abstract mask, get abstract action from agent
                #Use abstracted action to update agent, but use real action to step environment
                if agent.action_abstraction:
                    agent_mask = self.get_abstract_mask(agent_idx, mask)
                    agent_action = agent.get_action(agent_obs, agent_mask)
                    agent.update(reward, agent_obs, agent_action)
                    action = self.convert_abstract_action(agent_idx, obs, mask, agent_action)
                else:
                    action = agent.get_action(agent_obs, mask)
                    agent.update(reward, agent_obs, action)

            self.env.step(action)
            num_iterations += 1

        self.logger.updateLogs("num_iterations", num_iterations)
        self.logger.updateLogs("num_states", len(self.set_observations))
        self.env.reset()

    def runNumGames(self, agent_list: tuple[Agent], numGames: int):
        """
        Args:
            agent_list: A list of agents to interact with the environment
            numGames: Number of Games to play
        
        Runs a given number of episodes within the environment.
        """

        if len(agent_list) != len(self.agent_names):
            raise Exception("Incorrect number of agents provided")
        
        for _ in tqdm(range(numGames)):
            self.run(agent_list)

    def get_abstract_obs(self, agent_idx: int, obs: np.ndarray) -> np.ndarray:
        """
        Args:
            agent_idx: An integer to index into the object's list of agent name
            obs: An array representing a non-abstract observation inside the environment
            
        Returns:
            An abstract observation. By default, environments are assumed not to utilise
            abstraction, returning the parameter obs instead."""
        
        return obs
    
    def get_abstract_mask(self, agent_idx: int, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            agent_idx: An integer to index into the object's list of agent name
            mask: An array representing a non-abstract action mask.
            
        Returns:
            An abstract action mask. By default, environments are assumed not to utilise
            abstraction, returning the parameter mask instead.
        """
        
        return mask

    def convert_abstract_action(self, agent_idx: int, obs: np.ndarray, 
                                mask: np.ndarray, abstracted_action: int) -> int:
        """
        Args:
            agent_idx: An integer to index into the object's list of agent name
            obs: An array representing a non-abstract observation inside the environment.
            mask: An array representing a non-abstract action mask.
            abstracted_action: An integer representing the abstract action taken.
            
        Returns:
            A non-abstract action. By default, environments are assumed not to utilise
            abstraction, therefore the parameter abstracted_action is returned as it should not be
            abstract regardless
        """
        return abstracted_action

    def get_action_space(self, idx: int, abstract: bool) -> list:
        """
        Args:
            idx: Representing the agent index relative to the environment's agent names
            abstract: A boolean representing whether the agent uses abstract actions or not

        Returns:
            Action space for agent
        """

        return self.env.action_space(self.agent_names[idx])

    def get_observation_space(self, idx: int, abstract: bool) -> list:
        """
        Args:
            idx: Representing the agent index relative to the environment's agent names
            abstract: A boolean representing whether the agent uses abstract observations or not

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
        Args:
            render_type: A string representing the render_type. Can also be None
        
        Create a new environment and reset it. 
        """

        self.env.reset()

    def tear_down(self):
        """
        Destroy environment
        """

        self.env.close()
        self.set_observations = set()