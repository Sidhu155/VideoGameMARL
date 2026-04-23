from collections import defaultdict
from agents.agent import Agent
from tqdm import tqdm
from utils import time_func

class Environment:
    """
    Base Environment Class
    Used as a wrapper for pettingzoo and custom environments
    """

    def __init__(self):
        """
        Initialise environment
        """

        self.create_env()
        self.agent_names = [
            "player_0",
            "player_1"
        ]
        self.logger: defaultdict = defaultdict(list)

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
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            currAgent = agent_list[self.agent_names.index(agent)]

            obs = observation["observation"]
            if termination or truncation:
                obs = None
                action = None
                currAgent.final(reward)
            else:
                mask = observation["action_mask"]
                action = currAgent.get_action(obs, mask)

            currAgent.update(reward, obs, action)
            self.env.step(action)

            num_iterations += 1

        self.logger["num_iterations"].append(num_iterations)
        self.env.reset()

    def runNumGames(self, agent_list: tuple[Agent], numGames: int):
        """
        Args:
            agent0: The player Agent
            agent1: The adversary agent
            numGames: Number of Games to play
        
        Runs a given number of episodes within the environment.
        """

        for _ in tqdm(range(numGames)):
            self.run(agent_list)

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