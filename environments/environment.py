from agents.agent import Agent
from tqdm import tqdm

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

    def run(self, agent0: Agent, agent1: Agent):
        """
        Args:
            agent0: The player Agent
            agent1: The adversary agent

        Runs one episode of the environment
        At each step, gets actions from the agent by providing current observation.
        Updates agent with current reward and obs.
        """

        pass

    def runNumGames(self, agent0, agent1, numGames):
        """
        Args:
            agent0: The player Agent
            agent1: The adversary agent
            numGames: Number of Games to play
        
        Runs a given number of episodes within the environment.
        """

        for _ in tqdm(range(numGames)):
            self.run(agent0, agent1)

    def get_action_spaces(self) -> list:
        """
        Returns:
            List of Action Spaces for each player
        """

        return [
            self.env.action_space("player_0"),
            self.env.action_space("player_1")
        ]
    
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

    #https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
    def totuple(self, a):
        try:
            return tuple(self.totuple(i) for i in a)
        except TypeError as e:
            return a