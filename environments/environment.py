from agents.agent import Agent
from tqdm import tqdm

class Environment:
    def __init__(self):
        self.create_env()

    def run(self, agent0: Agent, agent1: Agent):
        pass

    def runNumGames(self, agent0, agent1, numGames):
        for _ in tqdm(range(numGames)):
            self.run(agent0, agent1)

    def get_action_spaces(self) -> list:
        return [
            self.env.action_space("player_0"),
            self.env.action_space("player_1")
        ]
    
    def enable_rendering(self):
        self.env.close()
        self.create_env("human")

    def disable_rendering(self):
        self.env.close()
        self.create_env()

    def create_env(self, render_type=None):
        pass

    def tear_down(self):
        self.env.close()

    #https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
    def totuple(self, a):
        try:
            return tuple(self.totuple(i) for i in a)
        except Exception:
            return a