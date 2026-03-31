"""Contains Connect Four wrapper class and relevant methods"""
from agents.agent import Agent
from pettingzoo.classic import connect_four_v3
import gymnasium as gym
from tqdm import tqdm

class ConnectFour:
    """A wrapper class around the Connect Four classic environment from PettingZoo"""

    def __init__(self):
        """Initialises object with two agents"""
        self.env = connect_four_v3.env()
        self.env.reset()

    def run(self, agent0: Agent, agent1: Agent):
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            currAgent = agent0 if agent == "player_0" else agent1

            obs = self.totuple(observation["observation"])
            if termination or truncation:
                obs = None
                action = None
                currAgent.final(reward)
            else:
                mask = observation["action_mask"]
                action = currAgent.get_action(obs, mask)

            currAgent.update(reward, obs, action)
            self.env.step(action)

        self.env.close()

    def runNumGames(self, agent0, agent1, numGames):
        for i in tqdm(range(numGames)):
            self.run(agent0, agent1)
            self.env.reset()

    def get_action_spaces(self) -> list:
        return [
            self.env.action_space("player_0"),
            self.env.action_space("player_1")
        ]

    #https://stackoverflow.com/questions/10016352/convert-numpy-array-to-tuple
    def totuple(self, a):
        try:
            return tuple(self.totuple(i) for i in a)
        except Exception:
            return a