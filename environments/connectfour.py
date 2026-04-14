"""Contains Connect Four wrapper class and relevant methods"""
from agents.agent import Agent
from .environment import Environment
from pettingzoo.classic import connect_four_v3
import gymnasium as gym
from tqdm import tqdm

class ConnectFour(Environment):
    """
    A wrapper class around the Connect Four classic environment from PettingZoo.
    Inherits from Environment class
    """

    def __init__(self):
        super().__init__()

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

        self.env.reset()

    def create_env(self, render_type=None):
        self.env = self.env = connect_four_v3.env(render_mode=render_type)
        self.env.reset()
