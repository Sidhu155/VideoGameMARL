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

    def create_env(self, render_type=None):
        self.env = self.env = connect_four_v3.env(render_mode=render_type, screen_scaling=7)
        self.env.reset()
