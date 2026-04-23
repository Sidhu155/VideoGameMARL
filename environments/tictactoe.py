from .environment import Environment
from agents.agent import Agent
from pettingzoo.classic import tictactoe_v3

class TicTacToe(Environment):
    """
    A wrapper class around the TicTacToe classic environment from PettingZoo.
    Inherits from Environment class
    """

    def __init__(self):
        super().__init__()
        self.agent_names = [
            "player_1",
            "player_2"
        ]
    
    def create_env(self, render_type=None):
        self.env = self.env = tictactoe_v3.env(render_mode=render_type, screen_height=600)
        self.env.reset()