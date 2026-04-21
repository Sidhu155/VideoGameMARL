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

    def run(self, agent0: Agent, agent1: Agent):
        for agent in self.env.agent_iter():
            observation, reward, termination, truncation, info = self.env.last()
            currAgent = agent0 if agent == "player_1" else agent1

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

        self.env.reset()

    def get_action_spaces(self) -> list:
        return [
            self.env.action_space("player_1"),
            self.env.action_space("player_2")
        ]

    def get_observation_spaces(self) -> list:
        return [
            self.env.observation_space("player_1")["observation"],
            self.env.observation_space("player_2")["observation"]
        ]
    
    def create_env(self, render_type=None):
        self.env = self.env = tictactoe_v3.env(render_mode=render_type, screen_height=600)
        self.env.reset()