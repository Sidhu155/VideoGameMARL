import numpy as np
from collections import defaultdict
from .environment import Environment
from gymnasium.spaces import Discrete, Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils import AgentSelector

class DotsAndBoxes(Environment):

    def __init__(self, num_agents: int = 2, board_length: int = 5):
        self.num_agents = num_agents
        self.board_length = board_length
        super().__init__()
        self.agent_names = ["player_" + str(i) for i in range(num_agents)]

    def create_env(self, render_type: str | None = None):
        self.env = DotsAndBoxesEnvironment(render_type, self.num_agents, self.board_length)
        self.env.reset()

class DotsAndBoxesEnvironment(AECEnv):
    metadata = {
        "name": "dots_and_boxes_v0",
        "render_modes": ["human"]
    }

    def __init__(self, render_mode=None, num_agents: int = 2, board_length: int = 5):
        self.possible_agents = ["player_" + str(i) for i in range(num_agents)]
        self.render_mode = render_mode
        self.board_length = board_length
        self.board_size = 2 * (self.board_length - 1) * self.board_length

        #The dots and boxes is represented through a square board with two layers. One represents
        #the horizontal lines and the other represents the vertical lines. The board is an array
        #initialised as ones indicating that a line can be placed there. Once a line is placed
        #it is converted into a 0. It is also represented as a flat array. It is done this way 
        #to make action masking and indexing into the board more efficient.
        self.board = np.ones((self.board_size,), dtype=np.int8)



    def reset(self, seed = None, options = None):
        self.board = np.ones((self.board_size,), dtype=np.int8)

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.num_moves = 0
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()
    
    def step(self, action):
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        assert self.board[action] == 1, "Invalid action selected"
        self.board[action] = 0

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        reward_agent = 0
        midpoint = int(self.board_size/2)
        if action < midpoint:
            row = int(action/self.board_length)
            col = action % self.board_length
            if ((col != 0) and (self.board[action - 1] == 0) and
                (self.board[midpoint + (row * (self.board_length - 1)) + (col - 1)] == 0) and
                (self.board[midpoint + ((row + 1) * (self.board_length - 1)) + (col - 1)] == 0)):
                reward_agent += 1
            if ((col != (self.board_length - 1)) and (self.board[action + 1] == 0) and
                (self.board[midpoint + (row * (self.board_length - 1)) + (col)] == 0) and
                (self.board[midpoint + ((row + 1) * (self.board_length - 1)) + (col)] == 0)):
                reward_agent += 1
        else:
            row = int((action - midpoint)/(self.board_length - 1))
            col = (action - midpoint)%(self.board_length - 1)
            if ((row != 0) and (self.board[action - (self.board_length - 1)] == 0) and
                (self.board[((row - 1) * self.board_length) + (col)] == 0) and
                (self.board[((row - 1) * self.board_length) + (col + 1)] == 0)):
                reward_agent += 1
            if ((row != self.board_length - 1) and (self.board[action + (self.board_length - 1)] == 0) and
                (self.board[((row) * self.board_length) + (col)] == 0) and
                (self.board[((row) * self.board_length) + (col + 1)] == 0)):
                reward_agent += 1

        if reward_agent == 0:
            self.agent_selection = self._agent_selector.next()
        else:
            self.rewards[agent] += reward_agent
            if self.agents.index(agent) == 0:
                for agent_idx in range(1, len(self.agents)):
                    self.rewards[self.agents[agent_idx]] -= reward_agent
            else:
                self.rewards[self.agents[0]] -= reward_agent
            self._accumulate_rewards()
        
        self.num_moves += 1
        if self.num_moves >= self.board_size:
            self.terminations = {agent: True for agent in self.agents}

        if self.render_mode == "human":
            self.render()
    
    def render(self):
        if self.render_mode is None:
            return
        else:
            midpoint = int(self.board_size/2)
            print('.' + '.'.join(
                ('_' if hor_line == 0 else ' ' for hor_line in self.board[midpoint:midpoint + self.board_length - 1])
                ) + '.')
            for i in range(self.board_length - 1):
                if self.board[self.board_length * i] == 0:
                    string = '|'
                else:
                    string = '.'
                
                start_idx_hor = midpoint + (self.board_length - 1) * (i + 1)
                for hor_line, ver_line in zip(
                    self.board[start_idx_hor:start_idx_hor + self.board_length - 1],
                    self.board[(self.board_length * i) + 1:self.board_length * (i + 1)]):
                    if hor_line == 0: string += '_' 
                    else: string += ' '
                    
                    if ver_line == 0: string += '|'
                    else: string += '.'
                
                print(string)
            print()
    
    def observe(self, agent):
        return {
            "observation": np.copy(self.board), #May be able to remove copy
            "action_mask": self._get_mask(agent)
        }
    
    def observation_space(self, agent):
        return Dict(
            {
                "observation": Box(
                    low=0, high=1, shape=(self.board_size,), dtype=np.int8
                ),
                "action_mask": Discrete(self.board_size, dtype=np.int8)
            }
        )
    
    def action_space(self, agent):
        return Discrete(self.board_size, dtype=np.int8)
    
    def _get_mask(self, agent):
        # Per the documentation, the mask of any agent other than the
        # currently selected one is all zeros.
        if agent == self.agent_selection:
            return np.copy(self.board)
        else:
            return np.zeros(self.board_size, dtype=np.int8)