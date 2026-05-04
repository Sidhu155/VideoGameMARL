import numpy as np
import random
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

    def get_abstract_obs(self, agent_idx, obs):
        ret_obs = np.zeros(5, dtype=np.int8)
        for x in np.nditer(self.env.filled_squares):
            ret_obs[x] += 1
        return ret_obs
    
    def convert_abstract_action(self, agent_idx, obs, mask, abstracted_action):
        midpoint = self.board_length * (self.board_length - 1)
        if abstracted_action == 0:
            priority = [4, 3, 1, 2]
        elif abstracted_action == 1:
            priority = [2, 1, 3, 4]
        elif abstracted_action == 2:
            priority = [3, 4, 2, 1]
        

        for prior in priority:
            indices = np.argwhere(self.env.filled_squares == prior)
            if len(indices) > 0:
                break

        index = np.random.choice(indices.shape[0])
        row = indices[index][0]
        col = indices[index][1]
        actions = []
        if obs[(row * self.board_length) + col] == 1: actions.append((row * self.board_length) + col)
        if obs[(row * self.board_length) + col + 1] == 1: actions.append((row * self.board_length) + (col + 1))
        if obs[midpoint + (row * (self.board_length - 1)) + col] == 1: 
            actions.append(midpoint + (row * (self.board_length - 1)) + col)
        if obs[midpoint + ((row + 1) * (self.board_length - 1)) + col] == 1:
            actions.append(midpoint + ((row + 1) * (self.board_length - 1)) + col)

        return random.choice(actions)

    def get_abstract_mask(self, agent_idx, mask):
        return np.ones(3, dtype=np.int8)
    
    def get_observation_space(self, idx: int, abstract: bool) -> list:
        if abstract:
            return Box(0, ((self.board_length - 1) ** 2), (5,))
        else:
            return super().get_observation_space(idx, abstract)
        
    def get_action_space(self, idx: int, abstract: bool) -> list:
        if abstract:
            return Discrete(3)
        else:
            return super().get_action_space(idx, abstract)
    
    def create_env(self, render_type: str | None = None):
        self.env = DotsAndBoxesEnvironment(render_type, self.num_agents, self.board_length)
        self.env.reset()

class DotsAndBoxesEnvironment(AECEnv):
    """
    An environment based on the dots and boxes game. Takes place on a square
    board where horizontal or vertical lines can be drawn between adjacent
    co-ordinates. If a square is created upon the drawing of a line, the agent
    gains a point and gets another turn. 
    |--------------------|-----------------------------------------------|
    | Actions            | Discrete                                      |
    | Parallel API       | No                                            |
    | Manual Control     | No                                            |
    | Agents             | `agents= ['player_0', 'player_1', ...]`       |
    | Agents             | >= 2                                          |
    | Action Shape       | (1,)                                          |
    | Action Values      | [0, board_length * (board_length - 1) * 2]    |
    | Observation Shape  | (board_length * (board_length - 1) * 2)       |
    | Observation Values | [0,1]                                         |

    Observation:
    Implemented as a flat array where the first half represents vertical lines and the second half
    represents horizontal lines. If there is a line present, the observation will show a zero, if
    a line is yet to be drawn, a one. This may be counterintuitive, however, this allows the action
    mask to be a one to one copy of the observation.

    A game with board_length 3 could have an observation as such:
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
                       ^ horizontal lines begin here

    In a more understandable manner:
    [   [   [0, 1, 0]           This represents the vertical lines on the board. The first row
            [1, 0, 1]]          shows a 1 in the middle column. This indicates a vertical line can be drawn

        [   [1, 1]              This represents the horizontal lines on the board. The second
            [0, 1]              row shows a 0 in the first column. This means that a line has already
            [1, 0]  ]]          been drawn here

    Drawn as a board:           
            .  .  .             The horizontal lines span 3 rows and 2 columns, whereas the vertical
            |__.  |             lines span 2 rows and one column. This aligns with their shape above.
            .  |__.

    Actions:
    Represented as a Discrete Space with the value chosen corresponding to an index within
    the observation. This indicates where the line is to be drawn.
    
    Action_Mask:
    Identical to the observation

    Rewards:
    Each agent receives a reward of 1 for completing a square. For every square the 
    player agent ('player_0') completes, every adversary agent ('player_1', 'player_2', ...) 
    will receive a reward of -1. For every square an adversary agent completes, the player agent
    will receive a reward of -1. In this sense, the game implements both competitive and co-operative
    elements as adversaries can work together to maximise their collective reward by reducing
    the number of squares completed by the player agent, or adversaries can show elements of
    greediness by trying to maximise their personal reward and complete more squares.
    """

    metadata = {
        "name": "dots_and_boxes_v0",
        "render_modes": ["human"]
    }

    def __init__(self, render_mode=None, num_agents: int = 2, board_length: int = 5):
        self.possible_agents = ["player_" + str(i) for i in range(num_agents)]
        self.render_mode = render_mode
        self.board_length = board_length
        self.board_size = 2 * (self.board_length - 1) * self.board_length

        
        self.board = np.ones((self.board_size,), dtype=np.int8)



    def reset(self, seed = None, options = None):
        self.board = np.ones((self.board_size,), dtype=np.int8)
        self.filled_squares = np.full(((self.board_length - 1), (self.board_length - 1)), 4, dtype=np.int8)

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
        
        self.rewards = {agent: 0 for agent in self.agents}
        
        assert self.board[action] == 1, "Invalid action selected"
        self.board[action] = 0

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0

        reward_agent = 0
        midpoint = int(self.board_size/2)
        if action < midpoint:
            row = int(action/self.board_length)
            col = action % self.board_length
            if (col != 0):
                self.filled_squares[row][col - 1] -= 1
                if self.filled_squares[row][col - 1] == 0:
                    reward_agent += 1
            if (col != (self.board_length - 1)):
                self.filled_squares[row][col] -= 1
                if self.filled_squares[row][col] == 0:
                    reward_agent += 1
        else:
            row = int((action - midpoint)/(self.board_length - 1))
            col = (action - midpoint)%(self.board_length - 1)
            if (row != 0):
                self.filled_squares[row - 1][col] -= 1
                if self.filled_squares[row - 1][col] == 0:
                    reward_agent += 1
            if (row != self.board_length - 1):
                self.filled_squares[row][col] -= 1
                if self.filled_squares[row][col] == 0:
                    reward_agent += 1

        if reward_agent == 0:
            self.agent_selection = self._agent_selector.next()
        else:
            self.rewards[agent] = reward_agent
            if self.agents.index(agent) == 0:
                for agent_idx in range(1, len(self.agents)):
                    self.rewards[self.agents[agent_idx]] = -1 * reward_agent
            else:
                self.rewards[self.agents[0]] = -1 * reward_agent
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
            "observation": np.copy(self.board),
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