import pytest
import numpy as np
import random
from gymnasium.spaces import Space
from agents.randomAgent import RandomAgent
from tests.agents.test_base_agent import BaseTestAgent

class TestRandomAgent(BaseTestAgent):
    
    @pytest.fixture
    def agent(self) -> RandomAgent:
        return RandomAgent() 
    
    @pytest.mark.parametrize('seed, expected_action', [
        (1, 1),
        (4, 2),
        (10, 3),
        (11, 0)
    ])
    def test_get_action(self, agent: RandomAgent, action_space: Space, seed: int, expected_action: int):
        agent.set_up(action_space, seed=seed)
        obs = np.array([1, 0, 1, 0])
        mask = np.array([1, 1, 1, 1], dtype='int8')
        action = agent.get_action(obs, mask)

        assert action == expected_action

    @pytest.mark.parametrize('mask, expected_action', [
        (np.array([1, 0, 0, 0], dtype='int8'), 0),
        (np.array([0, 1, 0, 0], dtype='int8'), 1),
        (np.array([0, 0, 1, 0], dtype='int8'), 2),
        (np.array([0, 0, 0, 1], dtype='int8'), 3),
        (np.array([1, 1, 1, 0], dtype='int8'), 0),
        (np.array([1, 0, 1, 1], dtype='int8'), 0),
        (np.array([0, 1, 1, 0], dtype='int8'), 1),
        (np.array([0, 1, 0, 1], dtype='int8'), 1),
        (np.array([1, 1, 1, 1], dtype='int8'), 0)
    ])
    def test_get_action_masked(self, set_up_agent: RandomAgent, mask: np.ndarray, expected_action: int):
        obs = np.array([1, 0, 1, 0])
        action = set_up_agent.get_action(obs, mask)

        assert action == expected_action
