import pytest
import numpy as np
import random
from gymnasium.spaces import Space, Box
from tests.agents.test_base_agent import BaseTestAgent
from agents.baseQAgent import FuncApprox

class BaseTestFuncApprox(BaseTestAgent):

    @pytest.fixture
    def observation_space(self) -> Space:
        return Box(low=0, high=1, shape=(1, 1, 2), dtype=np.int8)
    
    @pytest.fixture
    def set_up_agent(self, agent: FuncApprox, action_space: Space, observation_space: Space):
        agent.set_up(action_space, observation_space, seed=self.get_seed_val())
        return agent
    
    def test_init(self, agent: FuncApprox):
        super().test_init(agent)
        assert agent.prevAction is None
        assert agent.prevObs is None
        assert agent.training_error == []

    def test_set_up(self, agent: FuncApprox, action_space: Space, observation_space:Space):
        assert not hasattr(agent, 'q_function')
        assert not hasattr(agent, 'numActions')
        assert not hasattr(agent, 'numFeatures')

        agent.set_up(action_space, observation_space, seed=self.get_seed_val())
        assert agent.action_space == action_space 

        assert hasattr(agent, 'q_function')
        assert hasattr(agent, 'numActions')
        assert hasattr(agent, 'numFeatures')

        assert agent.numActions == 4
        assert agent.numFeatures == 2

    def test_get_q_val(self, set_up_agent: FuncApprox):
        obs = np.array([[[1, 2]]])
        set_up_agent.q_function = np.array([0, 1, 3, 0, 2.1, 4.2, 1.1, 3, 2])
        assert set_up_agent.get_q_value(obs, 0) == 4
        assert set_up_agent.get_q_value(obs, 1) == 5
        assert set_up_agent.get_q_value(obs, 2) == 12.5
        assert set_up_agent.get_q_value(obs, 3) == 9.1

    def test_get__max_q_val(self, set_up_agent: FuncApprox):
        obs = np.array([[[1, 2]]])
        set_up_agent.q_function = np.array([0, 1, 3, 0, 2.1, 4.2, 1.1, 3, 2])
        assert set_up_agent.get_max_q_value(obs) == 12.5
    
    def test_update_q_val(self, set_up_agent: FuncApprox):
        obs = np.array([[[1, 2]]])
        set_up_agent.prevObs = obs
        set_up_agent.prevAction = 2
        set_up_agent.q_function = np.array([0, 1, 3, 0, 2.1, 4.2, 1.1, 3, 2])
        set_up_agent.update_q_value(12.5, -10)
        assert np.array_equal(set_up_agent.q_function[:4], np.array([0, 1, 3, 0]))
        assert pytest.approx(set_up_agent.q_function[4]) == 2.1 + (set_up_agent.lr * -10) * 1
        assert pytest.approx(set_up_agent.q_function[5]) == 4.2 + (set_up_agent.lr * -10) * 2
        assert np.array_equal(set_up_agent.q_function[6:8], np.array([1.1, 3]))
        assert set_up_agent.q_function[8] == 2 + (set_up_agent.lr * -10)

    def test_obs_to_vector(self, set_up_agent: FuncApprox):
        pass

    def test_get_default_func(self, set_up_agent: FuncApprox):
        pass