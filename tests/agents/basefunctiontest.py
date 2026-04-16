import pytest
import numpy as np
import random
from gymnasium.spaces import Space, Box
from tests.agents.test_base_agent import BaseTestAgent
from agents.baseQAgent import FuncApprox

class BaseTestFuncApprox(BaseTestAgent):

    @pytest.fixture
    def observation_space(self) -> Space:
        return Box(low=0, high=1, shape=(2, 3, 2), dtype=np.int8)
    
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

        assert agent.numActions == action_space.n
        assert agent.numFeatures == np.prod(observation_space.shape)

    def test_get_q_val(self):
        pass