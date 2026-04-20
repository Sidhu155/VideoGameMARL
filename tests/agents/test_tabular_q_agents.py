import pytest
import numpy as np
from tests.agents.basetabulartest import BaseTestTabular
from agents.tabularQAgents import QTabAgent, SARSATabAgent

class TestQTabularAgent(BaseTestTabular):

    @pytest.fixture
    def agent(self) -> QTabAgent:
        return QTabAgent(
            learning_rate=0.2,
            epsilon=0.1,
            learning_decay=0.0002,
            epsilon_decay=0.0001,
            final_learning_rate=0.02,
            final_epsilon=0.001,
            discount_factor=0.95
        )
    
    def test_update(self, set_up_agent: QTabAgent):
        prevObs = np.array((0, 1, 2, 3, 4))
        prevAction = 1
        prevUtility = 0
        reward = 10
        obs = np.array((0, 1, 3, 2, 4))
        obs_q_vals = [0.0, 1.0, -1.0, 3.0]
        action = 2

        set_up_agent.prevObs = prevObs
        set_up_agent.prevAction = prevAction
        set_up_agent.q_values[prevObs.tobytes()][prevAction] = prevUtility
        set_up_agent.q_values[obs.tobytes()] = obs_q_vals

        set_up_agent.update(reward, obs, action)
        assert set_up_agent.q_values[prevObs.tobytes()][prevAction] == 2.57
        assert np.array_equal(set_up_agent.prevObs, obs)
        assert set_up_agent.prevAction == action

class TestSARSATabularAgent(BaseTestTabular):

    @pytest.fixture
    def agent(self) -> SARSATabAgent:
        return SARSATabAgent(
            learning_rate=0.2,
            epsilon=0.1,
            learning_decay=0.0002,
            epsilon_decay=0.0001,
            final_learning_rate=0.02,
            final_epsilon=0.001,
            discount_factor=0.95
        )
    
    def test_update(self, set_up_agent: QTabAgent):
        prevObs = np.array((0, 1, 2, 3, 4))
        prevAction = 1
        prevUtility = 0
        reward = 10
        obs = np.array((0, 1, 3, 2, 4))
        obs_q_vals = [0.0, 1.0, -1.0, 3.0]
        action = 2

        set_up_agent.prevObs = prevObs
        set_up_agent.prevAction = prevAction
        set_up_agent.q_values[prevObs.tobytes()][prevAction] = prevUtility
        set_up_agent.q_values[obs.tobytes()] = obs_q_vals

        set_up_agent.update(reward, obs, action)
        assert set_up_agent.q_values[prevObs.tobytes()][prevAction] == 1.809
        assert np.array_equal(set_up_agent.prevObs, obs)
        assert set_up_agent.prevAction == action