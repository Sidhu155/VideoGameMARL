import pytest
from tests.agents.test_base_agent import BaseTestAgent
from agents.qAgents import QTabAgent
from gymnasium.spaces import Space
import numpy as np
import random
from collections import defaultdict
from tests.agents.conftest import (
    parametrize_final_reward, parametrize_get_action, parametrize_epsilon
)

class TestQTabularAgent(BaseTestAgent):
    seed_val = 155

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

    def test_init(self, agent: QTabAgent):
        super().test_init(agent)
        assert agent.prevAction is None
        assert agent.prevObs is None
        assert agent.training_error == []

    def test_set_up(self, agent: QTabAgent, action_space: Space):
        assert not hasattr(agent, 'q_values')
        assert not hasattr(agent, 'numActions')
        super().test_set_up(agent, action_space)
        assert hasattr(agent, 'q_values')
        assert hasattr(agent, 'numActions')
        assert type(agent.q_values) == defaultdict
        assert agent.numActions == action_space.n

    @parametrize_get_action
    @parametrize_epsilon
    def test_get_action(self, set_up_agent: QTabAgent, q_vals, mask, epsilon, expected_max, expected_rand):
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        
        obs = np.array((0, 1, 3, 2, 4))
        set_up_agent.epsilon = epsilon
        set_up_agent.q_values[obs.tobytes()] = q_vals
        
        action = set_up_agent.get_action(obs, mask)
        if epsilon < 0.7:
            assert action == expected_max
        else:
            assert action == expected_rand

    @parametrize_get_action
    @parametrize_epsilon
    def test_get_action_learning_disabled(self, set_up_agent: QTabAgent, q_vals, mask, expected_max, expected_rand, epsilon):
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        obs = np.array((0, 1, 3, 2, 4))
        set_up_agent.disableLearning()
        set_up_agent.epsilon = epsilon
        set_up_agent.q_values[obs.tobytes()] = q_vals
        action = set_up_agent.get_action(obs, mask)
        assert action == expected_max

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

    def test_update_learning_disabled(self, set_up_agent: QTabAgent):
        set_up_agent.disableLearning()
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
        assert set_up_agent.q_values[prevObs.tobytes()][prevAction] == 0
        assert np.array_equal(set_up_agent.prevObs, obs)
        assert set_up_agent.prevAction == action

    def test_update_none_prev_act_obs(self, set_up_agent: QTabAgent):
        reward = 10
        obs = np.array((0, 1, 3, 2, 4))
        obs_q_vals = [0.0, 1.0, -1.0, 3.0]
        action = 2

        set_up_agent.q_values[obs.tobytes()] = obs_q_vals
        prev_q_values = {obs.tobytes(): obs_q_vals}

        set_up_agent.update(reward, obs, action)
        assert set_up_agent.q_values == prev_q_values
        assert np.array_equal(set_up_agent.prevObs, obs)
        assert set_up_agent.prevAction == action

    def test_update_none_curr_obs_act(self, set_up_agent: QTabAgent):
        prevObs = np.array((0, 1, 2, 3, 4))
        prevAction = 1
        prevUtility = 0
        reward = 10

        set_up_agent.prevObs = prevObs
        set_up_agent.prevAction = prevAction
        set_up_agent.q_values[prevObs.tobytes()][prevAction] = prevUtility

        set_up_agent.update(reward, None, None)
        assert set_up_agent.q_values[prevObs.tobytes()][prevAction] == 2
        assert set_up_agent.prevObs == None
        assert set_up_agent.prevAction == None

    @parametrize_final_reward
    def test_final(self, set_up_agent: QTabAgent, rewards, expected_record):
        super().test_final(set_up_agent, rewards, expected_record)
        assert pytest.approx(set_up_agent.epsilon) == 0.1 - (len(rewards) * 0.0001)
        assert pytest.approx(set_up_agent.lr) == 0.2 - (len(rewards) * 0.0002)
        
    def test_get_q_val(self, set_up_agent: QTabAgent):
        pass

    def test_get_max_q_val(self):
        pass

    def test_update_q_val(self):
        pass

    def test_get_default_vals(self, set_up_agent):
        default_vals = set_up_agent.getDefaultVals()
        assert np.array_equal(default_vals, np.array([0, 0, 0, 0]))
        assert default_vals.dtype == np.float16

    def test_decay(self, set_up_agent):
        pass