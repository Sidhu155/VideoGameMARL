import pytest
import numpy as np
import random
from gymnasium.spaces import Space
from tests.agents.test_base_agent import BaseTestAgent
from agents.funcQAgents import FuncApprox, QFuncApproxAgent, SARSAFuncApproxAgent
from tests.agents.conftest import parametrize_final_reward, parametrize_get_action, parametrize_epsilon

class BaseTestFuncApprox(BaseTestAgent):
    
    def test_init(self, agent: FuncApprox):
        super().test_init(agent)
        assert agent.prevAction is None
        assert agent.prevObs is None
        assert agent.logger["training_error"] == []

    def test_set_up(self, agent: FuncApprox, action_space: Space, observation_space:Space):
        assert not hasattr(agent, 'q_function')
        assert not hasattr(agent, 'numActions')
        assert not hasattr(agent, 'numFeatures')

        super().test_set_up(agent, action_space, observation_space)
        
        assert hasattr(agent, 'q_function')
        assert hasattr(agent, 'numActions')
        assert hasattr(agent, 'numFeatures')

        assert agent.numActions == 4
        assert agent.numFeatures == 2

    @parametrize_epsilon
    @parametrize_get_action
    def test_get_action(self, set_up_agent: FuncApprox,
                        mask: np.ndarray, epsilon: float,
                        expected_max: int, expected_rand: int):
        random.seed(self.get_seed_val())
        np.random.seed(self.get_seed_val())
        obs = np.array([[[1, 2]]])
        set_up_agent.q_function = np.array([0, 1, 3, 0, -2.1, -4.2, 1.1, 3, 2])

        set_up_agent.epsilon = epsilon
        
        action = set_up_agent.get_action(obs, mask)
        if epsilon < 0.7:
            assert action == expected_max
        else:
            assert action == expected_rand

    @parametrize_epsilon
    @parametrize_get_action
    def test_get_action_learning_disabled(self, set_up_agent: FuncApprox, mask, expected_max, expected_rand, epsilon):
        random.seed(self.get_seed_val())
        np.random.seed(self.get_seed_val())
        obs = np.array([[[1, 2]]])
        set_up_agent.q_function = np.array([0, 1, 3, 0, -2.1, -4.2, 1.1, 3, 2])

        set_up_agent.disableLearning()
        set_up_agent.epsilon = epsilon

        action = set_up_agent.get_action(obs, mask)
        assert action == expected_max

    @parametrize_final_reward
    def test_final(self, set_up_agent: FuncApprox, rewards, expected_record):
        super().test_final(set_up_agent, rewards, expected_record)
        assert pytest.approx(set_up_agent.epsilon) == 1e-1 - (len(expected_record) * 1e-6)
        assert pytest.approx(set_up_agent.lr) == 1e-3 - (len(expected_record) * 1e-8)

    def test_get_q_val(self, set_up_agent: FuncApprox):
        obs = np.array([[[1, 2]]])
        set_up_agent.q_function = np.array([0, 1, 3, 0, 2.1, 4.2, 1.1, 3, 2])
        assert set_up_agent.get_q_value(obs, 0) == 4
        assert set_up_agent.get_q_value(obs, 1) == 5
        assert set_up_agent.get_q_value(obs, 2) == 12.5
        assert set_up_agent.get_q_value(obs, 3) == 9.1

    def test_get_max_q_val(self, set_up_agent: FuncApprox):
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
        obs = np.array([[[1, 2]]])
        vector = set_up_agent.obs_to_feature_vector(obs, 2)
        
        assert np.array_equal(vector, np.array([0, 0, 0, 0, 1, 2, 0, 0, 1]))

    def test_obs_to_vector_diff_obs_size(self, set_up_agent: FuncApprox):
        obs = np.array([[[1, 2, 3]]])
        with pytest.raises(ValueError, match="Observation has size 3 but size 2 is required!"):
            vector = set_up_agent.obs_to_feature_vector(obs, 2)

    def test_default_func(self, agent: FuncApprox):
        agent.numFeatures = 3
        agent.numActions = 2
        func = agent.getDefaultFunc()

        assert np.array_equal(func, np.zeros(7))

    def test_decay(self, agent: FuncApprox):
        agent.decay()
        assert pytest.approx(agent.epsilon) == 1e-1 - 1e-6
        assert pytest.approx(agent.lr) == 1e-3 - 1e-8

class TestQFuncApproxAgent(BaseTestFuncApprox):

    @pytest.fixture
    def agent(self) -> QFuncApproxAgent:
        return QFuncApproxAgent(
            learning_rate = 1e-3,
            epsilon = 1e-1,
            learning_decay = 1e-8,
            epsilon_decay = 1e-6,
            final_learning_rate = 1e-4,
            final_epsilon = 1e-2,
            discount_factor = 0.95
        )

class TestSARSAFuncApproxAgent(BaseTestFuncApprox):

    @pytest.fixture
    def agent(self) -> SARSAFuncApproxAgent:
        return SARSAFuncApproxAgent(
            learning_rate = 1e-3,
            epsilon = 1e-1,
            learning_decay = 1e-8,
            epsilon_decay = 1e-6,
            final_learning_rate = 1e-4,
            final_epsilon = 1e-2,
            discount_factor = 0.95
        )