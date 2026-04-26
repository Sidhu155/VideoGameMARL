import pytest
import numpy as np
import random
from collections import defaultdict
from gymnasium.spaces import Space
from tests.agents.test_base_agent import BaseTestAgent
from agents.tabularQAgents import Tabular, QTabAgent, SARSATabAgent
from tests.agents.conftest import (
    parametrize_final_reward, parametrize_get_action, parametrize_epsilon,
    parametrize_q_table, parametrize_seed_expected_max_action
)

class BaseTestTabular(BaseTestAgent):

    def test_init(self, agent: Tabular):
        super().test_init(agent)
        assert agent.prevAction is None
        assert agent.prevObs is None
        assert agent.logger.getLogs("training_error") == []

    def test_set_up(self, agent: Tabular, action_space: Space, observation_space: Space):
        assert not hasattr(agent, 'q_values')
        assert not hasattr(agent, 'numActions')
        super().test_set_up(agent, action_space, observation_space)
        assert hasattr(agent, 'q_values')
        assert hasattr(agent, 'numActions')
        assert type(agent.q_values) == defaultdict
        assert agent.numActions == action_space.n

    @parametrize_get_action
    @parametrize_epsilon
    def test_get_action(self, set_up_agent: Tabular,
                        mask: np.ndarray, epsilon: float,
                        expected_max: int, expected_rand: int):
        random.seed(self.get_seed_val())
        np.random.seed(self.get_seed_val())
        obs = np.array((0, 1, 3, 2, 4))
        set_up_agent.q_values[obs.tobytes()] = [0.0, 1.0, -1.0, 3.0]

        set_up_agent.epsilon = epsilon
        
        action = set_up_agent.get_action(obs, mask)
        if epsilon < 0.7:
            assert action == expected_max
        else:
            assert action == expected_rand

    @parametrize_get_action
    @parametrize_epsilon
    def test_get_action_learning_disabled(self, set_up_agent: Tabular, mask, expected_max, expected_rand, epsilon):
        random.seed(self.get_seed_val())
        np.random.seed(self.get_seed_val())
        obs = np.array((0, 1, 3, 2, 4))
        set_up_agent.q_values[obs.tobytes()] = [0.0, 1.0, -1.0, 3.0]

        set_up_agent.disableLearning()
        set_up_agent.epsilon = epsilon

        action = set_up_agent.get_action(obs, mask)
        assert action == expected_max

    @parametrize_seed_expected_max_action
    def test_get_action_multiple_max_actions(self, set_up_agent: Tabular, seed: int, expected_action: int):
        random.seed(seed)
        np.random.seed(seed)
        obs = np.array((0, 1, 3, 2, 4))
        mask = np.array([1, 1, 1, 1])
        set_up_agent.q_values[obs.tobytes()] = [1.0, 1.0, 1.0, 1.0]
        set_up_agent.disableLearning()
        
        action = set_up_agent.get_action(obs, mask)
        assert action == expected_action

    def test_update_learning_disabled(self, set_up_agent: Tabular):
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

    def test_update_none_prev_act_obs(self, set_up_agent: Tabular):
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

    def test_update_none_curr_obs_act(self, set_up_agent: Tabular):
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
    def test_final(self, set_up_agent: Tabular, rewards, expected_record):
        super().test_final(set_up_agent, rewards, expected_record)
        assert pytest.approx(set_up_agent.epsilon) == 0.1 - (len(expected_record) * 0.0001)
        assert pytest.approx(set_up_agent.lr) == 0.2 - (len(expected_record) * 0.0002)

    @parametrize_q_table   
    def test_get_q_val(self, set_up_agent: Tabular, obs, utilities, action, action_util, max_util, new_util):
        set_up_agent.q_values[obs.tobytes()] = utilities
        assert set_up_agent.get_q_value(obs, action) == action_util

    def test_get_q_val_without_set_up(self, agent: Tabular):
        with pytest.raises(Exception) as excp:
            agent.get_q_value(np.array((0, 1, 1, 0)), 2)
        assert "Agent has not been set up!" in str(excp.value)

    @parametrize_q_table
    def test_max_q_val(self, set_up_agent: Tabular, obs, utilities, action, action_util, max_util, new_util):
        set_up_agent.q_values[obs.tobytes()] = utilities
        assert set_up_agent.get_max_q_value(obs) == max_util

    def test_get_max_q_val_without_set_up(self, agent: Tabular):
        with pytest.raises(Exception) as excp:
            agent.get_max_q_value(np.array((0, 1, 1, 0)))
        assert "Agent has not been set up!" in str(excp.value)

    @parametrize_q_table
    def test_update_q_val(self, set_up_agent: Tabular, obs, utilities, action, action_util, max_util, new_util):
        set_up_agent.prevObs = obs
        set_up_agent.prevAction = action
        set_up_agent.q_values[obs.tobytes()] == utilities
        set_up_agent.update_q_value(utilities[action], (new_util - utilities[action])/set_up_agent.lr)
        assert set_up_agent.q_values[obs.tobytes()][action] == new_util

    def test_update_q_val_without_set_up(self, agent: Tabular):
        with pytest.raises(Exception) as excp:
            agent.update_q_value(10, 20)
        assert "Agent has not been set up!" in str(excp.value)

    def test_get_default_vals(self, set_up_agent: Tabular):
        default_vals = set_up_agent.getDefaultVals()
        assert np.array_equal(default_vals, np.array([0, 0, 0, 0]))
        assert default_vals.dtype == np.float16

    def test_decay(self, agent: Tabular):
        agent.decay()
        assert pytest.approx(agent.epsilon) == 0.1 - 0.0001
        assert pytest.approx(agent.lr) == 0.2 - 0.0002
    
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