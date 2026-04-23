import pytest
import numpy as np
from collections import defaultdict
from gymnasium.spaces import Space, Discrete, Box
from agents.agent import Agent
from tests.agents.conftest import parametrize_final_reward, parametrize_learn_bool

class BaseTestAgent:

    @pytest.fixture
    def observation_space(self) -> Space:
        return Box(low=0, high=1, shape=(1, 1, 2), dtype=np.int8)
    
    @pytest.fixture
    def action_space(self) -> Space:
        return Discrete(4)

    @pytest.fixture
    def agent(self) -> Agent:
        return Agent() 
    
    @pytest.fixture
    def set_up_agent(self, agent: Agent, action_space: Space, observation_space: Space) -> Agent:
        agent.set_up(action_space, observation_space, seed=self.get_seed_val())
        return agent

    def test_init(self, agent: Agent):
        assert agent.learning == True
        assert agent.set_up_bool == False
        assert type(agent.logger) == defaultdict

    def test_set_up(self, agent: Agent, action_space: Space, observation_space: Space):
        agent.set_up(action_space, observation_space, seed=self.get_seed_val())
        assert agent.action_space == action_space
        assert agent.observation_space == observation_space
        assert agent.action_space.sample() == 0

    def test_set_up_already_set_up(self, set_up_agent: Agent, action_space: Space, observation_space: Space):
        with pytest.raises(Exception) as excp:
            set_up_agent.set_up(action_space, observation_space)
        assert "already been set up" in str(excp.value)

    @parametrize_final_reward
    def test_final(self, set_up_agent: Agent, rewards: list[float], expected_record: list[float]):
        for val in rewards:
            try:
                set_up_agent.final(val)
            except ValueError as e:
                assert type(val) != float
        assert set_up_agent.logger["record"] == expected_record

    def test_final_without_set_up(self, agent: Agent):
        with pytest.raises(Exception) as excp:
            agent.final(10.0)
        assert "Agent has not been set up!" in str(excp.value)

    def test_get_action_without_set_up(self, agent: Agent):
        with pytest.raises(Exception) as excp:
            agent.get_action(np.array((0, 1, 1, 0)), np.array((0, 0, 0, 1)))
        assert "Agent has not been set up!" in str(excp.value)

    def test_update_without_set_up(self, agent: Agent):
        with pytest.raises(Exception) as excp:
            agent.update(10, np.array((0, 1, 1, 0)), 1)
        assert "Agent has not been set up!" in str(excp.value)
    
    @parametrize_learn_bool
    def test_enable_learning(self, agent: Agent, learning: bool):
        agent.learning = learning
        agent.enableLearning()
        assert agent.learning == True

    @parametrize_learn_bool
    def test_disable_learning(self, agent: Agent, learning: bool):
        agent.learning = learning
        agent.disableLearning()
        assert agent.learning == False

    def get_seed_val(self) -> int:
        return 155

class TestAgent(BaseTestAgent):
    
    def test_get_action(self, set_up_agent: Agent, action_space: Space):
        assert set_up_agent.get_action(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])) == None
        assert set_up_agent.action_space == action_space
        assert set_up_agent.learning
        assert set_up_agent.logger["record"] == []
        assert set_up_agent.set_up_bool

    def test_update(self, set_up_agent: Agent, action_space: Space):
        assert set_up_agent.update(10.0, np.array([0, 1, 1, 0]), 1) == None
        assert set_up_agent.action_space == action_space
        assert set_up_agent.learning
        assert set_up_agent.logger["record"] == []
        assert set_up_agent.set_up_bool