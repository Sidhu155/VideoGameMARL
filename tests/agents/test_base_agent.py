import pytest
from agents.agent import Agent
from gymnasium.spaces import *
import numpy as np
from tests.agents.conftest import parametrize_final_reward, parametrize_learn_bool

class BaseTestAgent:
    seed_val = 155

    @pytest.fixture
    def action_space(self) -> Space:
        return Discrete(4)

    @pytest.fixture
    def agent(self) -> Agent:
        return Agent() 
    
    @pytest.fixture
    def set_up_agent(self, agent: Agent, action_space: Space) -> Agent:
        agent.set_up(action_space, seed=self.seed_val)
        return agent

    def test_init(self, agent: Agent):
        assert agent.record == []
        assert agent.learning == True
        assert agent.set_up_bool == False

    def test_set_up(self, agent: Agent, action_space: Space):
        agent.set_up(action_space)
        assert agent.action_space == action_space 

    @parametrize_final_reward
    def test_final(self, set_up_agent: Agent, rewards, expected_record):
        for val in rewards:
            set_up_agent.final(val)
        assert set_up_agent.record == expected_record

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

class TestAgent(BaseTestAgent):
    pass