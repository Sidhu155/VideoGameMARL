import pytest
from agents.agent import Agent
from gymnasium.spaces import *
import random
from tests.agents.conftest import parametrize_final_reward, parametrize_learn_bool

class BaseTestAgent:

    @pytest.fixture
    def action_space(self) -> Space:
        return Discrete(4)

    @pytest.fixture
    def agent(self) -> Agent:
        return Agent() 

    def test_init(self, agent: Agent):
        assert agent.record == []
        assert agent.learning == True

    def test_set_up(self, agent: Agent, action_space: Space):
        agent.set_up(action_space)
        assert agent.action_space == action_space 

    @parametrize_final_reward
    def test_final(self, agent: Agent, rewards, expected_record):
        for val in rewards:
            agent.final(val)
        assert agent.record == expected_record

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