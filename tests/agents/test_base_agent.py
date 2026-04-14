import pytest
from agents.agent import Agent
from gymnasium.spaces import *

@pytest.fixture
def action_space() -> Space:
    return Discrete(4)

@pytest.fixture
def agent() -> Agent:
    return Agent()

class TestBaseAgent:

    def test_init(self, agent: Agent):
        assert agent.record == []
        assert agent.learning == True

    def test_set_up(self, agent: Agent, action_space: Space):
        agent.set_up(action_space)
        assert agent.action_space == action_space 

    def test_get_action(self, agent: Agent):
        assert agent.get_action(None, None) == None

    def test_update(self, agent: Agent):
        assert agent.update(None, None, None) == None
    
    @pytest.mark.parametrize('rewards, expected_record', [
        ([0, 1, 0, 1, 0], [0, 1, 0, 1, 0]),
        (['h'], []),
        (["hello"], []),
        ([1.0, 3.5, 3, 2.4], [1.0, 3.5, 3, 2.4]),
        ([1.0, "h", 3.4, "s"], [1.0, 3.4])
    ])
    def test_final(self, agent: Agent, rewards, expected_record):
        for val in rewards:
            agent.final(val)
        assert agent.record == expected_record

    parametrize_bool = pytest.mark.parametrize('learning', [True, False]) 

    @parametrize_bool
    def test_enable_learning(self, agent: Agent, learning: bool):
        agent.learning = learning
        agent.enableLearning()
        assert agent.learning == True

    @parametrize_bool
    def test_disable_learning(self, agent: Agent, learning: bool):
        agent.learning = learning
        agent.disableLearning()
        assert agent.learning == False
