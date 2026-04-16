import pytest
import numpy as np
from tests.agents.basefunctiontest import BaseTestFuncApprox
from agents.qAgents import QFuncApproxAgent
from agents.sarsaAgents import SARSAFuncApproxAgent

class TestQTabularAgent(BaseTestFuncApprox):

    @pytest.fixture
    def agent(self) -> QFuncApproxAgent:
        return QFuncApproxAgent(
            learning_rate=0.2,
            epsilon=0.1,
            learning_decay=0.0002,
            epsilon_decay=0.0001,
            final_learning_rate=0.02,
            final_epsilon=0.001,
            discount_factor=0.95
        )

class TestSARSAFuncApproxAgent(BaseTestFuncApprox):

    @pytest.fixture
    def agent(self) -> SARSAFuncApproxAgent:
        return SARSAFuncApproxAgent(
            learning_rate=0.2,
            epsilon=0.1,
            learning_decay=0.0002,
            epsilon_decay=0.0001,
            final_learning_rate=0.02,
            final_epsilon=0.001,
            discount_factor=0.95
        )