import pytest
import numpy as np
from tests.agents.basefunctiontest import BaseTestFuncApprox
from agents.qAgents import QFuncApproxAgent
from agents.sarsaAgents import SARSAFuncApproxAgent

class TestQTabularAgent(BaseTestFuncApprox):

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