from .baseQAgent import Tabular, FuncApprox
import numpy as np

class SARSAMixin:

    def get_next_q(self, obs: np.ndarray, action: int) -> float:
        super().get_next_q(obs, action)
        if obs is None:
            return 0
        else:
            return self.get_q_value(obs, action)
        
class SARSATabAgent(SARSAMixin, Tabular):
    pass

class SARSAFuncApproxAgent(SARSAMixin, FuncApprox):
    pass