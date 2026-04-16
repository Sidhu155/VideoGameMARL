from .baseQAgent import Tabular, FuncApprox
import numpy as np

class QLearnMixin:
    
    def get_next_q(self, obs: np.ndarray, action: int) -> float:
        if obs is None:
            return 0
        else:
            return self.get_max_q_value(obs)
        
class QTabAgent(QLearnMixin, Tabular):
    pass
    
class QFuncApproxAgent(QLearnMixin, FuncApprox):
    pass
