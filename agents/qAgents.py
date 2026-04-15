from .baseQAgent import Tabular, FuncApprox

class QLearnMixin:
    
    def get_next_q(self, obs, action):
        if obs is None:
            return 0
        else:
            return self.get_max_q_value(obs)
        
class QTabAgent(QLearnMixin, Tabular):
    pass
    
class QFuncApproxAgent(QLearnMixin, FuncApprox):
    pass
