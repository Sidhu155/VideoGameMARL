import numpy as np

class QLearnMixin:
    
    def get_next_q(self, obs: np.ndarray, action: int) -> float:
        """
        Args:
            obs: The current observation
            action: The action taken under the observation

        Returns:
            The maximum q_value of observation regardless of the action taken.
            If the observation is None (happens when the episode is finished)
            return 0 as the observation has no utility.
        """
        
        super().get_next_q(obs, action)
        if obs is None:
            ret_val = 0
        else:
            if self.next_max_q is not None:
                ret_val = self.next_max_q
            else:
                ret_val = self.get_max_q_value(obs)
        self.next_max_q = None
        return ret_val
        
class SARSAMixin:

    def get_next_q(self, obs: np.ndarray, action: int) -> float:
        """
        Args:
            obs: The current observation
            action: The action taken under the observation

        Returns:
            The q_value of the action under the current observation.
            If the observation is None (happens when the episode is finished)
            return 0 as the observation has no utility.
        """

        super().get_next_q(obs, action)
        if obs is None:
            ret_val = 0
        else:
            if self.next_max_q is not None:
                ret_val = self.next_max_q
            else:
                ret_val = self.get_q_value(obs, action)
        self.next_max_q = None
        return ret_val