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
            return 0
        else:
            return self.get_max_q_value(obs)
        
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
            return 0
        else:
            return self.get_q_value(obs, action)