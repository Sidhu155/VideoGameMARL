from .agent import Agent
from utils import assert_agent_set_up

class RandomAgent(Agent):

    @assert_agent_set_up
    def get_action(self, obs, mask) -> int:
        return self.action_space.sample(mask)