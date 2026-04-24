from .agent import Agent
from utils import assert_agent_set_up, time_func

class RandomAgent(Agent):

    @time_func("get_action")
    @assert_agent_set_up
    def get_action(self, obs, mask) -> int:
        return self.action_space.sample(mask)