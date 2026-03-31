from .agent import Agent

class RandomAgent(Agent):
    def get_action(self, obs, mask) -> int:
        return self.action_space.sample(mask)