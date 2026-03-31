class Agent:
    def __init__(self):
        self.record = []
        self.learning = True

    def get_action(self, obs, mask) -> int:
        pass

    def update(self, reward, obs, action):
        pass

    def final(self, reward):
        self.record.append(reward)

    def set_up(self, action_space):
        self.action_space = action_space

    def enableLearning(self):
        self.learning = True

    def disableLearning(self):
        self.learning = False
