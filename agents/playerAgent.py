from .agent import Agent

class PlayerAgent(Agent):
    def __init__(self):
        super().__init__()
    
    def get_action(self, obs, mask):
        print("Possible Actions")
        possible_actions = []
        for i in range(len(mask)):
            if mask[i] == 1:
                possible_actions.append(i)
        
        print(*possible_actions, sep=", ")
        
        action = None
        while action is None:
            player_input = input("Please input a number:  ")
            try: 
                action = int(player_input)
            except Exception as e:
                print(f"An exception was raised: {e}")
                action = None
            else:
                if mask[action] == 0:
                    print("This action is invalid. Please choose another")
                    action = None
        return action
    
    def set_up(self, action_space):
        super().set_up(action_space)
