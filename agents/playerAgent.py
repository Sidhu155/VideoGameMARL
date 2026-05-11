from .agent import Agent
from utils import assert_agent_set_up

class PlayerAgent(Agent):
    """
    Player Agent Class. Allows players to choose actions and interact directly with
    the environments through the terminal."""

    @assert_agent_set_up
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
                if action >= len(mask):
                    print("Please choose within the range of available actions")
                    action = None
                elif mask[action] == 0:
                    print("This action is invalid. Please choose another")
                    action = None
        return action