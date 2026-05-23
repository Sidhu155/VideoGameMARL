import sys
import random
import numpy as np
from file import writeToFile, loadFromFile, get_file_path, make_results_path
from environments.environment import Environment
from environments import tictactoe, connectfour, dotsAndBoxes
from agents.agent import Agent
from agents.tabularQAgents import QTabAgent, SARSATabAgent
from agents.funcQAgents import QFuncApproxAgent, SARSAFuncApproxAgent
from agents.randomAgent import RandomAgent
from evaluator import Evaluator
from logger import Logger

default_seed = 15
default_num_games = 1000000

def convertStringToAgent(string, **kwargs) -> Agent:
    match string:
        case "qTab":
            return QTabAgent(**kwargs)
        case "sarsaTab":
            return SARSATabAgent(**kwargs)
        case "qFunc":
            return QFuncApproxAgent(**kwargs)
        case "sarsaFunc":
            return SARSAFuncApproxAgent(**kwargs)
        case "randAgent":
            return RandomAgent(**kwargs)
        
def convertStringToEnv(string) -> Environment:
    match string:
        case "tictactoe":
            return tictactoe.TicTacToe()
        case "connectfour":
            return connectfour.ConnectFour()
        case "dotsandboxes":
            return dotsAndBoxes.DotsAndBoxes(num_agents=2, board_length=5)
        
def agent_against_agent_experiment(players: list[str], adversaries: list[str], envs: list[str], 
                abstract_player_obs: bool, abstract_player_action: bool,
                abstract_adversary_obs: bool, abstract_adversary_action: bool,
                results_dir: str, allowed_keys: list[str] | None = None):
    
    if allowed_keys is not None:
        Logger.set_allowed_keys(allowed_keys)

    for env_name in envs:
        env = convertStringToEnv(env_name)
        for player_name in players:
            env_loggers = []
            adv_loggers = []
            for adversary_name in adversaries:
                player = convertStringToAgent(player_name)
                player.set_up(env.get_action_space(0, abstract_player_action), 
                              env.get_observation_space(0, abstract_player_obs), seed=default_seed)
                adversary = convertStringToAgent(adversary_name)
                adversary.set_up(env.get_action_space(1, abstract_adversary_action), 
                                 env.get_observation_space(1, abstract_adversary_obs), seed=default_seed)
                env.runNumGames((player, adversary), default_num_games)

                env_loggers.append(env.logger)
                adv_loggers.append(adversary.logger)
                env.logger = Logger()
                env.tear_down()
                env.create_env()

            path = '/'.join((results_dir, env_name, player_name))
            env_path = make_results_path(path + '/env')
            adv_path = make_results_path(path + '/adv')
            evaluator = Evaluator(env_path, 0.05)
            evaluator.plotEnvironments(env_loggers, adversaries)
            evaluator = Evaluator(adv_path, 0.05)
            evaluator.plotAgents(adv_loggers, adversaries)

    Logger.reset_allowed_keys()
        
def main(args):
    against_random = {
        "players": ["randAgent"],
        "adversaries": ["qFunc", "qTab", "sarsaFunc", "sarsaTab", "randAgent"],
        "envs": ["tictactoe", "connectfour", "dotsandboxes"],
        "abstract_player_obs": False,
        "abstract_player_action": False,
        "abstract_adversary_obs": False,
        "abstract_adversary_action": False,
        "results_dir": "against-random"
    }

    random.seed(default_seed)
    np.random.seed(default_seed)
    agent_against_agent_experiment(**against_random)


if __name__ == '__main__':
    main(sys.argv[1:])