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

def convertStringToAgent(string, **kwargs) -> Agent:
    match string:
        case "qtab":
            return QTabAgent(**kwargs)
        case "sarsatab":
            return SARSATabAgent(**kwargs)
        case "qfunc":
            return QFuncApproxAgent(**kwargs)
        case "sarsafunc":
            return SARSAFuncApproxAgent(**kwargs)
        
def convertStringToEnv(string) -> Environment:
    match string:
        case "tictactoe":
            return tictactoe.TicTacToe()
        case "connectfour":
            return connectfour.ConnectFour()
        case "dotsandboxes":
            return dotsAndBoxes.DotsAndBoxes(num_agents=2, board_length=5)
        
def optimisationExperiments():
    
    agents: list[(Agent, Agent)] = [
        (QTabAgent(), QTabAgent()),
        (SARSATabAgent(), SARSATabAgent()),
        (QFuncApproxAgent(), QFuncApproxAgent()),
        (SARSAFuncApproxAgent(), SARSAFuncApproxAgent())
    ]
    names = ['qtab', 'sarsatab', 'qfunc', 'sarsafunc']
    env_loggers = []
    pla_loggers = []
    adv_loggers = []

    for player, adversary in agents:
        env = dotsAndBoxes.DotsAndBoxes(2, 5)
        player.set_up(env.get_action_space(0, False), env.get_observation_space(0, False), seed=15)
        adversary.set_up(env.get_action_space(1, False), env.get_observation_space(1, False), seed=15)
        env.runNumGames((player, adversary), 10000)

        env_loggers.append(env.logger)
        pla_loggers.append(player.logger)
        adv_loggers.append(adversary.logger)


    evaluator = Evaluator(make_results_path('optimenv'), 0.05)
    evaluator.plotEnvironments(env_loggers, names)
    evaluator = Evaluator(make_results_path('optimpla'), 0.05)
    evaluator.plotAgents(pla_loggers, names)
    evaluator = Evaluator(make_results_path('optimadv'), 0.05)
    evaluator.plotAgents(adv_loggers, names)

def AgentsAgainstRandomExperiment():
    agents = ['qtab', 'sarsatab', 'qfunc', 'sarsafunc']
    envs = ['tictactoe', 'connectfour', 'dotsandboxes']

    for env_name in envs:
        env_loggers = []
        adv_loggers = []
        env = convertStringToEnv(env_name)
        player = RandomAgent()
        player.logger.disableLogging()
        player.set_up(env.get_action_space(0, False), env.get_observation_space(0, False), seed=15)
        for adversary_name in agents:
            adversary = convertStringToAgent(adversary_name)
            adversary.set_up(env.get_action_space(1, False), env.get_observation_space(1, False), seed=15)
            env.runNumGames((player, adversary), 1000000)

            env_loggers.append(env.logger)
            adv_loggers.append(adversary.logger)
            env.logger = Logger()
            env.tear_down()
            env.create_env()
        
        env_path = make_results_path('against-random/' + env_name + '/env')
        adv_path = make_results_path('against-random/' + env_name + '/adv')
        evaluator = Evaluator(env_path, 0.05)
        evaluator.plotEnvironments(env_loggers, agents)
        evaluator = Evaluator(adv_path, 0.05)
        evaluator.plotAgents(adv_loggers, agents)

def AgentsAgainstAgentsExperiment():
    agents = ['qtab', 'sarsatab', 'qfunc', 'sarsafunc']
    envs = ['tictactoe', 'connectfour', 'dotsandboxes']

    for env_name in envs:
        env = convertStringToEnv(env_name)
        for player_name in agents:
            env_loggers = []
            adv_loggers = []
            for adversary_name in agents:
                player = convertStringToAgent(player_name)
                player.set_up(env.get_action_space(0, False), env.get_observation_space(0, False), seed=15)
                adversary = convertStringToAgent(adversary_name)
                adversary.set_up(env.get_action_space(1, False), env.get_observation_space(1, False), seed=15)
                env.runNumGames((player, adversary), 1000000)

                env_loggers.append(env.logger)
                adv_loggers.append(adversary.logger)
                env.logger = Logger()
                env.tear_down()
                env.create_env()

            path = '/'.join(('against-agents', env_name, player_name))
            env_path = make_results_path(path + '/env')
            adv_path = make_results_path(path + '/adv')
            evaluator = Evaluator(env_path, 0.05)
            evaluator.plotEnvironments(env_loggers, agents)
            evaluator = Evaluator(adv_path, 0.05)
            evaluator.plotAgents(adv_loggers, agents)

def AbstractAgentsAgainstRandomExperiment():
    agents = ['qtab', 'sarsatab', 'qfunc', 'sarsafunc']
    envs = ['dotsandboxes']

    for env_name in envs:
        env_loggers = []
        adv_loggers = []
        env = convertStringToEnv(env_name)
        player = RandomAgent()
        player.logger.disableLogging()
        player.set_up(env.get_action_space(0, False), env.get_observation_space(0, False), seed=15)
        for adversary_name in agents:
            adversary = convertStringToAgent(adversary_name, action_abstraction=True, obs_abstraction=True)
            adversary.set_up(env.get_action_space(1, False), env.get_observation_space(1, False), seed=15)
            env.runNumGames((player, adversary), 1000000)

            env_loggers.append(env.logger)
            adv_loggers.append(adversary.logger)
            env.logger = Logger()
            env.tear_down()
            env.create_env()
        
        env_path = make_results_path('abstract-against-random/' + env_name + '/env')
        adv_path = make_results_path('abstract-against-random-abstract/' + env_name + '/adv')
        evaluator = Evaluator(env_path, 0.05)
        evaluator.plotEnvironments(env_loggers, agents)
        evaluator = Evaluator(adv_path, 0.05)
        evaluator.plotAgents(adv_loggers, agents)

def AbstractAgentsAgainstAgentsExperiment():
    agents = ['qtab', 'sarsatab', 'qfunc', 'sarsafunc']
    envs = ['dotsandboxes']

    for env_name in envs:
        env = convertStringToEnv(env_name)
        for player_name in agents:
            env_loggers = []
            adv_loggers = []
            for adversary_name in agents:
                player = convertStringToAgent(player_name, action_abstraction=True, obs_abstraction=True)
                player.set_up(env.get_action_space(0, False), env.get_observation_space(0, False), seed=15)
                adversary = convertStringToAgent(adversary_name, action_abstraction=True, obs_abstraction=True)
                adversary.set_up(env.get_action_space(1, False), env.get_observation_space(1, False), seed=15)
                env.runNumGames((player, adversary), 1000000)

                env_loggers.append(env.logger)
                adv_loggers.append(adversary.logger)
                env.logger = Logger()
                env.tear_down()
                env.create_env()

            path = '/'.join(('abstract-against-agents', env_name, player_name))
            env_path = make_results_path(path + '/env')
            adv_path = make_results_path(path + '/adv')
            evaluator = Evaluator(env_path, 0.05)
            evaluator.plotEnvironments(env_loggers, agents)
            evaluator = Evaluator(adv_path, 0.05)
            evaluator.plotAgents(adv_loggers, agents)
        
def main(args):
    random.seed(15)
    np.random.seed(15)
    AgentsAgainstRandomExperiment()

    random.seed(15)
    np.random.seed(15)
    AgentsAgainstAgentsExperiment()

    random.seed(15)
    np.random.seed(15)
    AbstractAgentsAgainstRandomExperiment()

    random.seed(15)
    np.random.seed(15)
    AbstractAgentsAgainstAgentsExperiment()

if __name__ == '__main__':
    main(sys.argv[1:])