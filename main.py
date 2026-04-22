import os
import sys
import torch
from argparse import ArgumentParser
from file import writeToFile, loadFromFile
from environments.environment import Environment
from environments import connectfour, tictactoe
from agents.agent import Agent
from agents.tabularQAgents import QTabAgent, SARSATabAgent
from agents.funcQAgents import QFuncApproxAgent, SARSAFuncApproxAgent
from agents.randomAgent import RandomAgent
from agents.playerAgent import PlayerAgent

def parse(args: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("environment", type=str)
    parser.add_argument("-p", "--player", dest="playerAgent")
    parser.add_argument("-a", "--adversary", dest="adversaryAgent")
    parser.add_argument("-n", "--numtrain", dest="numTrain", type=int, default=1000)
    parser.add_argument("-w", "--numwatch", dest="numWatch", type=int, default=0)
    parser.add_argument("-x", "--numplay", dest="numPlay", type=int, default=0)
    parser.add_argument("-s:a", "--save-adversary", action="store_true")
    parser.add_argument("-s:p", "--save-player", action="store_true")
    parser.add_argument("-s:e", "--save-env", action="store_true")
    parser.add_argument("-t:w", "--train-watch", action="store_true")
    parser.add_argument("-t:p", "--train-play", action="store_true")
    parser.add_argument("-d:p", "--disable-player-learn", action="store_true")
    parser.add_argument("-d:a", "--disable-adversary-learn", action="store_true")
    return parser.parse_args(args)

def match_args(args):
    loaded_player = False
    loaded_adversary = False

    match args.environment:
        case "connectfour":
            environment = connectfour.ConnectFour()
        case "tictactoe":
            environment = tictactoe.TicTacToe()
        case _:
            try:
                environment: Environment = loadFromFile(args.environment, 'e')
            except Exception as excp:
                print(type(excp))
                print(excp)
                raise Exception("Please input a valid environment or filepath")
            else:
                environment.create_env()
    action_space = environment.get_action_spaces()
    observation_space = environment.get_observation_spaces()

    match args.playerAgent:
        case "qTab":
            player = QTabAgent()
        case "sarsaTab":
            player = SARSATabAgent()
        case "qFunc":
            player = QFuncApproxAgent()
        case "sarsaFunc":
            player = SARSAFuncApproxAgent()
        case "playerAgent":
            player = PlayerAgent()
        case "randAgent" | None:
            player = RandomAgent()
        case _:
            try:
                player: Agent = loadFromFile(args.playerAgent, 'p')
                if player.observation_space != observation_space[0]:
                    raise ValueError("Loaded player agent's observation space does not match environment")
                if player.action_space != action_space[0]:
                    raise ValueError("Loaded player agent's action space does not match environment")
            except Exception as excp:
                print(type(excp))
                print(excp)
                raise Exception("Please input a valid player agent or filepath")
            else:
                loaded_player = True
    
    if not loaded_player:
        player.set_up(action_space[0], observation_space[0])
    if args.disable_player_learn:
        player.disableLearning()

    match args.adversaryAgent:
        case "qTab":
            adversary = QTabAgent()
        case "sarsaTab":
            adversary = SARSATabAgent()
        case "qFunc":
            adversary = QFuncApproxAgent()
        case "sarsaFunc":
            adversary = SARSAFuncApproxAgent()
        case "randAgent" | None:
            adversary = RandomAgent()
        case _:
            try:
                adversary: Agent = loadFromFile(args.adversaryAgent, 'a')
                if adversary.observation_space != observation_space[1]:
                    raise ValueError("Loaded adversary agent's observation space does not match environment")
                if adversary.action_space != action_space[1]:
                    raise ValueError("Loaded adversary agent's action space does not match environment")
            except Exception as excp:
                print(type(excp))
                print(excp)
                raise Exception("Please input a valid adversary agent or filepath")
            else:
                loaded_adversary = True
    if not loaded_adversary:
        adversary.set_up(action_space[1], observation_space[1])
    if args.disable_adversary_learn:
        adversary.disableLearning()

    if args.numTrain < 0:
        raise Exception("Number of games for training cannot be negative")
    
    if args.numWatch < 0:
        raise Exception("Number of games for watching cannot be negative")
    
    if args.numPlay < 0:
        raise Exception("Number of games for play cannot be negative")

    return (environment, player, adversary, args.numTrain, args.numWatch, args.numPlay,
            args.save_player, args.save_adversary, args.save_env, args.train_watch, args.train_play)

def main(args: list[str] | None =  None):
    torch.set_default_device('mps')
    (environment, player, adversary, numTrain, numWatch, numPlay,
    save_player, save_adversary, save_env, train_watch, train_play) = match_args(parse(args))

    print("Training...")
    environment.runNumGames((player, adversary), numTrain)

    environment.enable_rendering()
    if not train_watch:
        player.disableLearning()
        adversary.disableLearning()
    print("Watching...")
    environment.runNumGames((player, adversary), numWatch)

    if save_player: writeToFile(player, 'p') 
    
    temp_action_space = player.action_space
    player = PlayerAgent()
    player.set_up(temp_action_space)

    if not train_play:
        adversary.disableLearning()
    else:
        adversary.enableLearning()
    print("Playing...")
    environment.runNumGames((player, adversary), numPlay)
    if save_adversary: writeToFile(adversary, 'a') 

    environment.tear_down()
    if save_env: writeToFile(environment, 'e') 


if __name__ == "__main__":
    main(sys.argv[1:])