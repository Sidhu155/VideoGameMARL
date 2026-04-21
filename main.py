import os
import sys
import torch
import dill as pickle
from pathlib import Path
from argparse import ArgumentParser
from environments import connectfour, tictactoe
from agents.tabularQAgents import QTabAgent, SARSATabAgent
from agents.funcQAgents import QFuncApproxAgent, SARSAFuncApproxAgent
from agents.randomAgent import RandomAgent
from agents.playerAgent import PlayerAgent
from evaluator import Evaluator

path_objects = "saved_objects"

def writeToFile(object, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(object, outp)

def loadToFile(filename) -> object:
    with open(filename, 'rb') as input:
        return pickle.load(input)

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
                environment = loadToFile(args.environment)
            except Exception as e:
                print(type(e))
                print(e.value)
                raise Exception("Please input a valid environment")
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
                player = loadToFile('/'.join((path_objects, "players", args.playerAgent)))
            except Exception as excp:
                print(type(excp))
                print(excp)
                raise Exception("Please input a valid player agent")
            else:
                loaded_player = True
    
    if not loaded_player:
        player.set_up(action_space[0], observation_space[0])

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
                adversary = loadToFile('/'.join((path_objects, "adversaries", args.adversaryAgent)))
            except Exception as e:
                print(type(e))
                print(e.value)
                raise Exception("Please input a valid adversary agent")
            else:
                loaded_adversary = True
    if not loaded_adversary:
        adversary.set_up(action_space[1], observation_space[1])

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
    environment.runNumGames(player, adversary, numTrain)
    
    environment.enable_rendering()
    if not train_watch:
        player.disableLearning()
        adversary.disableLearning()
    print("Watching...")
    environment.runNumGames(player, adversary, numWatch)

    if save_player:
        Path("saved_objects/players").mkdir(parents=True, exist_ok=True)
        i = 0
        while os.path.exists(f"saved_objects/players/player{i}"):
            i += 1
        writeToFile(player, f"saved_objects/players/player{i}")
    
    temp_action_space = player.action_space
    player = PlayerAgent()
    player.set_up(temp_action_space)

    if not train_play:
        player.disableLearning()
        adversary.disableLearning()
    else:
        player.enableLearning()
        adversary.enableLearning()
    print("Playing...")
    environment.runNumGames(player, adversary, numPlay)

    environment.tear_down()


if __name__ == "__main__":
    main(sys.argv[1:])