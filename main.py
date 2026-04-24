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
    parser.add_argument("-p", "--player", dest="playerAgent", default="randAgent")
    parser.add_argument("-a", "--adversary", nargs='*', dest="adversaryAgent", default=["randAgent"])
    parser.add_argument("-n", "--numtrain", dest="numTrain", type=int, default=1000)
    parser.add_argument("-w", "--numwatch", dest="numWatch", type=int, default=0)
    parser.add_argument("-x", "--numplay", dest="numPlay", type=int, default=0)
    parser.add_argument("-o:p", "--outfile-player")
    parser.add_argument("-o:a", "--outfile-adversary", nargs='*')
    parser.add_argument("-o:e", "--outfile-env")
    parser.add_argument("-t:w", "--train-watch", action="store_true")
    parser.add_argument("-t:p", "--train-play", action="store_true")
    parser.add_argument("-d:p", "--disable-player-learn", action="store_true")
    parser.add_argument("-d:a", "--disable-adversary-learn", action="store_true")
    return parser.parse_args(args)

def match_args(args):
    match args.environment:
        case "connectfour":
            environment = connectfour.ConnectFour()
        case "tictactoe":
            environment = tictactoe.TicTacToe()
        case _:
            try:
                environment: Environment = loadFromFile(args.environment, 'e')
                assert isinstance(player, Environment)
            except Exception as excp:
                print(type(excp))
                print(excp)
                raise Exception("Please input a valid environment or filepath")
            else:
                environment.create_env()
    action_space = environment.get_action_spaces()
    observation_space = environment.get_observation_spaces()

    loaded_player = False
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
        case "randAgent":
            player = RandomAgent()
        case _:
            try:
                player: Agent = loadFromFile(args.playerAgent, 'p')
                assert isinstance(player, Agent)
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

    adversaries = []
    index = 1
    for adversary_type in args.adversaryAgent:
        loaded_adversary = False
        match adversary_type:
            case "qTab":
                adversary = QTabAgent()
            case "sarsaTab":
                adversary = SARSATabAgent()
            case "qFunc":
                adversary = QFuncApproxAgent()
            case "sarsaFunc":
                adversary = SARSAFuncApproxAgent()
            case "randAgent":
                adversary = RandomAgent()
            case _:
                try:
                    adversary: Agent = loadFromFile(adversary_type, 'a')
                    assert isinstance(adversary, Agent)
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
            adversary.set_up(action_space[index], observation_space[index])
        if args.disable_adversary_learn:
            adversary.disableLearning()

        adversaries.append(adversary)
        index += 1

    if args.numTrain < 0:
        raise Exception("Number of games for training cannot be negative")
    
    if args.numWatch < 0:
        raise Exception("Number of games for watching cannot be negative")
    
    if args.numPlay < 0:
        raise Exception("Number of games for play cannot be negative")

    return (environment, player, tuple(adversaries), args.numTrain, args.numWatch, args.numPlay,
            args.outfile_player, args.outfile_adversary, args.outfile_env, args.train_watch, args.train_play)

def main(args: list[str] | None =  None):
    torch.set_default_device('mps')
    (environment, player, adversaries, numTrain, numWatch, numPlay,
    outfile_player, outfile_adversary, outfile_env, train_watch, train_play) = match_args(parse(args))

    print("Training...")
    environment.runNumGames((player, *adversaries), numTrain)

    environment.enable_rendering()
    if not train_watch:
        player.disableLearning()
        for adversary in adversaries:
            adversary.disableLearning()
    else:
        player.enableLearning()
        for adversary in adversaries:
            adversary.enableLearning()
    print("Watching...")
    environment.runNumGames((player, *adversaries), numWatch)

    if outfile_player: writeToFile(player, outfile_player, 'p') 
    
    temp_action_space = player.action_space
    player = PlayerAgent()
    player.set_up(temp_action_space)
    if not train_play:
        for adversary in adversaries:
            adversary.disableLearning()
    else:
        for adversary in adversaries:
            adversary.enableLearning()
    print("Playing...")
    environment.runNumGames((player, *adversaries), numPlay)
    
    if outfile_adversary:
        for adversary, outfile in zip(adversaries, outfile_adversary):
            writeToFile(adversary, outfile, 'a') 

    environment.tear_down()
    if outfile_env: writeToFile(environment, outfile_env, 'e') 


if __name__ == "__main__":
    main(sys.argv[1:])