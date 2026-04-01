import sys
from argparse import ArgumentParser
from environments import connectfour
from agents.qLearnAgent import QLearnAgent
from agents.randomAgent import RandomAgent
from agents.playerAgent import PlayerAgent
import torch
from evaluator import Evaluator
import pickle

def parse(args: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("-e", "--env", dest="environment")
    parser.add_argument("-p", "--player", dest="playerAgent")
    parser.add_argument("-a", "--adversary", dest="adversaryAgent")
    parser.add_argument("-n", "--numtrain", dest="numTrain", type=int, default=1000)
    parser.add_argument("-x", "--numplay", dest="numPlay", type=int, default=0)
    return parser.parse_args(args)

def match_args(args):
    match args.environment:
        case "connectfour" | None:
            environment = connectfour.ConnectFour()
        case _:
            raise Exception("Please input a valid environment")
    

    match args.playerAgent:
        case "qAgent":
            player = QLearnAgent()
        case "playerAgent":
            player = PlayerAgent()
        case "randAgent" | None:
            player = RandomAgent()
        case _:
            raise Exception("Please input a valid player agent")

    match args.adversaryAgent:
        case "qAgent":
            adversary = QLearnAgent()
        case "randAgent" | None:
            adversary = RandomAgent()
        case _:
            raise Exception("Please input a valid adversary agent")
        
    if args.numTrain < 1:
        raise Exception("Number of Games must be higher than 0")
    
    if args.numPlay < 0:
        raise Exception("Number of play games cannot be negative")
        
    action_space = environment.get_action_spaces()
    player.set_up(action_space[0])
    adversary.set_up(action_space[1])

    return environment, player, adversary, args.numTrain, args.numPlay

def writeToFile(object, filename='out_file'):
    with open(filename, 'wb') as outp:
        pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)

def main(args: list[str] | None =  None):
    environment, player, adversary, numTrain, numPlay = match_args(parse(args))

    #environment.enable_rendering()
    environment.runNumGames(player, adversary, numTrain)
    """
    for x in adversary.q_values:
        print(adversary.q_values[x])
    """
    eval = Evaluator()
    eval.plotMovingAverage(adversary.training_error, 10000)
    eval.show()
    if numPlay > 0:
        temp_action_space = player.action_space
        player = PlayerAgent()
        player.set_up(temp_action_space)
        environment.enable_rendering()
        environment.runNumGames(player, adversary, numPlay)
        environment.disable_rendering()
    environment.tear_down()


if __name__ == "__main__":
    main(sys.argv[1:])