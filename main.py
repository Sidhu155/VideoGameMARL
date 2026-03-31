import sys
from argparse import ArgumentParser
from environments import connectfour
from agents.qLearnAgent import QLearnAgent
from agents.randomAgent import RandomAgent
import torch
from evaluator import Evaluator
import pickle

def parse(args: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("-e", "--env", dest="environment")
    parser.add_argument("-p", "--player", dest="playerAgent")
    parser.add_argument("-a", "--adversary", dest="adversaryAgent")
    parser.add_argument("-n", "--numgames", dest="numGames", type=int, default=1000)
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
        
    if args.numGames < 1:
        raise Exception("Number of Games must be higher than 0")
        
    action_space = environment.get_action_spaces()
    player.set_up(action_space[0])
    adversary.set_up(action_space[1])

    return environment, player, adversary, args.numGames

def writeToFile(object, filename='out_file'):
    with open(filename, 'wb') as outp:
        pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)

def main(args: list[str] | None =  None):
    environment, player, adversary, numGames = match_args(parse(args))

    environment.runNumGames(player, adversary, numGames)
    eval = Evaluator()
    eval.plotMovingAverage(adversary.training_error, 1000)
    eval.show()


if __name__ == "__main__":
    main(sys.argv[1:])