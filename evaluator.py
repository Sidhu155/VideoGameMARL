import sys
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from file import loadFromFile
from agents.agent import Agent
from environments.environment import Environment

class Evaluator:
    def __init__(self):
        pass

    def plotMovingAverage(self, values, window):
        values_moving_average = np.convolve(np.array(values), np.ones(window), mode="valid")/window
        self.plot(values_moving_average)

    def plot(self, values):
        plt.plot(range(len(values)), values)
    
    def show(self):
        plt.show()

    def plotAgents(self, agents: list[Agent]):
        pass

    def plotEnvironments(self, environments: list[Environment]):
        pass


def main(args: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("type", choices=["environment", "agent"])
    parser.add_argument("objects", nargs='+', type=str)
    args = parser.parse_args(args)
    eval = Evaluator()

    if args.type == "environment":
        for env in args.objects:
            assert type(env) == Environment
        eval.plotEnvironments(args.objects)
    else:
        for agent in args.objects:
            assert type(env) == Agent
        eval.plotAgents(args.objects)

    eval.show()
        

    

if __name__ == "__main__":
    main(sys.argv[1:])