import sys
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from file import loadFromFile, make_results_path
from agents.agent import Agent
from environments.environment import Environment

class Evaluator:
    def __init__(self, path, window):
        self.path = path
        self.window = window

    def plotMovingAverage(self, values, label):
        values_moving_average = np.convolve(np.array(values), np.ones(self.window), mode="valid")/self.window
        plt.plot(values_moving_average, label=label)

    def plotAgents(self, agents: list[Agent], names: list[str]):
        plots = [
            {"title": "Moving Average of Time taken for Get Action Method", "logger_param": "get_action",
             "xlabel": "Number of get action calls", "ylabel": "Average Time Taken(nanoseconds)",
             "filename": "get-action-time"},
            {"title": "Moving Average of Time Taken for Update Method", "logger_param": "update",
             "xlabel": "Number of updates", "ylabel": "Average Time Taken (nanoseconds)",
             "filename": "update-time"},
            {"title": "Moving Average of Cumulative Rewards", "logger_param": "record",
             "xlabel": "Number of episodes", "ylabel": "Average Cumulative Reward",
             "filename": "cumulative-rewards"},
            {"title": "Moving Average of Training Error", "logger_param": "training_error",
             "xlabel": "Number of updates", "ylabel": "Average Training error",
             "filename": "training-error"}
        ]
        
        for plot in plots:
            plt.title(plot["title"])
            plt.xlabel(plot["xlabel"])
            plt.ylabel(plot["ylabel"])
            for agent, name in zip(agents, names):
                if plot["logger_param"] in agent.logger:
                    self.plotMovingAverage(agent.logger[plot["logger_param"]], name)
            self.saveResult(plot["filename"])

    def plotEnvironments(self, environments: list[Environment], names: list[str]):
        plots = [
            {"title": "Moving Average of Number of Iterations", "logger_param": "num_iterations",
             "xlabel": "Episode Number", "ylabel": "Average Number of Iterations", "filename": "iterations"},
            {"title": "Moving Average of Time Taken per Episode", "logger_param": "run",
             "xlabel": "Episode Number", "ylabel": "Average Time Taken (nanoseconds)", "filename": "time-per-episode"}
        ]
        for plot in plots:
            plt.title(plot["title"])
            plt.xlabel(plot["xlabel"])
            plt.ylabel(plot["ylabel"])
            for env, name in zip(environments, names):
                if plot["logger_param"] in env.logger:
                    self.plotMovingAverage(env.logger[plot["logger_param"]], name)
            self.saveResult(plot["filename"])
    
    def saveResult(self, filename):
        plt.legend()
        plt.savefig(self.path + f"/{filename}", dpi=300, bbox_inches='tight')
        plt.close()


def main(args: list[str] | None = None):
    parser = ArgumentParser()
    parser.add_argument("type", choices=["environment", "player", "adversary"])
    parser.add_argument("-o", "--objects", nargs='+', type=str)
    parser.add_argument("-n", "--names", nargs='+', type=str)
    parser.add_argument("-d", "--destination", default="res", type=str)
    parser.add_argument("-w", "--window-size", default="1000", type=int)
    parsed_args = parser.parse_args(args)
    
    if parsed_args.window_size <= 0:
        raise Exception("Window size must be greater than 0")
    
    eval = Evaluator(make_results_path(parsed_args.destination), 1000)

    if len(parsed_args.objects) != len(parsed_args.names):
        raise Exception("The number of objects and number of names must be equal")
    
    if parsed_args.type == "environment":
        envs = []
        for path in parsed_args.objects:
            env = loadFromFile(path, 'e')
            assert isinstance(env, Environment)
            envs.append(env)
        eval.plotEnvironments(envs, parsed_args.names)
    else:
        if parsed_args.type == 'player': short = 'p'
        else: short = 'a'

        agents = []
        for path in parsed_args.objects:
            agent = loadFromFile(path, short)
            assert isinstance(agent, Agent)
            agents.append(agent)
        eval.plotAgents(agents, parsed_args.names)
        

if __name__ == "__main__":
    main(sys.argv[1:])