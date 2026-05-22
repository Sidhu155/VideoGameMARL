import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from logger import Logger
from agents.agent import Agent
from environments.environment import Environment
from file import loadFromFile, make_results_path

class Evaluator:
    """
    The Evaluator Class takes environment, player or adversary objects and compares
    data about them provided by their loggers. It plots this information under the 
    results directory (stated in file.py) and also provides a csv file of important
    features of the data.
    """

    def __init__(self, path: str, window_ratio: float) -> None:
        """
        Args:
            path: The path that the evaluator saves the results in
            window_ratio: The ratio of the rolling window to the length of the data
        """

        self.path = path
        self.window_ratio = window_ratio
        self.writer = csv.writer(open(self.path + '/data.csv', 'w', newline=''))
        self.writer.writerow(self.getCSVHeaders())

    def plotMovingAverage(self, values: np.ndarray, label: str) -> None:
        """
        Args:
            values: An array of data points
            label: The label for the line being drawn. This is included in the legend of the plot
        
        Plot a moving average onto the graph based on the values provided.
        """

        rolling_window = int(self.window_ratio * values.size) + 1

        #Use convolve to create a moving average. Convolve essentially pushes the data points
        #backwards through an array of ones (size of the rolling window). It multiplies all 
        #data points in values that have a corresponding one in the window by that one, adds them
        #together and divides by the window length
        values_moving_average = np.convolve(values, np.ones(rolling_window), mode="valid")/rolling_window
        plt.plot(values_moving_average, label=label)

    def plotAgents(self, agent_loggers: list[Logger], names: list[str]) -> None:
        """
        Args:
            agent_loggers: A list of agent loggers
            names: A list of names for the agents
        
        Plots graphs based on information from the loggers of the agents. Saves these graphs to
        the directory provided to the evaluator object. Also saves information such as median and mean
        of said logs to a csv file in the directory.
        """

        #This defines the plots that will be made using the loggers of the agents
        plots = [
            {"title": "Moving Average of Time taken for Get Action Method", "logger_param": "get_action",
             "xlabel": "Number of get action calls", "ylabel": "Average Time Taken (seconds)",
             "filename": "get-action-time"},
            {"title": "Moving Average of Time Taken for Update Method", "logger_param": "update",
             "xlabel": "Number of updates", "ylabel": "Average Time Taken (seconds)",
             "filename": "update-time"},
            {"title": "Moving Average of Cumulative Rewards", "logger_param": "record",
             "xlabel": "Number of episodes", "ylabel": "Average Cumulative Reward",
             "filename": "cumulative-rewards"},
            {"title": "Moving Average of Training Error", "logger_param": "training_error",
             "xlabel": "Number of updates", "ylabel": "Average Training error",
             "filename": "training-error"}
        ]

        for plot in plots:
            self.plot(agent_loggers, names, **plot)

        plt.title("Proportion of Actions Taken")
        plt.xlabel("Action")
        plt.ylabel("Fraction of Total Number of Actions Taken")
        for logger, name in zip(agent_loggers, names):
            if logger.hasKeyInLogs("history_actions"):
                log = np.array(logger.getLogs("history_actions"))
                values, counts = np.unique(log, return_counts=True)
                fractions = counts/len(log)

                plt.plot(values, fractions, label=name)
        self.saveResult("action_proportions")

    def plotEnvironments(self, env_loggers: list[Logger], names: list[str]) -> None:
        """
        Args:
            env_loggers: A list of environment loggers
            names: A list of names for the environments
        
        Plots graphs based on information from the loggers of the environments. Saves these graphs to
        the directory provided to the evaluator object. Also saves information such as median and mean
        of said logs to a csv file in the directory.
        """

        plots = [
            {"title": "Moving Average of Number of Iterations", "logger_param": "num_iterations",
             "xlabel": "Episode Number", "ylabel": "Average Number of Iterations", "filename": "iterations"},
            {"title": "Moving Average of Number of States", "logger_param": "num_states",
             "xlabel": "Episode Number", "ylabel": "Average Number of Cumulative States", "filename": "states"},
            {"title": "Moving Average of Time Taken per Episode", "logger_param": "run",
             "xlabel": "Episode Number", "ylabel": "Average Time Taken (seconds)", "filename": "time-per-episode"},
            {"title": "Moving Average of Memory Usage per Episode", "logger_param": "mem_run",
             "xlabel": "Episode Number", "ylabel": "Average Memory Usage (MB)", "filename": "mem-per-episode"},
        ]

        for plot in plots:
            self.plot(env_loggers, names, **plot)

        plt.title("Moving Average of Time per Iteration")
        plt.xlabel("Episode Number")
        plt.ylabel("Average Time per Iteration")
        for logger, name in zip(env_loggers, names):
            if logger.hasKeyInLogs("num_iterations") and logger.hasKeyInLogs("run"):
                log = np.array(logger.getLogs("run"), dtype=np.float32)
                log /= np.array(logger.getLogs("num_iterations"), dtype=np.float32)
                self.plotMovingAverage(log, name)
                self.writeLogToCSV(name, "time_per_iter", log)
        self.saveResult("time-per-iteration")
    
    def plot(self, loggers: list[Logger], names: list[str], title: str, 
             xlabel: str, ylabel: str, logger_param: str, filename: str) -> None:
        """
        Args:
            loggers: A list of Logger objects
            names: A list of names corresponding to each logger object
            title: A title for the plot
            xlabel: A label for the x-axis
            ylabel: A label for the y-axis
            logger_param: The key in the logger to get logs from
            filename: The name of the file to save the plot under
            
        Plot a graph and save the figure based on loggers provided
        """
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for logger, name in zip(loggers, names):
            if logger.hasKeyInLogs(logger_param):
                #Convert logs into array with float type. Float prevents issues with infinities
                #when calculating mean
                log = np.array(logger.getLogs(logger_param), dtype=np.float32)
                self.plotMovingAverage(log, name)
                self.writeLogToCSV(name, logger_param, log)
        self.saveResult(filename)

    def saveResult(self, filename: str) -> None:
        """
        Args:
            filename: A string indicating what name to save the file under
            
        Saves the current plot under the filename and inside the directory provided to the evaluator object
        """
        
        plt.legend()
        plt.savefig(self.path + f"/{filename}", dpi=300, bbox_inches='tight')
        plt.close()

    def writeLogToCSV(self, name: str, param: str, log: np.ndarray) -> None:
        """
        Args:
            name: The name of the object from which the log is given
            param: A string that describes the nature of the logs
            log: An array of data points
        
        Takes a list of data points, with their corresponding parameter and the name of
        the object, and converts them into min, median, mean, max and std, writing these into a
        csv file using the objects csv writer.
        """

        num_data_points = len(log)
        min = np.min(log)
        median = np.median(log)
        mean = np.mean(log)
        max = np.max(log)
        std = np.std(log)
        self.writer.writerow([name, param, num_data_points, min, median, mean, max, std])

    def getCSVHeaders(self) -> list[str]:
        """
        Return:
            A List of csv table headers that are used when creating the csv file
        """
        return [
            'Name', 'Parameter', 'Num Data Points', 
            'Min', 'Median', 'Mean', 'Max', 'Standard Deviation'
        ]

def get_parsed_args(args) -> tuple:
    """
    Args:
        args: Arguments passed when calling the evaluator file from the command line
        
    Returns:
        Tuple of type, objects, names, destination and window-ratio
        
    Creates a parser with which to parse given arguments. Ensures window ratio, objects, name 
    arguments are also usable by the main method.
    """

    parser = ArgumentParser()
    parser.add_argument("type", choices=["environment", "player", "adversary"])
    parser.add_argument("-o", "--objects", nargs='+', type=str)
    parser.add_argument("-n", "--names", nargs='*', type=str, default=[])
    parser.add_argument("-d", "--destination", default="res", type=str)
    parser.add_argument("-w", "--window-ratio", default="0.05", type=float)
    parsed_args = parser.parse_args(args)
    
    if parsed_args.window_ratio <= 0:
        raise Exception("Window ratio must be greater than 0")
    
    if len(parsed_args.objects) < len(parsed_args.names):
        parsed_args.names = parsed_args.names[:len(parsed_args.objects)]
    elif len(parsed_args.objects) > len(parsed_args.names):
        parsed_args.names.extend(parsed_args.objects[len(parsed_args.names):])

    return (parsed_args.type, parsed_args.objects, parsed_args.names, 
            parsed_args.destination, parsed_args.window_ratio)

def main(args: list[str] | None = None):
    type, objects, names, destination, window_ratio = get_parsed_args(args)
    eval = Evaluator(make_results_path(destination), window_ratio)

    #Check what type of object is being evaluated
    if type == "environment":
        env_loggers = []
        #Load all environments and ensure they are instances of Environments. Then plot
        for path in objects:
            env = loadFromFile(path, 'e')
            assert isinstance(env, Environment)
            env_loggers.append(env.logger)
        eval.plotEnvironments(env_loggers, names)
    else:
        #The short determines which directory to search under. Player agents are located
        #in the players directory whereas adversaries are located in the adversaries directory.
        if type == 'player': short = 'p'
        else: short = 'a'

        #Ensure all agents loaded are indeed agents. Then plot agents
        agent_loggers = []
        for path in objects:
            agent = loadFromFile(path, short)
            assert isinstance(agent, Agent)
            agent_loggers.append(agent.logger)
        eval.plotAgents(agent_loggers, names)

if __name__ == "__main__":
    main(sys.argv[1:])