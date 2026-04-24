import os
import dill as pickle
from pathlib import Path

path_objects = "saved_objects"
path_results = "results"

short_to_full = {
    'p': 'players',
    'a': 'adversaries',
    'e': 'environments'
}

Path('/'.join((path_objects, "players"))).mkdir(parents=True, exist_ok=True)
Path('/'.join((path_objects, "adversaries"))).mkdir(parents=True, exist_ok=True)
Path('/'.join((path_objects, "environments"))).mkdir(parents=True, exist_ok=True)
Path(path_results).mkdir(parents=True, exist_ok=True)

def writeToFile(object: object, filename: str, short: str) -> None:
    """
    Args:
        object: A python object
        filename: A string representing the name of the file to save the object in
        short: A character that indicates the directory under which the object is to be saved

    Save an object to a file.
    """

    with open(get_file_path(filename, short), 'wb') as outp:
        pickle.dump(object, outp)

def loadFromFile(filename: str, short) -> object:
    """
    Args:
        filename: A string representing the name of the file to load the object from
        short: A character that indicates the directory from which the object is to be loaded.

    Returns:
        An object

    Load an object from a file.
    """

    with open(get_file_path(filename, short), 'rb') as input:
        return pickle.load(input)

def get_file_path(filename: str, short: str) -> str:
    """
    Args:
        filename: A string representing the name of the file
        short: A character that indicates the directory preceding the filename

    Returns:
        A string representation of the path to the file

    Get a filepath based on the filename and short
    """

    return '/'.join((path_objects, short_to_full[short], filename))

def make_results_path(dir_name: str) -> str:
    """
    Args:
        dir_name: The name of the subdirectory under which the results will be stored.

    Returns:
        A string representation of the path to the directory
    """

    path = '/'.join((path_results, dir_name))

    #Check whether the path already exists. If so try adding incrementing numbers to the path.
    if (os.path.exists(path)):
        i = 2
        while os.path.exists(path + str(i)):
            i += 1
        path = path + str(i)
    
    #Create the path and then return
    Path(path).mkdir(parents=True, exist_ok=True)
    return path