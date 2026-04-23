import os
import dill as pickle
from pathlib import Path

path_objects = "saved_objects"
short_to_full = {
    'p': 'players',
    'a': 'adversaries',
    'e': 'environments'
}
Path('/'.join((path_objects, "players"))).mkdir(parents=True, exist_ok=True)
Path('/'.join((path_objects, "adversaries"))).mkdir(parents=True, exist_ok=True)
Path('/'.join((path_objects, "environments"))).mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

def writeToFile(object: object, filename: str, short: str):
    with open(make_file_path(filename, short), 'wb') as outp:
        pickle.dump(object, outp)

def loadFromFile(filename, short) -> object:
    with open(get_file_path(filename, short), 'rb') as input:
        return pickle.load(input)

def make_file_path(filename, short):
    return '/'.join((path_objects, short_to_full[short], filename))

def get_file_path(filename, short):
    return '/'.join((path_objects, short_to_full[short], filename))

def make_results_path(dir_name: str):
    path = '/'.join(("results", dir_name))
    if (os.path.exists(path)):
        i = 2
        while os.path.exists(path + str(i)):
            i += 1
        path = path + str(i)
    
    Path(path).mkdir(parents=True, exist_ok=True)
    return path