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

def writeToFile(object: object, filename: str, short: str):
    with open(make_file_path(filename, short), 'wb') as outp:
        pickle.dump(object, outp)

def loadFromFile(filename, short) -> object:
    with open(get_file_path(filename, short), 'rb') as input:
        return pickle.load(input)

def make_file_path(filename, short):
    path = '/'.join((path_objects, short_to_full[short], filename))
    if not (os.path.exists(path)):
        return path
    else:
        i = 0
        while os.path.exists(path + str(i)):
            i += 1
        return path + str(i)

def get_file_path(filename, short):
    return '/'.join((path_objects, short_to_full[short], filename))