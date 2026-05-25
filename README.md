# VideoGame MARL

This project is designed to provide an accessible, easy to use framework with which RL agents can
be trained in Multi-Agent scenarios. Q-Learning and SARSA agents in both Tabular and Function Approximation forms are provided and any PettingZoo environment can be used within this framework. Custom environments using the PettingZoo API are also acceptable; the DotsAndBoxes environment acts as an example implementation.

## Installation

Ensure python is installed.
Create a virtual environment:

```bash 
python3 -m venv {venv_name}
```

Initialise virtual environment:

```bash
source {venv_name}/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Training

Agents can be trained directly through the command line by calling the main file. They can be trained with or without rendering and subsequently player against by inputting actions into the command line. Multiple adversaries, observation and action abstractions and saving/loading of agents and environments is supported.

To train agents:

```bash
python3 main.py {env_type}
```

There are three environment types: "tictactoe", "connectfour" and "dotsandboxes". By default, calling the main file from the command line will use Random Agents as player and adversary and will train for 10000 games. DotsAndBoxes is currently the only environment that supports abstraction, other environments will not behave differently if abstraction options are specified.

All options are listed below. These can also be seen by calling `python3 main.py -h`.

| Option | Functionality |
| :----: | ----------------------------------------------------------------------------------------- |
|   -n   | Specifies number of games to train for |
|   -w   | Specifies number of games to watch agents play against agents |
|   -x   | Specifies number of games to play against adversarial agent |
|   -p   | Select a player Agent type e.g. qTab, sarsaFunc. Can also load a saved agent |
|   -a   | Select adversary Agents. Multiple adversaries are supported by compatible environments (dotsandboxes) |
|  -o:p  | Save player to saved_objects directory |
|  -o:a  | Save adversaries to saved_objects directory |
|  -o:e  | Save environment to saved_objects directory |
| -abs:o | Specify list of agent indexes to abstract observations for |
| -abs:a | Specify list of agent indexes to abstract actions for |
|  -l:p  | Allow adversary to learn and update during play sessions |
|  -d:p  | Prevent player from learning during train and watch sessions |
|  -d:a  | Prevent adversaries from learning during train and watch sessions |

Examples that illustrate the flexibility of the framework are listed below:

Training a Q-Learning Function Approximation adversary in ConnectFour against a random player for 100000 games and then playing against the adversary yourself for 10 games
```bash
python3 main.py connectfour -p randAgent -a qFunc -n 100000 -x 10
```

Saving a Q-Tabular adversary that is trained against a SARSA Func-Approx player in TicTacToe for 50000 games and then loading the adversary back into another training scenario (with learning disabled for it) to train a Q-Learning Func-Approx player for 50000 games.
```bash
python3 main.py tictactoe -p sarsafunc -a qtab -n 50000 -o:a afilename
python3 main.py tictactoe -p qfunc -a afilename -n 50000 -d:a
```

Training 3 SARSA Tabular adversarial agents against a Q-Learning Func-Approx Player in DotsAndBoxes for 123456789 human-rendered games with action abstraction enabled only for the player and the second adversary, and observation abstraction enabled for the first and third adversary, then playing against the adversaries for 100 games yourself while they continue to learn.

```bash
python3 main.py dotsandboxes -p qfunc -a sarsatab sarsatab sarsatab -w 123456789 -x 100 -abs:a 0 2 -abs:o 1 3 -l:p
```

## Evaluating

The evaluator file can be called from the command line or used as an object in other modules in order to provide representations and comparisons of logs maintained by agents and environments. The most common way to use it is by training and saving agents and/or environments and then using the evaluator to create graphs and csv files based on their logs.

To evaluate data for players, adversaries and environments respectively:
```bash
python3 evaluator.py player -o {filename1} {filename2} {filename3} ...
python3 evaluator.py adversary -o {filename1} {filename2} {filename3} ...
python3 evaluator.py environment -o {filename1} {filename2} {filename3} ...
```

At least one filename must be specified. The methods for evaluating players and adversaries are the same, the use of player and adversary only specifies which subdirectory of saved_objects to search for objects in.

Options for the evalutor are listed here. These can also be seen by calling `python3 evaluator.py -h`
| options | Functionality |
| :-----: | ------------------------------------------------------------------------------------- |
|   -o    | Specify the filenames for the objects that are being compared. Must have at least one |
|   -n    | Specify the names to give objects in legends. If not provided, filenames are used |
|   -d    | Specify folder name to put results in. |
|   -w    | Specify the size of the rolling window relative to the amount of data |

All results by default are stored in the results directory. This can be configured in file.py

## Experimenting

There are several experiments listed in the experiments file that were carried out during the project. These experiments are intended to evaluate the performance of different RL agents in different circumstances, both in terms of efficiency and effectiveness.

To perform the existing experiments:

```bash
python3 experiments.py
```

It is simple to add your own experiments to the experiments.py file based on the ones already provided.

## Testing

To run the test suite:

```bash
pytest
```

To check the coverage of the test suite:

```bash
coverage run -m pytest
coverage html
```

The coverage report will be in the htmlcov directory as index.html.