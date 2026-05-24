# Starting Up

Create a virtual environment

`python3 -m venv {virtualenv name}`

Initialise virtual environment

`source {virtualenv name}/bin/activate`

Install requirements

`pip install -r requirements.txt`

# Using the software

To train agents

`python3 main.py {env_tyoe}`

There are several optional parameters

| Option | Functionality |
| ------ | ------------------------------------------------------------- |
|   -n   | Specifies number of games to train for |
|   -w   | Specifies number of games to watch agents play against agents |
|   -p   | Specifies number of games to play against adversarial agent |

To perform the existing experiments

`python3 experiments.py`
