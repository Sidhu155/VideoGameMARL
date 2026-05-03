import sys
from argparse import ArgumentParser, Namespace
from gymnasium import Space
from file import writeToFile, loadFromFile, get_file_path
from environments.environment import Environment
from environments import connectfour, tictactoe, dotsAndBoxes
from agents.agent import Agent
from agents.tabularQAgents import QTabAgent, SARSATabAgent
from agents.funcQAgents import QFuncApproxAgent, SARSAFuncApproxAgent
from agents.randomAgent import RandomAgent
from agents.playerAgent import PlayerAgent

def parse(args: list[str] | None = None) -> Namespace:
    """
    Args:
        args: A list of arguments passed to the command line when calling main
    
    Returns the command line arguments parsed by an Argument Parser
    """

    parser = ArgumentParser()
    parser.add_argument("environment", type=str)
    parser.add_argument("-p", "--player", dest="playerAgent", default="randAgent")
    parser.add_argument("-a", "--adversary", nargs='*', dest="adversaryAgent", default=["randAgent"])
    parser.add_argument("-n", "--numtrain", dest="numTrain", type=int, default=1000)
    parser.add_argument("-w", "--numwatch", dest="numWatch", type=int, default=0)
    parser.add_argument("-x", "--numplay", dest="numPlay", type=int, default=0)
    parser.add_argument("-o:p", "--outfile-player")
    parser.add_argument("-o:a", "--outfile-adversary", nargs='*', default=[])
    parser.add_argument("-o:e", "--outfile-env")
    parser.add_argument("-l", "--learn-watch-play", action="store_true")
    parser.add_argument("-d:p", "--disable-player-learn", action="store_true")
    parser.add_argument("-d:a", "--disable-adversary-learn", action="store_true")
    return parser.parse_args(args)

def match_env(arg: str, num_agents: int, bool_save: bool):
    match arg:
        case "connectfour":
            environment = connectfour.ConnectFour()
        case "tictactoe":
            environment = tictactoe.TicTacToe()
        case "dotsandboxes":
            environment = dotsAndBoxes.DotsAndBoxes(
                num_agents=num_agents,
                board_length=5
            )
        case _:
            #Attempt to load the environment from the directory specified in file.py
            try:
                environment: Environment = loadFromFile(arg, 'e')
                assert isinstance(environment, Environment)
            except Exception:
                raise Exception("Please input a valid environment or filename. Ensure that " \
                f"saved environments are located in the {get_file_path('', 'e')} directory")
            else:
                environment.tear_down()
                environment.logger.enableLogging()      #Enable logging in case the saved env has it disabled
                environment.logger.resetLogger()
                environment.create_env()                #Create env as it may be closed beforehand

    #Logging is unnecessary if the environment will not be saved
    if not bool_save:
        environment.logger.disableLogging()

    return environment

def match_agent(arg: str, short: str, action_space: Space, observation_space: Space, bool_save: bool) -> Agent:
    loaded = False
    match arg:
        case "qTab":
            agent = QTabAgent()
        case "sarsaTab":
            agent = SARSATabAgent()
        case "qFunc":
            agent = QFuncApproxAgent()
        case "sarsaFunc":
            agent = SARSAFuncApproxAgent()
        case "playerAgent":
            agent = PlayerAgent()
        case "randAgent":
            agent = RandomAgent()
        case _:
            try:
                agent: Agent = loadFromFile(arg, short)
                assert isinstance(agent, Agent)
                if agent.observation_space != observation_space:
                    raise ValueError("Loaded agent's observation space does not match environment")
                if agent.action_space != action_space:
                    raise ValueError("Loaded agent's action space does not match environment")
            except Exception as excp:
                #If the error is incorrect action or observation space, raise it
                #Otherwise it is incorrect filename or agent type being passed through the command line
                if type(excp) == ValueError:
                    raise excp
                else:
                    raise Exception("Please input a valid agent or filename. Ensure that " \
                    f"saved agents are located in the {get_file_path('', short)} directory")
            else:
                agent.enableLearning()          #Enable learning in case the saved object had it disabled
                agent.logger.enableLogging()    #Same reason as above
                agent.logger.resetLogger()
                loaded = True
    
    #If agent was loaded, action_space and observation_space will already be set up
    if not loaded:
        agent.set_up(action_space, observation_space)
    #Logging is unnecessary if agent will not be saved.
    if not bool_save:
        agent.logger.disableLogging()

    return agent

def match_args(args) -> tuple:
    """
    Args:
        args: The parsed arguments from the command line
        
    Returns:
        A tuple of objects including environment, players, adversaries, other flags and constants
        
    Takes parsed args and creates objects based on their information. Provides the objects needed
    for the main class.
    """

    #Create Environment
    environment = match_env(args.environment, len(args.adversaryAgent) + 1, bool(args.outfile_env))

    action_spaces = environment.get_action_spaces()
    observation_spaces = environment.get_observation_spaces()

    #Create Player
    player = match_agent(args.playerAgent, 'p', action_spaces[0], observation_spaces[0], bool(args.outfile_player))

    #Create adversaries
    adversaries = []
    index = 1           #Represents index of action and observation spaces corresponding to adversary
    for adversary_type in args.adversaryAgent:
        #Bool save is true even if index is equal to length of adversaries as index starts at one
        bool_save = (index <= len(args.outfile_adversary))
        adversary = match_agent(adversary_type, 'a', action_spaces[index], observation_spaces[index], bool_save)
        adversaries.append(adversary)
        index += 1

    #Check non-negative train, watch and play
    if args.numTrain < 0:
        raise Exception("Number of games for training cannot be negative")
    
    if args.numWatch < 0:
        raise Exception("Number of games for watching cannot be negative")
    
    if args.numPlay < 0:
        raise Exception("Number of games for play cannot be negative")
    
    return (environment, player, tuple(adversaries), args.numTrain, args.numWatch, args.numPlay,
            args.outfile_player, args.outfile_adversary, args.outfile_env, 
            args.learn_watch_play, args.disable_player_learn, args.disable_adversary_learn)

def main(args: list[str] | None =  None):
    (environment, player, adversaries, numTrain, numWatch, numPlay,
    outfile_player, outfile_adversary, outfile_env, 
    learn_watch_play, disable_player_learn, disable_adversary_learn) = match_args(parse(args))

    #When match_args creates any agent, learning is enabled by default.
    #Here if specified by the user, learning can be disabled for players and adversaries throughout main process.
    if disable_player_learn:
        player.disableLearning()
    if disable_adversary_learn:
        for adversary in adversaries:
            adversary.disableLearning()

    #Training period
    print("Training...")
    environment.runNumGames((player, *adversaries), numTrain)

    #If learning is specified as false during watch and play period, disable all learning
    if not learn_watch_play:
        player.disableLearning()
        for adversary in adversaries:
            adversary.disableLearning()

    environment.enable_rendering()

    #Watching period
    print("Watching...")
    environment.runNumGames((player, *adversaries), numWatch)

    #If outfile specified save player agent.
    if outfile_player: writeToFile(player, outfile_player, 'p') 
    #Swap to playerAgent and set up action space
    temp_action_space = player.action_space
    player = PlayerAgent()
    player.set_up(temp_action_space)
    
    #Playing period
    print("Playing...")
    environment.runNumGames((player, *adversaries), numPlay)
    
    #Save adversaries if outfiles specified
    if outfile_adversary:
        for adversary, outfile in zip(adversaries, outfile_adversary):
            writeToFile(adversary, outfile, 'a') 

    environment.tear_down()
    #Save environments if outfile specified
    if outfile_env: writeToFile(environment, outfile_env, 'e') 


if __name__ == "__main__":
    main(sys.argv[1:])