import sys
import warnings
from argparse import ArgumentParser, Namespace
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
    parser.add_argument("-n", "--numtrain", dest="numTrain", type=int, default=10000)
    parser.add_argument("-w", "--numwatch", dest="numWatch", type=int, default=0)
    parser.add_argument("-x", "--numplay", dest="numPlay", type=int, default=0)
    parser.add_argument("-o:p", "--outfile-player")
    parser.add_argument("-o:a", "--outfile-adversary", nargs='*', default=[])
    parser.add_argument("-o:e", "--outfile-env")
    parser.add_argument("-l:p", "--learn-play", action="store_true", 
                        help='Allow adversary to learn and update when you ' \
                        'are playing against it')
    parser.add_argument("-d:p", "--disable-player-learn", action="store_true")
    parser.add_argument("-d:a", "--disable-adversary-learn", action="store_true")
    parser.add_argument("-abs:o", "--abstraction-observation", nargs='*', type=int, default=[],
                        help='Specify agent indices for observation abstraction. Player is 0, ' \
                        'first adversary is 1, second adversary is 2, etc.')
    parser.add_argument("-abs:a", "--abstraction-action", nargs='*', type=int, default=[], 
                        help='Specify agent indices for action abstraction. Player is 0, ' \
                        'first adversary is 1, second adversary is 2, etc.')
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

def match_agent(arg: str, short: str, env: Environment, idx: int,
                obs_abstract: bool, act_abstract: bool, bool_save: bool) -> Agent:
    loaded = False
    kwargs = {"obs_abstraction":obs_abstract, "action_abstraction":act_abstract}
    match arg:
        case "qTab":
            agent = QTabAgent(**kwargs)
        case "sarsaTab":
            agent = SARSATabAgent(**kwargs)
        case "qFunc":
            agent = QFuncApproxAgent(**kwargs)
        case "sarsaFunc":
            agent = SARSAFuncApproxAgent(**kwargs)
        case "playerAgent":
            agent = PlayerAgent(**kwargs)
        case "randAgent":
            agent = RandomAgent(**kwargs)
        case _:
            try:
                agent: Agent = loadFromFile(arg, short)
                assert isinstance(agent, Agent)
                if agent.obs_abstraction != obs_abstract:
                    warnings.warn(f"Loaded Agent {idx} obs_abstraction: {agent.obs_abstraction}")
                if agent.action_abstraction != act_abstract:
                    warnings.warn(f"Loaded Agent {idx} action_abstraction: {agent.action_abstraction}")
                
                if agent.observation_space != env.get_observation_space(idx, agent.obs_abstraction):
                    raise ValueError("Loaded agent's observation space does not match environment")
                if agent.action_space != env.get_action_space(idx, agent.action_abstraction):
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
        agent.set_up(env.get_action_space(idx, act_abstract), env.get_observation_space(idx, obs_abstract))
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

    #Create Player
    player = match_agent(args.playerAgent, 'p', environment, 0,
                         0 in args.abstraction_observation, 0 in args.abstraction_action,
                         bool(args.outfile_player))

    #Create adversaries
    adversaries = []
    index = 1           #Represents index of action and observation spaces corresponding to adversary
    for adversary_type in args.adversaryAgent:
        #Bool save is true even if index is equal to length of adversaries as index starts at one
        bool_save = (index <= len(args.outfile_adversary))

        adversary = match_agent(adversary_type, 'a', environment, index,
                                index in args.abstraction_observation, index in args.abstraction_action, bool_save)
        adversaries.append(adversary)
        index += 1

    #Check non-negative train, watch and play
    if args.numTrain < 0 or args.numWatch < 0 or args.numPlay < 0:
        raise Exception("Number of games cannot be negative")
    
    return (environment, player, tuple(adversaries), args.numTrain, args.numWatch, args.numPlay,
            args.outfile_player, args.outfile_adversary, args.outfile_env, 
            args.learn_play, args.disable_player_learn, args.disable_adversary_learn)

def main(args: list[str] | None =  None):
    (environment, player, adversaries, numTrain, numWatch, numPlay,
    outfile_player, outfile_adversary, outfile_env, 
    learn_play, disable_player_learn, disable_adversary_learn) = match_args(parse(args))

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

    environment.enable_rendering()
    #Watching period
    print("Watching...")
    environment.runNumGames((player, *adversaries), numWatch)

    #If outfile specified save player agent.
    if outfile_player: writeToFile(player, outfile_player, 'p') 
    #Swap to playerAgent and set up action space
    temp_player = PlayerAgent(player.obs_abstraction, player.action_abstraction)
    temp_player.set_up(player.action_space, player.observation_space)
    player = temp_player
    
    #If learning is specified as false during play period, disable adversarial learning
    if not learn_play:
        for adversary in adversaries:
            adversary.disableLearning()

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