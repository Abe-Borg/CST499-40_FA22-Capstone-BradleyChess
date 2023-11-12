import Bradley
import pandas as pd
import game_settings
# import logging
# import log_config
# logger = logging.getLogger(__name__)

def play_game(bubs: Bradley.Bradley, rl_agent_color: str) -> None:
    def handle_move(player_color):
        if player_color == rl_agent.color:
            print('=== RL AGENT\'S TURN ===\n')
            return bubs.rl_agent_selects_chess_move(player_color)
        else:
            print('=== OPPONENT\'S TURN ===')
            move = input('Enter chess move: ')
            while not bubs.receive_opp_move(move):
                print('Invalid move, try again.')
                move = input('Enter chess move: ')
            return move

    def log_error(e):
        errors_file.write(f'An error occurred: {e}\n')

    rl_agent = bubs.W_rl_agent if rl_agent_color == 'W' else bubs.B_rl_agent
    player_turn = 'W'

    with open(game_settings.helper_methods_errors_filepath, 'a') as errors_file:
        while bubs.is_game_on():
            try:
                print(f'\nCurrent turn is :  {bubs.get_curr_turn()}\n')
                chess_move = handle_move(player_turn)
                print(f'{player_turn} played {chess_move}\n')
            except Exception as e:
                log_error(e)

            player_turn = 'B' if player_turn == 'W' else 'W'

        print(f'Game is over, result is: {bubs.get_game_outcome()}')
        print(f'The game ended because of: {bubs.get_game_termination_reason()}')
        bubs.reset_environ()
### end of play_game

def agent_vs_agent(bubs: Bradley.Bradley) -> None:
    """Play two trained agents against each other.
    Args:
        bubs: An object of the `Bradley` class representing the chess game environment.
    """    
    agent_vs_agent_file = open(game_settings.agent_vs_agent_filepath, 'a')

    while bubs.is_game_on():
        try:        
            agent_vs_agent_file.write(f'\nCurrent turn: {bubs.get_curr_turn()}')
        except Exception as e:
            agent_vs_agent_file.write(f'An error occurred: {e}')
            raise Exception from e

        try:
            chess_move_bubs: str = bubs.rl_agent_selects_chess_move('W')
        except Exception as e:
            agent_vs_agent_file.write(f'An error occurred: {e}')
            raise Exception from e

        agent_vs_agent_file.write(f'Bubs played {chess_move_bubs}\n')

        # imman's turn, check for end of game again, since the game could have ended after W's move.
        if bubs.is_game_on():
            agent_vs_agent_file.write(f'Current turn:  {turn_num}')
            try:
                chess_move_imman: str = bubs.rl_agent_selects_chess_move('B')
            except Exception as e:
                agent_vs_agent_file.write(f'An error occurred: {e}')
                raise Exception from e
        
        agent_vs_agent_file.write(f'Imman played {chess_move_imman}\n')
            
    agent_vs_agent_file.write('Game is over, chessboard looks like this:\n')
    agent_vs_agent_file.write(bubs.get_chessboard())
    agent_vs_agent_file.write('\n\n')
    try:
        agent_vs_agent_file.write(f'Game result is: {bubs.get_game_outcome()}')
    except Exception as e:
        agent_vs_agent_file.write(f'An error occurred at get_game_outcome: {e}')

    try:
        agent_vs_agent_file.write(f'Game ended because of: {bubs.get_game_termination_reason()}')
    except Exception as e:
        agent_vs_agent_file.write(f'An error occurred at get_game_termination_reason: {e}')

    bubs.reset_environ()
    agent_vs_agent_file.close()
### end of agent_vs_agent

def pikl_q_table(bubs: Bradley.Bradley, rl_agent_color: str, q_table_path: str) -> None:
    """Save the Q-table of a trained RL agent to a file.
    Args:
        bubs (Bradley.Bradley): An object of the `Bradley` class representing the chess game environment.
        rl_agent_color (str): A string representing the color of the RL agent ('W' for white or 'B' for black).
        q_table_path (str): A string representing the path to the output file.
    Returns:
        None
    """
    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent

    rl_agent.Q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_Q_table

def bootstrap_agent(bubs: Bradley.Bradley, rl_agent_color: str, existing_q_table_path: str) -> None:
    """Assign an agent's Q-table to an existing Q-table.
    Args:
        bubs (imman.Bradley): An object of the `Bradley` class representing the chess game environment.
        rl_agent_color (str): A string representing the color of the RL agent ('W' for white or 'B' for black).
        existing_q_table_path (str): A string representing the path to the existing Q-table file.
    Returns:
        None
    """
    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent

    rl_agent.Q_table = pd.read_pickle(existing_q_table_path, compression = 'zip')
    rl_agent.is_trained = True
### end of bootstrap_agent

def get_number_with_probability(probability: float) -> int:
    """Generate a random number with a given probability.
    Args:
        probability (float): A float representing the probability of generating a 1.
    Returns:
        int: A random integer value of either 0 or 1.
    """
    if game_settings.PRINT_DEBUG:
        debug_file = open(game_settings.helper_methods_debug_filepath, 'a')
        debug_file.write('========== Hello from Helper Methods get_number_with_probability ==========\n')
        debug_file.write(f'probability: {probability}\n')

    if random.random() < probability:
        if game_settings.PRINT_DEBUG:
            debug_file.write("Random number is less than probability, returning 1\n")
            debug_file.write("========== Bye from Helper Methods get_number_with_probability ==========\n\n\n")
        debug_file.close()
        return 1
    else:
        if game_settings.PRINT_DEBUG:
            debug_file.write("Random number is >= than probability, returning 0\n")
            debug_file.write("========== Bye from Helper Methods get_number_with_probability ==========\n\n\n")
        debug_file.close()
        return 0
### end of get_number_with_probability