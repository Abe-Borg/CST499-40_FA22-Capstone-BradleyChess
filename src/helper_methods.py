import Bradley
import pandas as pd
import game_settings
# import logging
# import log_config
# logger = logging.getLogger(__name__)

def play_game(bubs: Bradley.Bradley, rl_agent_color: str) -> None:
    """Plays a game of chess against a human player using the terminal.
    Args:
        bubs: An object of the Bradley class.
        rl_agent_color : A string representing the color of the RL agent, either 'W' or 'B'.
    Returns:
        None
    """    
    errors_file = open(game_settings.helper_methods_errors_filepath, 'a')

    is_W_turn: bool = True

    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent
    
    while bubs.is_game_on():
        if is_W_turn:
            player_turn = 'W'
        else:
            player_turn = 'B'
        
        try:
            print(f'\nCurrent turn is :  {bubs.get_curr_turn()}\n')
        except Exception as e:
            errors_file.write(f'An error occurred: {e}')
            # logger.error(f'Error occurred while getting current turn: {e}')
        
        if rl_agent.color == player_turn:
            print('=== RL AGENT\'S TURN ===\n')
            try:
                chess_move: str = bubs.rl_agent_selects_chess_move(rl_agent.color) 
                print(f'RL agent played {chess_move}\n')
            except Exception as e:
                errors_file.write(f'An error occurred: {e}')
                # logger.error(f'Error occurred during RL agent turn: {e}')
        else:
            print('=== OPPONENT\' TURN ===')
            try:
                chess_move = str(input('Enter chess move: '))
                while not bubs.receive_opp_move(chess_move):
                    print('Invalid move, try again.')
                    chess_move = str(input('Enter chess move: '))
            except Exception as e:
                error_file.write(f'An error occurred: {e}')
                # logger.error(f'Error occured durring humans turn: {e}')
            print('\n')

        is_W_turn = not is_W_turn    
        # end single turn where a turn is W and B moving once each
    
    try:
        print(f'Game is over, result is: {bubs.get_game_outcome()}')
    except Exception as e:
        error_file.write(f'An error occurred while getting game outcome: {e}')
        # logger.error(f'Error occurred while getting game outcome: {e}')

    try:
        print(f'The game ended because of: {bubs.get_game_termination_reason()}')
    except Exception as e:
        error_file.write(f'An error occurred while getting game termination reason: {e}')
        # logger.error(f'Error occurred while getting game termination reason: {e}')
        #     
    try:
        bubs.reset_environ()
    except Exception as e:
        error_file.write(f'An error occurred while resetting environ: {e}')
        # logger.error(f'Error occurred while resetting game environment: {e}')

    errors_file.close()
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
        
        
def clean_chess_data(chess_data: pd.DataFrame, chess_data_cleaned_filepath: str) -> pd.DataFrame:
    # Calculate the number of moves for White and Black based on 'Num Moves'
    # + 1 accounts for an odd number of Num Moves
    chess_data['WhiteMoves'] = (chess_data['Num Moves'] + 1) // 2
    chess_data['BlackMoves'] = chess_data['Num Moves'] // 2
    
    # Generate column names for all possible moves
    max_moves = (chess_data['Num Moves'].max() + 1) // 2
    white_move_cols = ['W' + str(i) for i in range(1, max_moves + 1)]
    black_move_cols = ['B' + str(i) for i in range(1, max_moves)]
    
    # Create a mask for non-empty moves for White and Black
    white_moves_mask = chess_data[white_move_cols].apply(lambda x: x.str.strip().astype(bool), axis=0)
    black_moves_mask = chess_data[black_move_cols].apply(lambda x: x.str.strip().astype(bool), axis=0)
    
    # Filter out rows where the number of non-empty moves is less than the required moves
    white_valid_moves = (white_moves_mask.sum(axis=1) >= chess_data['WhiteMoves'])
    black_valid_moves = (black_moves_mask.sum(axis=1) >= chess_data['BlackMoves'])
    
    valid_games_mask = white_valid_moves & black_valid_moves

    # Drop the auxiliary columns
    chess_data.drop(['WhiteMoves', 'BlackMoves'], axis=1, inplace=True)
    
    # Apply the mask to the DataFrame to filter out games with any empty moves
    chess_data_cleaned = chess_data[valid_games_mask]

    chess_data_cleaned.to_pickle(chess_data_cleaned_filepath, compression = 'zip')