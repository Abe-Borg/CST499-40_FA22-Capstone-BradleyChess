import Bradley as imman
import pandas as pd
import random
# import logging
# import log_config

# logger = logging.getLogger(__name__)

print_debug_statements_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\print_statements.txt'
print_statements_debug = open(print_debug_statements_filepath, 'a')
PRINT_RESULTS_DEBUG: bool = True
# print_statements_debug.write(f'\n\n Start of {game_num_str} training\n\n')

def init_bradley(chess_data: pd.DataFrame) -> imman.Bradley:
    """Initializes a Bradley object with the given chess data.

    The Bradley object is used to calculate the Bradley-Terry scores for each player in the chess data.

    Args:
        chess_data (pd.DataFrame): A Pandas DataFrame containing the chess data.

    Returns:
        imman.Bradley: An object of the Bradley class.

    """
    bubs = imman.Bradley(chess_data)
    return bubs
### end of init_bradley

def play_game(bubs: imman.Bradley, rl_agent_color: str) -> None:
    """Plays a game of chess against a human player using the terminal.

    The function alternates between the human player and the RL agent's turns until the game is over. The function prints the current turn number and prompts the human player to enter their chess move. If it is the RL agent's turn, the function calls the `rl_agent_selects_chess_move` method to select a move. The function also prints the move played by the RL agent. If the human player enters an invalid move, the function prompts them to enter a valid move. The function prints the game outcome and the reason for the game termination.

    Args:
        bubs (imman.Bradley): An object of the Bradley class.
        rl_agent_color (str): A string representing the color of the RL agent, either 'W' or 'B'.

    Returns:
        None

    Raises:
        ValueError: If `rl_agent_color` is not 'W' or 'B'.

    """
    if rl_agent_color not in ['W', 'B']:
        # logger.warning(f"invalid input: {rl_agent_color}")
        raise ValueError("rl_agent_color must be 'W' or 'B'")
        
    W_turn: bool = True

    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent
    
    while bubs.game_on():
        if W_turn:
            player_turn = 'W'
        else:
            player_turn = 'B'
        
        print(f'\nCurrent turn is :  {bubs.get_curr_turn()}\n')

        if rl_agent.color == player_turn:
            print('=== RL AGENT\'S TURN ===\n')
            try:
                chess_move: dict[str] = bubs.rl_agent_selects_chess_move(rl_agent.color)
                chess_move_str = chess_move['chess_move_str']
                print(f'RL agent played {chess_move_str}\n')
            except Exception as e:
                print(f'An error occurred: {e}')
                # logger.error(f'Error occurred during RL agent turn: {e}')
                break
        else:
            print('=== OPPONENT\' TURN ===')
            try:
                chess_move = str(input('Enter chess move: '))

                if chess_move == 'q':
                    break
                elif chess_move == 'pop':
                    try:
                        bubs.environ.undo_move()
                        W_turn = not W_turn
                        continue
                    except Exception as e:
                        print(f'An error occurred: {e}')
                        # logger.error(f'Error occurred while undoing move: {e}')
                        continue
                elif chess_move == 'pop 2x':
                    try:
                        bubs.environ.undo_move()
                        W_turn = not W_turn
                        bubs.environ.undo_move()
                        W_turn = not W_turn
                        continue
                    except Exception as e:
                        print(f'An error occurred: {e}')
                        # logger.error(f'Error occurred while undoing move: {e}')
                        continue
                else:
                    while not bubs.recv_opp_move(chess_move):
                        print('Invalid move, try again.')
                        chess_move = str(input('Enter chess move: '))
            except Exception as e:
                print(f'An error occurred: {e}')
                # logger.error(f'Error occured durring humans turn: {e}')
                break
            
            print('\n')
                
        W_turn = not W_turn    
        # end while loop
    
    print(f'Game is over, result is: {bubs.get_game_outcome()}')
    print(f'The game ended because of: {bubs.get_game_termination_reason()}')

    try:
        bubs.reset_environ()
    except Exception as e:
        print(f'An error occurred: {e}')
        # logger.error(f'Error occurred while resetting game environment: {e}')
### end of play_game

def agent_vs_agent(bubs: imman.Bradley) -> None:
    """Play two trained agents against each other.

    This function alternates between two trained agents' turns until the game is over. 
    The function prints the current turn number and the move played by each agent. 
    The function also prints the current state of the chessboard after each move. 
    Once the game is over, the function prints the final state of the chessboard, 
    the game outcome, and the reason for the game termination.

    Args:
        bubs (imman.Bradley): An object of the `Bradley` class representing the chess game environment.

    Returns:
        None

    Raises:
        None
    """    
    while bubs.game_on():        
        # bubs's turn
        print(f'\nCurrent turn: {bubs.get_curr_turn()}')
        chess_move_bubs: dict[str] = bubs.rl_agent_selects_chess_move('W')
        bubs_chess_move_str: str = chess_move_bubs['chess_move_str']
        print(f'Bubs played {bubs_chess_move_str}\n')

        # imman's turn, check for end of game again, since the game could have ended after W's move.
        if bubs.game_on():
            print(f'Current turn:  {turn_num}')
            chess_move_imman: dict[str] = bubs.rl_agent_selects_chess_move('B')
            imman_chess_move_str: str = chess_move_imman['chess_move_str']
            print(f'Imman played {imman_chess_move_str}\n')

        print(bubs.environ.board)
    
    print('Game is over, chessboard looks like this:\n')
    print(bubs.environ.board)
    print('\n\n')
    print(f'Game result is: {bubs.get_game_outcome()}')
    print(f'Game ended because of: {bubs.get_game_termination_reason()}')
    bubs.reset_environ()
### end of agent_vs_agent

def pikl_q_table(bubs: imman.Bradley, rl_agent_color: str, q_table_path: str) -> None:
    """Save the Q-table of a trained RL agent to a file.

    This function saves the Q-table of a trained RL agent to a file in the specified path. The function takes three arguments: an object of the `Bradley` class representing the chess game environment, a string representing the color of the RL agent ('W' for white or 'B' for black), and a string representing the path to the output file. The function selects the Q-table of the specified RL agent based on the `rl_agent_color` argument 
    and saves it to the specified file using the `to_pickle` method of the Q-table object.

    Args:
        bubs (imman.Bradley): An object of the `Bradley` class representing the chess game environment.
        rl_agent_color (str): A string representing the color of the RL agent ('W' for white or 'B' for black).
        q_table_path (str): A string representing the path to the output file.

    Returns:
        None

    Raises:
        None
    """
    if rl_agent_color == 'W':
        rl_agent = bubs.W_rl_agent
    else:
        rl_agent = bubs.B_rl_agent

    rl_agent.Q_table.to_pickle(q_table_path, compression = 'zip')
### end of pikl_Q_table

def bootstrap_agent(bubs: imman.Bradley, rl_agent_color: str, existing_q_table_path: str) -> None:
    """Assign an agent's Q-table to an existing Q-table.

    This function assigns an agent's Q-table to an existing Q-table stored in a file. 
    The function takes three arguments: an object of the `Bradley` class representing 
    the chess game environment, a string representing the color of the RL agent 
    ('W' for white or 'B' for black), and a string representing the path to the existing Q-table file. 
    The function selects the Q-table of the specified RL agent based on the `rl_agent_color` 
    argument and assigns it the Q-table stored in the specified file using the 
    `read_pickle` method of the Pandas library. 

    Args:
        bubs (imman.Bradley): An object of the `Bradley` class representing the chess game environment.
        rl_agent_color (str): A string representing the color of the RL agent ('W' for white or 'B' for black).
        existing_q_table_path (str): A string representing the path to the existing Q-table file.

    Returns:
        None

    Raises:
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

    This function takes a probability value as an argument, which should be between 
    0 and 1 (inclusive). The function generates a random number between 0 and 1 
    using the `random` module of the Python standard library. 
    If the randomly generated number is less than the probability value, 
    the function returns 1. Otherwise, the function returns 0.

    Args:
        probability (float): A float representing the probability of generating a 1.

    Returns:
        int: A random integer value of either 0 or 1.

    Raises:
        None
    """
    if random.random() < probability:
        return 1
    else:
        return 0