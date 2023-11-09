import chess.engine
import pandas as pd

PRINT_DEBUG: bool = True
PRINT_TRAINING_RESULTS = True

PIECE_VALUES: dict[str, int] = {
        'pawn': 1,
        'knight': 3,
        'bishop': 3,
        'rook': 5,
        'queen': 9
    }

# the following numbers are based on centipawn scores
CHESS_MOVE_VALUES: dict[str, int] = {
        'new_move': 100, # a move that has never been made before
        'capture': 150,
        'piece_development': 200,
        'check': 300,
        'promotion': 500,
        'promotion_queen': 900,
        'mate_score': 1_000
    }


training_sample_size = 2 # number of games in database to use for training

max_num_turns_per_player = 50
max_turn_index = max_num_turns_per_player * 2 - 1

initial_q_val = 50 # this is relevant when first training an agent. SARSA algorithm requires an initial value
agent_vs_agent_num_games = 1 # number of games that agents will play against each other
chance_for_random_move = 0.10 # 10% chance that RL agent selects random chess move
        
# The following values are for the chess engine analysis of moves.
# we only want to look ahead one move, that's the anticipated q value at next state, and next action
num_moves_to_return = 1
depth_limit = 2
time_limit = None
search_limit = chess.engine.Limit(depth = depth_limit, time = time_limit)

stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
kaggle_chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"
bradley_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\bradley_agent_q_table.pkl"
imman_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\imman_agent_q_table.pkl"

agent_vs_agent_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\training_results\agent_vs_agent_games.txt'

helper_methods_debug_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\helper_methods_debug_log.txt'
helper_methods_errors_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\helper_methods_errors_log.txt'

agent_debug_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\agent_debug_log.txt'
agent_errors_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\agent_errors_log.txt'

bradley_debug_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\bradley_debug_log.txt'
bradley_errors_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\bradley_errors_log.txt'

initial_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\training_results\initial_training_results.txt'
additional_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\training_results\additional_training_results.txt'

environ_debug_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\environ_debug_log.txt'
environ_errors_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\environ_errors_log.txt'

q_est_log_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\q_est_log.txt'
chess_data_cleaned_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\chess_data\chess_data_cleaned.pkl'