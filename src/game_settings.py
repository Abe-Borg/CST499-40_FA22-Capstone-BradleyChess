import chess.engine
import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)

PRINT_DEBUG: bool = True
PRINT_TRAINING_RESULTS = False
PRINT_Q_EST = True

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


training_sample_size = 1_000 # number of games in database to use for training

max_num_turns_per_player = 200
max_turn_index = max_num_turns_per_player * 2 - 1

initial_q_val = 50 # this is relevant when first training an agent. SARSA algorithm requires an initial value
agent_vs_agent_num_games = 1 # number of games that agents will play against each other
chance_for_random_move = 0.10 # 10% chance that RL agent selects random chess move
        
# The following values are for the chess engine analysis of moves.
# we only want to look ahead one move, that's the anticipated q value at next state, and next action
num_moves_to_return = 1
depth_limit = 1
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

chess_pd_dataframe_file_path_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_1.pkl"
chess_pd_dataframe_file_path_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_2.pkl"
chess_pd_dataframe_file_path_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_3.pkl"
chess_pd_dataframe_file_path_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_4.pkl"
chess_pd_dataframe_file_path_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_5.pkl"
chess_pd_dataframe_file_path_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_6.pkl"
chess_pd_dataframe_file_path_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_7.pkl"
chess_pd_dataframe_file_path_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_8.pkl"
chess_pd_dataframe_file_path_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_9.pkl"
chess_pd_dataframe_file_path_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_10.pkl"

chess_pd_dataframe_file_path_part_11_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_1.pkl"
chess_pd_dataframe_file_path_part_11_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_2.pkl"
chess_pd_dataframe_file_path_part_11_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_3.pkl"
chess_pd_dataframe_file_path_part_11_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_4.pkl"
chess_pd_dataframe_file_path_part_11_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_5.pkl"
chess_pd_dataframe_file_path_part_11_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_6.pkl"
chess_pd_dataframe_file_path_part_11_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_7.pkl"
chess_pd_dataframe_file_path_part_11_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_8.pkl"
chess_pd_dataframe_file_path_part_11_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_9.pkl"
chess_pd_dataframe_file_path_part_11_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11_Part_10.pkl"

chess_pd_dataframe_file_path_part_12_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_1.pkl"
chess_pd_dataframe_file_path_part_12_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_2.pkl"
chess_pd_dataframe_file_path_part_12_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_3.pkl"
chess_pd_dataframe_file_path_part_12_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_4.pkl"
chess_pd_dataframe_file_path_part_12_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_5.pkl"
chess_pd_dataframe_file_path_part_12_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_6.pkl"
chess_pd_dataframe_file_path_part_12_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_7.pkl"
chess_pd_dataframe_file_path_part_12_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_8.pkl"
chess_pd_dataframe_file_path_part_12_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_9.pkl"
chess_pd_dataframe_file_path_part_12_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12_Part_10.pkl"

chess_pd_dataframe_file_path_part_13_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_1.pkl"
chess_pd_dataframe_file_path_part_13_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_2.pkl"
chess_pd_dataframe_file_path_part_13_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_3.pkl"
chess_pd_dataframe_file_path_part_13_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_4.pkl"
chess_pd_dataframe_file_path_part_13_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_5.pkl"
chess_pd_dataframe_file_path_part_13_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_6.pkl"
chess_pd_dataframe_file_path_part_13_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_7.pkl"
chess_pd_dataframe_file_path_part_13_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_8.pkl"
chess_pd_dataframe_file_path_part_13_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_9.pkl"
chess_pd_dataframe_file_path_part_13_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13_Part_10.pkl"

chess_pd_dataframe_file_path_part_14_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_1.pkl"
chess_pd_dataframe_file_path_part_14_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_2.pkl"
chess_pd_dataframe_file_path_part_14_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_3.pkl"
chess_pd_dataframe_file_path_part_14_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_4.pkl"
chess_pd_dataframe_file_path_part_14_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_5.pkl"
chess_pd_dataframe_file_path_part_14_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_6.pkl"
chess_pd_dataframe_file_path_part_14_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_7.pkl"
chess_pd_dataframe_file_path_part_14_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_8.pkl"
chess_pd_dataframe_file_path_part_14_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_9.pkl"
chess_pd_dataframe_file_path_part_14_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14_Part_10.pkl"

chess_pd_dataframe_file_path_part_15_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_1.pkl"
chess_pd_dataframe_file_path_part_15_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_2.pkl"
chess_pd_dataframe_file_path_part_15_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_3.pkl"
chess_pd_dataframe_file_path_part_15_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_4.pkl"
chess_pd_dataframe_file_path_part_15_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_5.pkl"
chess_pd_dataframe_file_path_part_15_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_6.pkl"
chess_pd_dataframe_file_path_part_15_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_7.pkl"
chess_pd_dataframe_file_path_part_15_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_8.pkl"
chess_pd_dataframe_file_path_part_15_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_9.pkl"
chess_pd_dataframe_file_path_part_15_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15_Part_10.pkl"

chess_pd_dataframe_file_path_part_16_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_1.pkl"
chess_pd_dataframe_file_path_part_16_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_2.pkl"
chess_pd_dataframe_file_path_part_16_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_3.pkl"
chess_pd_dataframe_file_path_part_16_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_4.pkl"
chess_pd_dataframe_file_path_part_16_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_5.pkl"
chess_pd_dataframe_file_path_part_16_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_6.pkl"
chess_pd_dataframe_file_path_part_16_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_7.pkl"
chess_pd_dataframe_file_path_part_16_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_8.pkl"
chess_pd_dataframe_file_path_part_16_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_9.pkl"
chess_pd_dataframe_file_path_part_16_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16_Part_10.pkl"

chess_pd_dataframe_file_path_part_17_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_1.pkl"
chess_pd_dataframe_file_path_part_17_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_2.pkl"
chess_pd_dataframe_file_path_part_17_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_3.pkl"
chess_pd_dataframe_file_path_part_17_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_4.pkl"
chess_pd_dataframe_file_path_part_17_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_5.pkl"
chess_pd_dataframe_file_path_part_17_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_6.pkl"
chess_pd_dataframe_file_path_part_17_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_7.pkl"
chess_pd_dataframe_file_path_part_17_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_8.pkl"
chess_pd_dataframe_file_path_part_17_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_9.pkl"
chess_pd_dataframe_file_path_part_17_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17_Part_10.pkl"

chess_pd_dataframe_file_path_part_18_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_1.pkl"
chess_pd_dataframe_file_path_part_18_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_2.pkl"
chess_pd_dataframe_file_path_part_18_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_3.pkl"
chess_pd_dataframe_file_path_part_18_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_4.pkl"
chess_pd_dataframe_file_path_part_18_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_5.pkl"
chess_pd_dataframe_file_path_part_18_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_6.pkl"
chess_pd_dataframe_file_path_part_18_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_7.pkl"
chess_pd_dataframe_file_path_part_18_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_8.pkl"
chess_pd_dataframe_file_path_part_18_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_9.pkl"
chess_pd_dataframe_file_path_part_18_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18_Part_10.pkl"

chess_pd_dataframe_file_path_part_19_part_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_1.pkl"
chess_pd_dataframe_file_path_part_19_part_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_2.pkl"
chess_pd_dataframe_file_path_part_19_part_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_3.pkl"
chess_pd_dataframe_file_path_part_19_part_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_4.pkl"
chess_pd_dataframe_file_path_part_19_part_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_5.pkl"
chess_pd_dataframe_file_path_part_19_part_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_6.pkl"
chess_pd_dataframe_file_path_part_19_part_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_7.pkl"
chess_pd_dataframe_file_path_part_19_part_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_8.pkl"
chess_pd_dataframe_file_path_part_19_part_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_9.pkl"
chess_pd_dataframe_file_path_part_19_part_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19_Part_10.pkl"

