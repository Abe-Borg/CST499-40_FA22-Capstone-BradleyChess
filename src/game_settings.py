import chess.engine
import pandas as pd

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

chess_pgn_file_path_1 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_1.pgn"
chess_pgn_file_path_2 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_2.pgn"
chess_pgn_file_path_3 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_3.pgn"
chess_pgn_file_path_4 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_4.pgn"
chess_pgn_file_path_5 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_5.pgn"
chess_pgn_file_path_6 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_6.pgn"
chess_pgn_file_path_7 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_7.pgn"
chess_pgn_file_path_8 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_8.pgn"
chess_pgn_file_path_9 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_9.pgn"
chess_pgn_file_path_10 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_10.pgn"
chess_pgn_file_path_11 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_11.pgn"
chess_pgn_file_path_12 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_12.pgn"
chess_pgn_file_path_13 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_13.pgn"
chess_pgn_file_path_14 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_14.pgn"
chess_pgn_file_path_15 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_15.pgn"
chess_pgn_file_path_16 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_16.pgn"
chess_pgn_file_path_17 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_17.pgn"
chess_pgn_file_path_18 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_18.pgn"
chess_pgn_file_path_19 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_Part_19.pgn"

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
chess_pd_dataframe_file_path_part_11 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_11.pkl"
chess_pd_dataframe_file_path_part_12 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_12.pkl"
chess_pd_dataframe_file_path_part_13 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_13.pkl"
chess_pd_dataframe_file_path_part_14 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_14.pkl"
chess_pd_dataframe_file_path_part_15 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_15.pkl"
chess_pd_dataframe_file_path_part_16 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_16.pkl"
chess_pd_dataframe_file_path_part_17 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_17.pkl"
chess_pd_dataframe_file_path_part_18 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_18.pkl"
chess_pd_dataframe_file_path_part_19 = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\Chess_Games_Database\Chess_Games_DB_pd_df_Part_19.pkl"
