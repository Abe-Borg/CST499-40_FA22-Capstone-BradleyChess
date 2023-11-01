import chess

PRINT_DEBUG: bool = False
PRINT_ERRORS: bool = False

PIECE_VALUES: dict[str, int] = {
        'pawn': 1,
        'knight': 3,
        'bishop': 3,
        'rook': 5,
        'queen': 9
    }

# the following numbers are based on centipawn scores
REWARD_POINTS_FOR_CHESS_MOVES: dict[str, int] = {
        'new_move': 100, # a move that has never been made before
        'piece_development': 300,
        'capture': 100,
        'promotion_queen': 900,
        'mate_score': 1_000
    }

initial_q_val = 50 # this is relevant when first training an agent. SARSA algorithm requires an initial value
training_sample_size = 1 # number of games in database to use for training
agent_vs_agent_num_games = 100 # number of games that agents will play against each other
max_num_turns_per_player = 50
chance_for_random = 0.10 # 10% chance that RL agent selects random chess move
        
# The following values are for the chess engine analysis of moves.
# we only want to look ahead one move, that's the anticipated q value at next state, and next action
# this number has a massive inpact on how long it takes to analyze a position and 
# it doesn't help to go beyond depth_limit 4.
num_moves_to_return = 1
depth_limit = 4
time_limit = None
search_limit = chess.engine.Limit(depth = self.depth_limit, time = self.time_limit)

stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"
bradley_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\bradley_agent_q_table.pkl"
imman_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\imman_agent_q_table.pkl"

helper_methods_debug_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\helper_methods_debug.txt'
helper_methods_errors_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\debug\helper_methods_errors_log.txt'
agent_vs_agent_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\training_results\agent_vs_agent_games.txt'