import chess

class Settings:
    """
        A class to store all settings for BradleyChess.
        my reason for making this a class is that I want flexibility across all other classes and also the main file.
        If I make it a class, then at any point I can change things. For example, doing it this way, 
        each agent can have its own version of the values in this file, like the learn rate and discount factor.
    """

    def __init__(self):
        # HYPERPARAMETERS, no clue how to fine tune all of these, and the online documentation basically says
        # just try different values and see what happens...
        
        # learn_rate and discount factor can be different for each agent. range is 0 to 1 for both parameters
        self.learn_rate = 0.6 # too high num here means too focused on recent knowledge, 
        self.discount_factor = 0.35   # lower number means more opportunistic, but not good long term planning
        self.training_sample_size = 10_000
        self.agent_vs_agent_num_games = 40_000
        self.num_turns_per_player = 150     # turns per player, most games don't go nearly this long, but the agents do play this long. 


        # the following numbers are based on centipawn scores, but not exactly. for example, the checkmate point value is made up.
        self.new_move_pts = 1_000
        self.chance_for_random = 0.10
        self.initial_q_val = 50  # this is about the centipawn score for W on its first move
        self.piece_dev_pts = 50
        self.capture_pts = 100
        self.promotion_Queen_pts = 1_000
        self.checkmate_pts = 1_000_000
        self.mate_score_factor = 1_000
        
        # The following values are for the ches engine analysis of moves.
        # we only want to look ahead one move, that's the anticipated q value at next state, and next action
        # this number has a massive inpact on how long it takes to analyze a position and it really doesn't help to go beyond 8.
        self.num_moves_to_return = 1
        self.depth_limit = 8
        self.time_limit = None
        self.search_limit = chess.engine.Limit(depth = self.depth_limit, time = self.time_limit)

        self.stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
        self.chess_data_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\chess_data\kaggle_chess_data.pkl"
        self.bradley_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\bradley_agent_q_table.pkl"
        self.imman_agent_q_table_path = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\Q_Tables\imman_agent_q_table.pkl"
        self.initial_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\training_results\initial_training_results.txt'
        self.additional_training_results_filepath = r'C:\Users\Abrah\Dropbox\PC (2)\Desktop\GitHub Repos\CST499-40_FA22-Capstone-BradleyChess\training_results\additional_training_results.txt'
