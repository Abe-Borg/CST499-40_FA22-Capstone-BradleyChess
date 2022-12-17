class Settings:
    """A class to store all settings for BradleyChess."""
    def __init__(self):
        # HYPERPARAMETERS
        # learn_rate and discount factor can be different for each agent. range is 0 to 1 for both parameters
        self.learn_rate = 0.6 # too high num here means too focused on recent knowledge, 
        self.discount_factor = 0.35   # lower number means more opportunistic, but not good long term planning

        self.new_move_pts = 1_000
        self.chance_for_random = 0.10
        self.initial_q_val = 50  # this is about the centipawn score for W on its first move
        self.piece_dev_pts = 50
        self.capture_pts = 100
        self.promotion_Queen_pts = 1_000
        self.checkmate_pts = 1_000_000
        self.mate_score_factor = 1_000
        # end of hyperparameters
        
        self.num_turns_per_player = 50     # turns per player   
        self.stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Desktop\CST499-40_FA22-Capstone-BradleyChess\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
