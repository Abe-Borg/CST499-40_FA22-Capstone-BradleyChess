class Settings:
    """A class to store all settings for BradleyChess."""
    def __init__(self):
        self.learn_rate = 0.6 # too high num here means too focused on recent knowledge
        self.discount_factor = 0.35   # lower number means more opportunistic, but not good long term planning
        self.num_columns = 50     # (75 per player)   
        self.new_move_pts = 1_000
        self.chance_for_random = 0.10  # percent
        self.initial_q_val = 50  # this is about the centipawn score for W on its first move
        self.piece_dev_pts = 50
        self.capture_pts = 100
        self.promotion_Queen_pts = 1_000
        self.checkmate_pts = 1_000_000
        self.mate_score_factor = 1_000
        self.stockfish_filepath = r"C:\Users\Abrah\Dropbox\PC (2)\Downloads\stockfish_15_win_x64_avx2\stockfish_15_win_x64_avx2\stockfish_15_x64_avx2.exe"
