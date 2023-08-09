import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ
from chess import Move

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_load_chessboard_for_Q_est(self):
        # Test the method when the analysis results contain a valid anticipated chess move
        analysis_results = {'mate_score': None, 'centipawn_score': None, 'anticipated_next_move': Move.from_uci('e2e4')}
        self.assertIsNone(self.environ.load_chessboard_for_Q_est(analysis_results))

        # Test the method when the analysis results contain an invalid anticipated chess move
        analysis_results = {'mate_score': None, 'centipawn_score': None, 'anticipated_next_move': 'invalid_move'}
        with self.assertRaises(ValueError):
            self.environ.load_chessboard_for_Q_est(analysis_results)

if __name__ == '__main__':
    unittest.main()