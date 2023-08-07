import unittest
import chess
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_analyze_board_state(self):
        # Test that the method returns a dictionary
        board = chess.Board()
        result = self.bradley.analyze_board_state(board)
        self.assertIsInstance(result, dict)

        # Test that the dictionary contains the expected keys
        expected_keys = ['mate_score', 'centipawn_score']
        self.assertCountEqual(list(result.keys()), expected_keys)

        # Test that the mate score is an integer or None
        mate_score = result['mate_score']
        self.assertIsInstance(mate_score, (int, type(None)))

        # Test that the centipawn score is an integer
        centipawn_score = result['centipawn_score']
        self.assertIsInstance(centipawn_score, int)

        # Test that the anticipated next move is included if is_for_est_Qval_analysis is True
        result = self.bradley.analyze_board_state(board, is_for_est_Qval_analysis=True)
        self.assertIn('anticipated_next_move', result)

if __name__ == '__main__':
    unittest.main()