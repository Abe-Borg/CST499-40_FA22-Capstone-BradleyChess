import unittest
import chess
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_chessboard(self):
        # Test that the method returns a chess.Board instance
        result = self.bradley.get_chessboard()
        self.assertIsInstance(result, chess.Board)

        # Test that the chess.Board instance is in the starting position
        starting_position = chess.Board().fen()
        self.assertEqual(result.fen(), starting_position)

if __name__ == '__main__':
    unittest.main()