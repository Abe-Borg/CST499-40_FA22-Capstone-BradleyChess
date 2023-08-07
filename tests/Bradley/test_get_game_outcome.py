import unittest
import chess
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_game_outcome(self):
        # Test that the method returns either a chess.Outcome instance or a string
        result = self.bradley.get_game_outcome()
        self.assertIsInstance(result, (chess.Outcome, str))

        # Test that the chess.Outcome instance has a result() method
        if isinstance(result, chess.Outcome):
            self.assertTrue(hasattr(result, 'result'))

if __name__ == '__main__':
    unittest.main()