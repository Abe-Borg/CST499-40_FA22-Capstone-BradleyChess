import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_game_termination_reason(self):
        # Test that the method returns a string
        result = self.bradley.get_game_termination_reason()
        self.assertIsInstance(result, str)

        # Test that the string is one of the valid termination reasons
        valid_termination_reasons = ['termination.CHECKMATE', 'termination.STALEMATE', 'termination.INSUFFICIENT_MATERIAL', 'termination.FIFTY_MOVES', 'termination.THREEFOLD_REPETITION', 'termination.AGREEMENT']
        self.assertIn(result, valid_termination_reasons)

if __name__ == '__main__':
    unittest.main()