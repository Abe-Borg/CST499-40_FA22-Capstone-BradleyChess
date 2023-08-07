import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_game_on_true(self):
        # Test that the method returns True when the game is ongoing
        result = self.bradley.game_on()
        self.assertTrue(result)

    def test_game_on_false(self):
        # Test that the method returns False when the game is over
        self.bradley.environ.board.set_result('1-0')
        result = self.bradley.game_on()
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()