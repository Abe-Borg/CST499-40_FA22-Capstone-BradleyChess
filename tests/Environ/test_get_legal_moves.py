import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_get_legal_moves(self):
        # Test the method when there are legal moves available
        self.environ.load_chessboard('e4')
        legal_moves = self.environ.get_legal_moves()
        self.assertIn('Nf3', legal_moves)
        self.assertIn('Nc3', legal_moves)

        # Test the method when there are no legal moves available
        self.environ.load_chessboard('k7/8/8/8/8/8/8/K7 w - - 0 1')
        legal_moves = self.environ.get_legal_moves()
        self.assertEqual(legal_moves, ["legal moves list is empty"])

if __name__ == '__main__':
    unittest.main()