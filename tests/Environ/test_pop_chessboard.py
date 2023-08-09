import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_pop_chessboard(self):
        # Test the method when the move stack is not empty
        self.environ.load_chessboard('e4')
        self.environ.pop_chessboard()
        self.assertEqual(str(self.environ.board), 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1')

        # Test the method when the move stack is empty
        with self.assertRaises(IndexError):
            self.environ.pop_chessboard()

if __name__ == '__main__':
    unittest.main()