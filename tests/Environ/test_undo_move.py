import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_undo_move(self):
        # Test the method when the move stack is not empty
        self.environ.load_chessboard('e4')
        self.environ.undo_move()
        self.assertEqual(str(self.environ.board), 'rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1')
        self.assertEqual(self.environ.get_curr_turn(), 'W1')

        # Test the method when the move stack is empty
        with self.assertRaises(IndexError):
            self.environ.undo_move()

if __name__ == '__main__':
    unittest.main()