import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_reset_environ(self):
        # Test the method when the Environ object is not reset
        self.environ.load_chessboard('e4')
        self.environ.reset_environ()
        self.assertEqual(str(self.environ.board), 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.assertEqual(self.environ.turn_index, 0)

if __name__ == '__main__':
    unittest.main()