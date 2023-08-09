import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_load_chessboard(self):
        # Test the method when the chess move is valid
        self.assertTrue(self.environ.load_chessboard('e4'))

        # Test the method when the chess move is invalid
        self.assertFalse(self.environ.load_chessboard('e5'))

if __name__ == '__main__':
    unittest.main()