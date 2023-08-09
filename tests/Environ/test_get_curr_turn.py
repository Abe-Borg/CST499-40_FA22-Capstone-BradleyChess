import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_get_curr_turn(self):
        # Test the method when the turn index is within the range of the turn list
        self.environ.turn_index = 0
        self.assertEqual(self.environ.get_curr_turn(), 'W1')

        # Test the method when the turn index is out of range of the turn list
        self.environ.turn_index = 100
        with self.assertRaises(IndexError):
            self.environ.get_curr_turn()

if __name__ == '__main__':
    unittest.main()