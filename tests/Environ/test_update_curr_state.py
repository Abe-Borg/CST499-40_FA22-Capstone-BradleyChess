import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_update_curr_state(self):
        # Test the method when the turn index is less than the maximum
        self.environ.turn_index = 0
        self.environ.update_curr_state()
        self.assertEqual(self.environ.turn_index, 1)

        # Test the method when the turn index is equal to the maximum
        self.environ.turn_index = self.environ.settings.max_num_turns_per_player * 2 - 1
        with self.assertRaises(IndexError):
            self.environ.update_curr_state()

if __name__ == '__main__':
    unittest.main()