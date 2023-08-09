import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Environ

class TestEnviron(unittest.TestCase):
    def setUp(self):
        self.environ = Environ()

    def test_get_curr_state(self):
        curr_state = self.environ.get_curr_state()
        self.assertEqual(isinstance(curr_state, dict), True)
        self.assertEqual(len(curr_state), 3)
        self.assertEqual('turn_index' in curr_state, True)
        self.assertEqual('curr_turn' in curr_state, True)
        self.assertEqual('legal_moves' in curr_state, True)

if __name__ == '__main__':
    unittest.main()