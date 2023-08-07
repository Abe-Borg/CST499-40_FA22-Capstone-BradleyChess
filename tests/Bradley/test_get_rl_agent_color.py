import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_rl_agent_color(self):
        # Test that the method returns a string
        result = self.bradley.get_rl_agent_color()
        self.assertIsInstance(result, str)

        # Test that the method returns either 'W' or 'B'
        self.assertIn(result, ['W', 'B'])

if __name__ == '__main__':
    unittest.main()