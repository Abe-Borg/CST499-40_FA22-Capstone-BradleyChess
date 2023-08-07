import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_curr_turn(self):
        # Test that the method returns a string
        result = self.bradley.get_curr_turn()
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()