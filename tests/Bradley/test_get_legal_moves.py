import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_legal_moves(self):
        # Test that the method returns a list of strings
        result = self.bradley.get_legal_moves()
        self.assertIsInstance(result, list)
        for move in result:
            self.assertIsInstance(move, str)

if __name__ == '__main__':
    unittest.main()