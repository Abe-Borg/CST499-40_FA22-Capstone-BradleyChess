import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_rl_agent_selects_chess_move(self):
        # Test that the method returns a dictionary containing a chess move string
        result = self.bradley.rl_agent_selects_chess_move('W')
        self.assertIsInstance(result, dict)
        self.assertIn('chess_move_str', result)

if __name__ == '__main__':
    unittest.main()


# We then define a test method test_rl_agent_selects_chess_move. In this test method, we call the rl_agent_selects_chess_move method with the color 
# 'W' and assert that the method returns a dictionary containing a chess move string. We use the isinstance method to check that the 
# return value is a dictionary, and the assertIn method to check that the dictionary contains the key 'chess_move_str'.

# Finally, we use the unittest.main() method to run the tests. When you run this test file, you should see output indicating whether the tests passed or failed.