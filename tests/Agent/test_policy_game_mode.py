import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)

    def test_policy_game_mode(self):
        environ_state = {'legal_moves': ['a', 'b', 'c'], 'curr_turn': 'W'}
        action = self.agent.policy_game_mode()
        self.assertEqual(isinstance(action, dict), True)
        self.assertEqual(len(action), 1)
        self.assertEqual(isinstance(list(action.values())[0], str), True)

if __name__ == '__main__':
    unittest.main()