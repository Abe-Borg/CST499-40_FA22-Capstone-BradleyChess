import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)
        self.agent.Q_table = pd.DataFrame({'W1': [1, 2, 3], 'W2': [4, 5, 6], 'B1': [7, 8, 9], 'B2': [10, 11, 12]}, index=['a', 'b', 'c'])
        self.agent.curr_turn = 'W1'

    def test_get_Q_values(self):
        q_values = self.agent.get_Q_values()
        self.assertEqual(isinstance(q_values, pd.Series), True)
        self.assertEqual(q_values.index.tolist(), ['a', 'b', 'c'])
        self.assertEqual(q_values.dtype, 'int32')
        self.assertEqual(q_values.tolist(), [1, 2, 3])

if __name__ == '__main__':
    unittest.main()