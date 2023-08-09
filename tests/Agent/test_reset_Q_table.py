import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)
        self.agent.Q_table = pd.DataFrame({'W1': [1, 2, 3], 'W2': [4, 5, 6], 'B1': [7, 8, 9], 'B2': [10, 11, 12]}, index=['a', 'b', 'c'])

    def test_reset_Q_table(self):
        self.agent.reset_Q_table()
        self.assertEqual(self.agent.Q_table.values.sum(), 0)

if __name__ == '__main__':
    unittest.main()