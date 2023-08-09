import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)
        self.agent.Q_table = pd.DataFrame({'W1': [0, 0, 0], 'W2': [0, 0, 0], 'B1': [0, 0, 0], 'B2': [0, 0, 0]}, index=['a', 'b', 'c'])

    def test_change_Q_table_pts(self):
        self.agent.change_Q_table_pts('a', 'W1', 10)
        self.assertEqual(self.agent.Q_table.at['a', 'W1'], 10)

    def test_change_Q_table_pts_negative(self):
        self.agent.change_Q_table_pts('b', 'B2', -5)
        self.assertEqual(self.agent.Q_table.at['b', 'B2'], -5)

if __name__ == '__main__':
    unittest.main()