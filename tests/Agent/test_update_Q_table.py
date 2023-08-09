import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)
        self.agent.Q_table = pd.DataFrame({'W1': [0, 0, 0], 'W2': [0, 0, 0], 'B1': [0, 0, 0], 'B2': [0, 0, 0]}, index=['a', 'b', 'c'])

    def test_update_Q_table(self):
        self.agent.update_Q_table(['d4', 'e4'])
        self.assertEqual(self.agent.Q_table.shape, (5, 4))
        self.assertEqual(self.agent.Q_table.index.tolist(), ['a', 'b', 'c', 'd4', 'e4'])

    def test_update_Q_table_empty(self):
        result = self.agent.update_Q_table([])
        self.assertEqual(result, ["new_chess_moves list is empty"])

if __name__ == '__main__':
    unittest.main()