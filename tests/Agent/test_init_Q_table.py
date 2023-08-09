import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)

    def test_init_Q_table(self):
        q_table = self.agent.init_Q_table(self.chess_data)
        self.assertEqual(isinstance(q_table, pd.DataFrame), True)
        self.assertEqual(q_table.shape, (3, 2))
        self.assertEqual(q_table.dtypes[0], 'int32')

if __name__ == '__main__':
    unittest.main()