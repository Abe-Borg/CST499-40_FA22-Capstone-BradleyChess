import pandas as pd
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({'state': ['1', '2', '3'], 'action': ['a', 'b', 'c'], 'reward': [0, 1, 2]})
        self.agent = Agent('W', self.chess_data)

    def test_init(self):
        self.assertEqual(self.agent.color, 'W')
        self.assertEqual(self.agent.chess_data.equals(self.chess_data), True)
        self.assertEqual(isinstance(self.agent.settings, Settings.Settings), True)
        self.assertEqual(self.agent.is_trained, False)
        self.assertEqual(isinstance(self.agent.Q_table, pd.DataFrame), True)

if __name__ == '__main__':
    unittest.main()