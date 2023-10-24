import unittest
import pandas as pd
import imman
import os
from my_module import bootstrap_agent

class TestBootstrapAgent(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({
            'white': ['Magnus Carlsen', 'Fabiano Caruana', 'Wesley So'],
            'black': ['Hikaru Nakamura', 'Levon Aronian', 'Maxime Vachier-Lagrave'],
            'result': ['1-0', '0-1', '1/2-1/2']
        })
        self.bubs = imman.Bradley(self.chess_data)
        self.q_table_path = 'test_q_table.pkl'
        self.bubs.W_rl_agent.train()
        self.bubs.W_rl_agent.save_q_table(self.q_table_path)

    def tearDown(self):
        if os.path.exists(self.q_table_path):
            os.remove(self.q_table_path)

    def test_bootstrap_agent(self):
        # Test the function with valid inputs
        bootstrap_agent(self.bubs, 'W', self.q_table_path)
        self.assertTrue(self.bubs.W_rl_agent.is_trained)

        # Test the function with invalid rl_agent_color
        with self.assertRaises(ValueError):
            bootstrap_agent(self.bubs, 'invalid_color', self.q_table_path)

if __name__ == '__main__':
    unittest.main()