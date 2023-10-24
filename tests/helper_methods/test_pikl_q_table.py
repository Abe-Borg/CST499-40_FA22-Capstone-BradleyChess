import unittest
import pandas as pd
import imman
import os
from my_module import pikl_q_table

class TestPiklQTable(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({
            'white': ['Magnus Carlsen', 'Fabiano Caruana', 'Wesley So'],
            'black': ['Hikaru Nakamura', 'Levon Aronian', 'Maxime Vachier-Lagrave'],
            'result': ['1-0', '0-1', '1/2-1/2']
        })
        self.bubs = imman.Bradley(self.chess_data)
        self.q_table_path = 'test_q_table.pkl'

    def tearDown(self):
        if os.path.exists(self.q_table_path):
            os.remove(self.q_table_path)

    def test_pikl_q_table(self):
        # Test the function with valid inputs
        pikl_q_table(self.bubs, 'W', self.q_table_path)
        self.assertTrue(os.path.exists(self.q_table_path))

        # Test the function with invalid rl_agent_color
        with self.assertRaises(ValueError):
            pikl_q_table(self.bubs, 'invalid_color', self.q_table_path)

if __name__ == '__main__':
    unittest.main()