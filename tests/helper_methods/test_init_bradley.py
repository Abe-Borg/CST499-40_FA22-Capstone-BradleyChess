import unittest
import pandas as pd
import imman
from my_module import init_bradley

class TestInitBradley(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({
            'white': ['Magnus Carlsen', 'Fabiano Caruana', 'Wesley So'],
            'black': ['Hikaru Nakamura', 'Levon Aronian', 'Maxime Vachier-Lagrave'],
            'result': ['1-0', '0-1', '1/2-1/2']
        })

    def test_init_bradley(self):
        # Test the function with a valid chess data
        bubs = init_bradley(self.chess_data)
        self.assertIsInstance(bubs, imman.Bradley)

        # Test the function with an invalid chess data
        with self.assertRaises(TypeError):
            init_bradley('invalid_chess_data')

if __name__ == '__main__':
    unittest.main()