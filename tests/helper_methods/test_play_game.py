import unittest
import pandas as pd
import imman
from my_module import play_game

class TestPlayGame(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame({
            'white': ['Magnus Carlsen', 'Fabiano Caruana', 'Wesley So'],
            'black': ['Hikaru Nakamura', 'Levon Aronian', 'Maxime Vachier-Lagrave'],
            'result': ['1-0', '0-1', '1/2-1/2']
        })
        self.bubs = imman.Bradley(self.chess_data)

    def test_play_game(self):
        # Test the function with valid inputs
        with self.assertRaises(ValueError):
            play_game(self.bubs, 'invalid_color')
        # TODO: Add more test cases

if __name__ == '__main__':
    unittest.main()