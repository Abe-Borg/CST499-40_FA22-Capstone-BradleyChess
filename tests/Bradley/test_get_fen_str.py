import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_fen_str_initial_position(self):
        expected_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        actual_fen = self.bradley.get_fen_str()
        self.assertEqual(actual_fen, expected_fen)

    def test_get_fen_str_after_e4(self):
        self.bradley.make_move('e2', 'e4')
        expected_fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
        actual_fen = self.bradley.get_fen_str()
        self.assertEqual(actual_fen, expected_fen)

    def test_get_fen_str_invalid_board_state(self):
        self.bradley.make_move('e2', 'e4')
        self.bradley.make_move('e7', 'e5')
        self.bradley.make_move('f1', 'c4')
        self.bradley.make_move('d8', 'h4')
        expected_fen = 'invalid board state, no fen str'
        actual_fen = self.bradley.get_fen_str()
        self.assertEqual(actual_fen, expected_fen)

if __name__ == '__main__':
    unittest.main()

# test_get_fen_str_initial_position: Tests that the FEN string for the initial chess position is correct.
# test_get_fen_str_after_e4: Tests that the FEN string after the move e2-e4 is correct.
# test_get_fen_str_invalid_board_state: Tests that the method returns an error message when the board state is invalid.