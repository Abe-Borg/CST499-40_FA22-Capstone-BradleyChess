import chess
import random
import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Agent

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent('W')

    def test_choose_high_val_move(self):
        self.agent.board.set_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.agent.legal_moves = [str(move) for move in self.agent.board.legal_moves]
        action = self.agent.choose_high_val_move()
        self.assertEqual(isinstance(action, dict), True)
        self.assertEqual(len(action), 1)
        self.assertEqual(isinstance(list(action.values())[0], str), True)

if __name__ == '__main__':
    unittest.main()