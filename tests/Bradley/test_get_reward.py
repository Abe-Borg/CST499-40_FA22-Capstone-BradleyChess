import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_reward(self):
        # Test that the method returns an integer
        result = self.bradley.get_reward('e4')
        self.assertIsInstance(result, int)

        # Test that the method returns the expected reward for each type of move
        expected_rewards = {
            'Nf3': self.bradley.settings.piece_dev_pts,
            'Rb1': self.bradley.settings.piece_dev_pts,
            'Bc4': self.bradley.settings.piece_dev_pts,
            'Qd2': self.bradley.settings.piece_dev_pts,
            'exd5': self.bradley.settings.capture_pts,
            'e8=Q': self.bradley.settings.promotion_Queen_pts,
            'Ke2#': self.bradley.settings.checkmate_pts
        }
        for move, expected_reward in expected_rewards.items():
            result = self.bradley.get_reward(move)
            self.assertEqual(result, expected_reward)

if __name__ == '__main__':
    unittest.main()