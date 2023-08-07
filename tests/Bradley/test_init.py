import unittest
import pandas as pd
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.chess_data = pd.DataFrame()
        self.bradley = Bradley(self.chess_data)

    def test_init_chess_data(self):
        self.assertEqual(self.bradley.chess_data, self.chess_data)

    def test_init_settings(self):
        self.assertIsNotNone(self.bradley.settings)

    def test_init_environ(self):
        self.assertIsNotNone(self.bradley.environ)

    def test_init_W_rl_agent(self):
        self.assertIsNotNone(self.bradley.W_rl_agent)
        self.assertEqual(self.bradley.W_rl_agent.color, 'W')

    def test_init_B_rl_agent(self):
        self.assertIsNotNone(self.bradley.B_rl_agent)
        self.assertEqual(self.bradley.B_rl_agent.color, 'B')

    def test_init_W_rl_agent_learn_rate(self):
        self.assertEqual(self.bradley.W_rl_agent.settings.learn_rate, 0.6)

    def test_init_W_rl_agent_discount_factor(self):
        self.assertEqual(self.bradley.W_rl_agent.settings.discount_factor, 0.3)

    def test_init_B_rl_agent_learn_rate(self):
        self.assertEqual(self.bradley.B_rl_agent.settings.learn_rate, 0.2)

    def test_init_B_rl_agent_discount_factor(self):
        self.assertEqual(self.bradley.B_rl_agent.settings.discount_factor, 0.8)

    def test_init_engine(self):
        self.assertIsNotNone(self.bradley.engine)

if __name__ == '__main__':
    unittest.main()


# test_init_chess_data: Tests that the chess_data attribute is initialized correctly.
# test_init_settings: Tests that the settings attribute is not None.
# test_init_environ: Tests that the environ attribute is not None.
# test_init_W_rl_agent: Tests that the W_rl_agent attribute is not None and has the correct color.
# test_init_B_rl_agent: Tests that the B_rl_agent attribute is not None and has the correct color.
# test_init_W_rl_agent_learn_rate: Tests that the learn_rate attribute of the W_rl_agent is initialized correctly.
# test_init_W_rl_agent_discount_factor: Tests that the discount_factor attribute of the W_rl_agent is initialized correctly.
# test_init_B_rl_agent_learn_rate: Tests that the learn_rate attribute of the B_rl_agent is initialized correctly.
# test_init_B_rl_agent_discount_factor: Tests that the discount_factor attribute of the B_rl_agent is initialized correctly.
# test_init_engine: Tests that the engine attribute is not None.