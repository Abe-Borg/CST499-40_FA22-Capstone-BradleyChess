import unittest
from unittest.mock import patch, mock_open
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    
    def setUp(self):
        self.my_class = MyClass()
        
    @patch('builtins.open', new_callable=mock_open)
    def test_continue_training_rl_agents(self, mock_file):
        # Set up test data
        training_results_filepath = 'test_results.txt'
        num_games_to_play = 2
        self.my_class.W_rl_agent.is_trained = True
        self.my_class.B_rl_agent.is_trained = True
        
        # Call the method
        self.my_class.continue_training_rl_agents(training_results_filepath, num_games_to_play)
        
        # Assert that the method writes to the training results file
        mock_file.assert_called_with(training_results_filepath, 'a')
        handle = mock_file()
        handle.write.assert_called()
        
        # Assert that the method resets the environment after each game
        self.assertEqual(self.my_class.environ.board, self.my_class.settings.starting_board)
        
if __name__ == '__main__':
    unittest.main()