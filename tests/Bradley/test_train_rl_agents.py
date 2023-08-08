import unittest
from unittest.mock import patch, mock_open
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    
    def setUp(self):
        self.my_class = MyClass()
        
    @patch('builtins.open', new_callable=mock_open)
    def test_train_rl_agents(self, mock_file):
        # Set up test data
        training_results_filepath = 'test_results.txt'
        self.my_class.chess_data = pd.DataFrame({'Num Moves': [2, 3, 4]}, index=['W1', 'W2', 'W3'])
        
        # Call the method
        self.my_class.train_rl_agents(training_results_filepath)
        
        # Assert that the method sets the is_trained flag to True for both agents
        self.assertTrue(self.my_class.W_rl_agent.is_trained)
        self.assertTrue(self.my_class.B_rl_agent.is_trained)
        
        # Assert that the method writes to the training results file
        mock_file.assert_called_with(training_results_filepath, 'a')
        handle = mock_file()
        handle.write.assert_called()
        
        # Assert that the method resets the environment after each game
        self.assertEqual(self.my_class.environ.board, self.my_class.settings.starting_board)
        
if __name__ == '__main__':
    unittest.main()