import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_recv_opp_move_success(self):
        # Test that a valid move is successfully loaded onto the chessboard
        result = self.bradley.recv_opp_move('Nf3')
        self.assertTrue(result)

    def test_recv_opp_move_failure(self):
        # Test that an invalid move fails to load onto the chessboard
        result = self.bradley.recv_opp_move('invalid_move')
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()

# We then define two test methods: test_recv_opp_move_success and test_recv_opp_move_failure. 
# In the first test method, we call the recv_opp_move method with a valid chess move (Nf3) 
# and assert that the method returns True, indicating that the move was successfully loaded onto the chessboard.

# In the second test method, we call the recv_opp_move method with an invalid chess move (invalid_move) 
# and assert that the method returns False, indicating that the move failed to load onto the chessboard.

# Finally, we use the unittest.main() method to run the tests. When you run this test file, 
# you should see output indicating whether the tests passed or failed.