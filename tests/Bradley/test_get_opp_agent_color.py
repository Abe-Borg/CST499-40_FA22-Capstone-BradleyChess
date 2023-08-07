import unittest
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_get_opp_agent_color_white(self):
        # Test that the method returns 'B' when given 'W'
        result = self.bradley.get_opp_agent_color('W')
        self.assertEqual(result, 'B')

    def test_get_opp_agent_color_black(self):
        # Test that the method returns 'W' when given 'B'
        result = self.bradley.get_opp_agent_color('B')
        self.assertEqual(result, 'W')

if __name__ == '__main__':
    unittest.main()

# In this example, we define a test class TestBradley that inherits from unittest.TestCase. 
# In the setUp method, we create an instance of the Bradley class to use in our tests.

# We then define two test methods: test_get_opp_agent_color_white and test_get_opp_agent_color_black. 
# In the first test method, we call the get_opp_agent_color method with the argument 'W' and assert that the method returns 'B', 
# indicating that the opposing agent is black. In the second test method, we call the get_opp_agent_color method with the argument 
# 'B' and assert that the method returns 'W', indicating that the opposing agent is white.

# Finally, we use the unittest.main() method to run the tests. When you run this test file, 
# you should see output indicating whether the tests passed or failed.