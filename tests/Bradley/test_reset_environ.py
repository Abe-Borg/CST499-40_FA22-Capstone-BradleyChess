import unittest
from unittest.mock import MagicMock
from CST499_40_FA22_Capstone_BradleyChess.src import Bradley

class TestBradley(unittest.TestCase):
    def setUp(self):
        self.bradley = Bradley()

    def test_reset_environ(self):
        # Test that the method calls the reset_environ method of the Environ object
        self.bradley.environ.reset_environ = MagicMock()
        self.bradley.reset_environ()
        self.bradley.environ.reset_environ.assert_called_once()

if __name__ == '__main__':
    unittest.main()