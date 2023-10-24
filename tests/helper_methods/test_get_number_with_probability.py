import unittest
import random
from my_module import get_number_with_probability

class TestGetNumberWithProbability(unittest.TestCase):
    def test_get_number_with_probability(self):
        # Test the function with probability = 0
        self.assertEqual(get_number_with_probability(0), 0)

        # Test the function with probability = 1
        self.assertEqual(get_number_with_probability(1), 1)

        # Test the function with probability = 0.5
        for i in range(100):
            result = get_number_with_probability(0.5)
            self.assertIn(result, [0, 1])

        # Test the function with invalid probability values
        with self.assertRaises(ValueError):
            get_number_with_probability(-0.1)

        with self.assertRaises(ValueError):
            get_number_with_probability(1.1)

if __name__ == '__main__':
    unittest.main()