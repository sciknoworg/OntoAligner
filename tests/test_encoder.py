import unittest
from onmatcher.encoder import ConceptNaiveEncoder


class TestScript1(unittest.TestCase):

    def test_add(self):
        # Test cases for the add function
        self.assertIn("<Problem Definition>", ConceptNaiveEncoder().prompt_template)


if __name__ == '__main__':
    unittest.main()
