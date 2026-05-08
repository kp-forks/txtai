"""
LiteRT module tests
"""

import unittest

from txtai.pipeline import LLM


class TestLiteRT(unittest.TestCase):
    """
    LiteRT tests.
    """

    def testGeneration(self):
        """
        Test generation with LiteRT
        """

        # Test model generation with LiteRT
        model = LLM("litert-community/SmolLM2-360M-Instruct/SmolLM2_360M_instruct.litertlm")

        # Test standard
        self.assertIsNotNone(model("Hello"))

        # Test streaming
        self.assertIsNotNone(list(model("Hello", stream=True)))
