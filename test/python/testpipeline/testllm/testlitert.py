"""
LiteRT module tests
"""

import os
import unittest

from txtai.pipeline import LLM


class TestLiteRT(unittest.TestCase):
    """
    LiteRT tests.
    """

    @unittest.skipIf(os.name == "nt", "testGeneration skipped on Windows")
    def testGeneration(self):
        """
        Test generation with LiteRT
        """

        # Test model generation with LiteRT
        model = LLM("litert-community/gemma-4-E2B-it-litert-lm/gemma-4-E2B-it.litertlm")

        # Test standard
        self.assertIsNotNone(model("Hello"))

        # Test streaming
        self.assertIsNotNone(list(model("Hello", stream=True)))
