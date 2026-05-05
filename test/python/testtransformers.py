"""
Transformers module tests
"""

import sys
import unittest

# pylint: disable=C0415,W0611,W0621
import txtai


class TestTransformers(unittest.TestCase):
    """
    Simulates transformers not being installed.
    """

    @classmethod
    def setUpClass(cls):
        """
        Simulate transformers not being installed
        """

        modules = [
            "transformers",
            "transformers.configuration_utils",
            "transformers.modeling_utils",
            "transformers.modeling_outputs",
            "torch",
            "torch.nn",
            "torch.onnx",
            "torch.utils.data",
        ]

        # Get handle to all currently loaded txtai modules
        modules = modules + [key for key in sys.modules if key.startswith("txtai")]
        cls.modules = {module: None for module in modules}

        # Replace loaded modules with stubs. Save modules for later reloading
        for module in cls.modules:
            if module in sys.modules:
                cls.modules[module] = sys.modules[module]

            # Remove txtai modules. Set optional dependencies to None to prevent reloading.
            if "txtai" in module:
                if module in sys.modules:
                    del sys.modules[module]
            else:
                sys.modules[module] = None

    @classmethod
    def tearDownClass(cls):
        """
        Resets modules environment back to initial state.
        """

        # Reset replaced modules in setup
        for key, value in cls.modules.items():
            if value:
                sys.modules[key] = value
            else:
                del sys.modules[key]

    def testTransformers(self):
        """
        Test transformers not installed
        """

        from txtai.util import TransformersLib

        lib = TransformersLib()

        # Test transformers stubs
        for x in [lib.arguments(), lib.config(), lib.dataset(), lib.module(), lib.model()]:
            self.assertTrue(x.__module__.endswith("transformerslib"))

        # pylint: disable=W0106
        with self.assertRaises(ImportError):
            lib.transformers().AutoModel

        # pylint: disable=W0106
        with self.assertRaises(ImportError):
            lib.torch().device
