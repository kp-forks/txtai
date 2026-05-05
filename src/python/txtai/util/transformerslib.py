"""
TransformersLib module
"""


# pylint: disable=C0415
class TransformersLib:
    """
    Imports transformers and dependencies with fallbacks when the library is not installed.
    """

    def arguments(self):
        """
        Imports transformers.TrainingArguments
        """

        try:
            from transformers import TrainingArguments

        except ImportError:

            class TrainingArguments:
                """
                Stub for TrainingArguments
                """

        return TrainingArguments

    def config(self):
        """
        Imports transformers.configuration_utils.PretrainedConfig.

        Returns:
            PreTrainedConfig
        """

        try:
            from transformers.configuration_utils import PretrainedConfig

        except ImportError:

            class PretrainedConfig:
                """
                Stub for PretrainedConfig
                """

        return PretrainedConfig

    def dataset(self):
        """
        Import torch.utils.data.Dataset.

        Returns:
            Dataset
        """

        try:
            from torch.utils.data import Dataset

        except ImportError:

            class Dataset:
                """
                Stub for Dataset
                """

        return Dataset

    def model(self):
        """
        Imports transformers.modeling_utils.PreTrainedModel.

        Returns:
            PreTrainedModel
        """

        try:
            from transformers.modeling_utils import PreTrainedModel

        except ImportError:

            class PreTrainedModel:
                """
                Stub for PreTrainedModel
                """

        return PreTrainedModel

    def module(self):
        """
        Imports torch.nn.Module

        Returns:
            Module
        """

        try:
            import torch.nn

            # pylint: disable=C0103
            Module = torch.nn.Module

        except ImportError:

            class Module:
                """
                Stub for Module
                """

        return Module

    def torch(self):
        """
        Imports torch.

        Returns:
            torch
        """

        try:
            import torch

        except ImportError:

            class Torch:
                """
                Stub for torch
                """

                def __getattr__(self, name):
                    raise ImportError("Torch is not installed, install torch to use this module")

            torch = Torch()

        return torch

    def transformers(self):
        """
        Imports transformers.

        Returns:
            transformers
        """

        try:
            import transformers

        except ImportError:

            class Transformers:
                """
                Stub for transformers
                """

                def __getattr__(self, name):
                    raise ImportError("Transformers is not installed, install transformers to use this module")

            transformers = Transformers()

        return transformers
