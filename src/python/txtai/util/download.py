"""
Download module
"""

import os

from huggingface_hub import hf_hub_download


class Download:
    """
    Downloads files from the Hugging Face Hub. This method also supports local file paths that exist.
    """

    def __call__(self, path, name=None):
        """
        Downloads path from the Hugging Face Hub. Supports local file paths.

        Args:
            path: model path or repo
            name: file name to load

        Returns:
            local cached model path, if available
        """

        # Reject invalid input
        if not isinstance(path, str) or (name and not isinstance(name, str)):
            return None

        # Split into repo and name components
        if not name:
            # Split into parts
            parts = path.split("/")

            # Calculate repo id split
            repo = 2 if len(parts) > 2 else 1

            # Set path and name components
            path, name = "/".join(parts[:repo]), "/".join(parts[repo:])

        # Download (if necessary) and return local file path
        local = os.path.join(path, name)
        return local if os.path.exists(local) else hf_hub_download(repo_id=path, filename=name)
