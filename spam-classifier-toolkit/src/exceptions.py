"""
exceptions.py
-------------
Custom exception hierarchy so calling code can catch specific,
meaningful errors instead of bare Exceptions. This implements the
project's error-handling strategy (a required non-functional aspect).
"""


class SpamToolkitError(Exception):
    """Base class for all application-specific errors."""


class DatasetError(SpamToolkitError):
    """Raised when the dataset is missing, empty, or malformed."""


class ModelNotFoundError(SpamToolkitError):
    """Raised when a requested model name is not registered."""


class ModelNotTrainedError(SpamToolkitError):
    """Raised when a prediction is requested before training."""
