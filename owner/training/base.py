"""Base class that is derived by all methods
"""
import os
import abc
from accelerate import Accelerator


class BaseTrainer(metaclass=abc.ABCMeta):
    """Base class to define a trainer
    """

    def __init__(self,
                 config: dict  # pylint: disable=unused-argument
                 ):
        """Constructor
        Args:
            config (dict): model configuration
        """
        self.config = config
        self.accelerator = Accelerator()

    @abc.abstractmethod
    def load_data(self, training: bool = True):
        """Load and preprocess dataset
        """

    @abc.abstractmethod
    def load_model(self, folder: str):
        """Load model from folder
        """

    def save_model(self, folder: str):
        """Load model to folder
        """
        os.makedirs(folder, exist_ok=True)

    @abc.abstractmethod
    def train(self):
        """Train method (if needed)
        """

    @abc.abstractmethod
    def evaluate(self):
        """Evaluate method (always)
        """
