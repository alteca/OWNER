"""Base data preprocessing class
"""
import abc


class BasePreprocessor(abc.ABC):
    """Base class for data preprocessing
    """

    def __init__(self, config: dict):
        """Constructor
        Args:
            config (dict): data config
        """
        self.config = config

    @abc.abstractmethod
    def preprocess_and_save(self, output_folder: str):
        """Preprocess and save data
        Args:
            output_folder (str): output folder to store data
        """
