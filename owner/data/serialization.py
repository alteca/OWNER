"""Utilities to load/save datasets
"""
from hashlib import md5
import json
from typing import Any, Dict, Optional
import mlflow
from .model import Dataset, MiniDataset


def serialize_mini_owner_dataset(dataset: MiniDataset, output_file: str):
    """Serialize mini dataset
    Args:
        dataset (MiniDataset): dataset to save (for evaluation purpose)
        output_file (str): file to save dataset
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(dataset.model_dump_json())


def parse_mini_owner_dataset(input_file: str) -> MiniDataset:
    """Read dataset from file
    Args:
        input_file (str): file
    Returns:
        MiniDataset: dataset (for evaluation purpose)
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        return MiniDataset.model_validate_json(file.read())


def serialize_owner_dataset(dataset: Dataset, output_file: str):
    """Serialize dataset
    Args:
        dataset (Dataset): dataset to save
        output_file (str): file to save dataset
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(dataset.model_dump_json())


def parse_owner_dataset(input_file: str) -> Dataset:
    """Read dataset from file
    Args:
        input_file (str): file
    Returns:
        Dataset: dataset
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        return Dataset.model_validate_json(file.read())


##### MLFlow #####
class OwnerDatasetSource(mlflow.data.DatasetSource):
    """mlflow dataset source for LinkedDocRED dataset
    """

    def __init__(self, path: str) -> None:
        """Constructor
        Args:
            path (str): path to linked docred dataset
        """
        super().__init__()
        self.path = path

    def _get_source_type(self) -> str:
        return 'file'

    def load(self) -> Any:
        return self.path

    @staticmethod
    def _can_resolve(raw_source: Any) -> bool:
        return False

    @classmethod
    def _resolve(cls, raw_source: Any) -> "OwnerDatasetSource":
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': self.path,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, source_dict: Dict[Any, Any]) -> "OwnerDatasetSource":
        """
        Constructs an instance of the DatasetSource from a dictionary representation.

        :param source_dict: A dictionary representation of the DatasetSource.
        :return: A DatasetSource instance.
        """
        return cls(source_dict.get('path'))


class OwnerDataset(mlflow.data.Dataset):
    """LinkedDocRED dataset
    """

    def __init__(self, source: OwnerDatasetSource, dataset: Dataset, name: str):
        super().__init__(source, name, "empty")
        self._dataset = dataset
        self._digest = self._compute_digest()

    def _compute_digest(self) -> str:
        return md5(self.dataset.model_dump_json().encode('utf-8')).hexdigest()

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        config = base_dict
        return config

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def profile(self) -> Optional[Any]:
        """
        Summary statistics for the dataset
        """
        return {
            "num_rows": len(self.dataset),
        }


def from_owner(path: str, name: str) -> OwnerDataset:
    """Create mlflow dataset from OWNER
    Args:
        path (str): path to OWNER file
        name (str): name of dataset
    Returns:
        OwnerDataset: dataset
    """
    dataset = parse_owner_dataset(path)
    source = OwnerDatasetSource(path)
    return OwnerDataset(source, dataset, name)
