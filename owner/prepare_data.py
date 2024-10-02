"""Main entrypoint for data preprocessing
"""
from argparse import ArgumentParser
import logging
import tomllib
import tempfile
import os
import mlflow
from .utils.mlflow import log_config, absolutify
from .data.preprocessing.base import BasePreprocessor
from .data.preprocessing.crossner import CrossNerPreprocessor
from .data.preprocessing.conll2003 import Conll2003Preprocessor
from .data.preprocessing.i2b2 import I2b2Preprocessor
from .data.preprocessing.wnut17 import Wnut17Preprocessor
from .data.preprocessing.genia import GeniaPreprocessor
from .data.preprocessing.gum import GumPreprocessor
from .data.preprocessing.mit import MitPreprocessor
from .data.preprocessing.pilener import PileNerPreprocessor
from .data.preprocessing.fabner import FabNerPreprocessor

logger = logging.getLogger("mlflow")


def main(config: dict):
    """Main entry point
    Args:
        config (dict): config
    """
    preprocessor: BasePreprocessor = None

    data_type = config['data_type']
    logger.info("Preprocessing using: %s", data_type)
    match data_type:
        case 'crossner':
            preprocessor = CrossNerPreprocessor(config)
        case 'conll2003':
            preprocessor = Conll2003Preprocessor(config)
        case 'i2b2':
            preprocessor = I2b2Preprocessor(config)
        case 'wnut17':
            preprocessor = Wnut17Preprocessor(config)
        case 'genia':
            preprocessor = GeniaPreprocessor(config)
        case 'gum':
            preprocessor = GumPreprocessor(config)
        case 'mit':
            preprocessor = MitPreprocessor(config)
        case 'pilener':
            preprocessor = PileNerPreprocessor(config)
        case 'fabner':
            preprocessor = FabNerPreprocessor(config)
        case _:
            raise ValueError(f'Unknown data type: "{data_type}"')

    preprocessor.preprocess_and_save(config['output_folder'])


if __name__ == '__main__':
    parser = ArgumentParser('Preprocessor')
    parser.add_argument('--config-file', type=str, required=True)
    args = parser.parse_args()

    with open(args.config_file, 'rb') as file:
        toml_config = tomllib.load(file)
        absolutify(toml_config)
        log_config(toml_config)
    mlflow.log_artifact(args.config_file, '')

    with tempfile.TemporaryDirectory() as tmpdir_name:
        os.chdir(tmpdir_name)
        main(toml_config)
