"""Scorer for baseline
"""
from argparse import ArgumentParser
import tomllib
import tempfile
import os
import mlflow
from .utils.mlflow import log_config, absolutify
from .data.serialization import parse_owner_dataset, parse_mini_owner_dataset
from .evaluation.mention_detection import evaluate_mention_detection
from .evaluation.entity_typing import evaluate_entity_typing


def main(config: dict):
    """Main entry point
    Args:
        config (dict): config
    """
    dataset = parse_owner_dataset(config['test_dataset_path'])
    predictions = parse_mini_owner_dataset(config['test_prediction_path'])

    evaluate_mention_detection(dataset.documents, predictions.documents, list(
        dataset.metadata.entity_types), 'test')
    evaluate_entity_typing(dataset.documents, predictions.documents, list(
        dataset.metadata.entity_types), list(predictions.metadata.entity_types),
        'test', prefix='ner')


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
