"""Utils for mlflow
"""
import os
import mlflow


def absolutify(config: dict):
    """Transform all relative paths to absolute paths
    Args:
        config (dict): config
    """
    for key, value in config.items():
        if isinstance(value, dict):
            absolutify(value)
        elif key.endswith('_path'):
            config[key] = os.path.abspath(value)


def log_config(config: dict, context: str = "config"):
    """Log config
    Args:
        config (dict): config
        context: context of current config
    """
    for key, value in config.items():
        key_context = f'{context}.{key}'

        if isinstance(value, dict):
            log_config(value, key_context)
        else:
            mlflow.log_param(key_context, value)
