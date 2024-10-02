"""Runner to run experiments listed in file runs.csv
"""
import os
import argparse
from string import Template
import pandas as pd
from tqdm.auto import tqdm


def read_runs(runs_file: str) -> pd.DataFrame:
    """Parse runs file
    Returns:
        pd.DataFrame: runs
    """
    runs = pd.read_csv(runs_file, header=0, sep=";", dtype=object)
    return runs


def execute_run(run: dict, config_template: Template):
    """Execute run
    Args:
        run (dict): run arguments
                    Format:
                        - run_name = name of run
                        - entrypoint = name of entrypoint
                        - experiment_name = name of experiment
                        - others columns are used to fill config template
        config_template (Template): template file to fill
    """
    run_name = run['run_name']
    experiment_name = run['experiment_name']
    entrypoint = run['entrypoint']

    with open('config.toml', 'w', encoding='utf-8') as file:
        file.write(config_template.safe_substitute(run))

    os.system(('export MLFLOW_TRACKING_URI=http://localhost:5000 && '
               'export TOKENIZERS_PARALLELISM=false &&'
               'mlflow run '
               f'--experiment-name="{experiment_name}" '
               f'--run-name="{run_name}" '
               f'-e {entrypoint} '
               '-P config_file=config.toml .'))


def main(template_file: str, runs_file: str):
    """Main entrypoint
    """
    with open(template_file,
              'r', encoding='utf-8') as file:
        config_template = Template(file.read())

    runs = read_runs(runs_file)
    for _, run in tqdm(runs.iterrows(), total=len(runs)):
        execute_run(run.to_dict(), config_template)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run experiments')
    parser.add_argument('--template', type=str,
                        default='runner/config_template.toml.txt')
    parser.add_argument('--runs', type=str, default='runner/runs.csv')
    args = parser.parse_args()
    main(args.template, args.runs)
