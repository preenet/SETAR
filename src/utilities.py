"""
This script provides useful funcs to all other scripts
"""
from distutils.command.config import config
import yaml
import pandas as pd


def read_config():
    # Read in config file
    config = {k: v for d in yaml.load(
        open('config.yaml'),
             Loader=yaml.SafeLoader) for k, v in d.items()}
    return config


def generate_report_10repeated(file_name):
    df = pd.read_csv(config['output'] + file_name, header=None)
    df = df.sum().to_frame().T.rename_axis(None, axis=1)
    df.to_csv(config['output'] + file_name + 'report.csv', header = False)
    return


if __name__ == "__main__":
    config = read_config()
    generate_report_10repeated('CNN_pad_ws.csv')