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
    df = df.mean().round(decimals=4).astype(str).add(u"\u00B1" + \
        df.std().round(decimals=4).astype(str)).to_frame(0).T
   
    df.to_csv(config['output'] + file_name + '_report.csv', header = False)
    return


if __name__ == "__main__":
    config = read_config()
    generate_report_10repeated('CNN_pad_ws.csv')