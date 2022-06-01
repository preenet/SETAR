"""
This script provides useful funcs to all other scripts
"""
from distutils.command.config import config
from regex import D
import yaml
import pandas as pd


def read_config():
    # Read in config file
    config = {k: v for d in yaml.load(
        open('config.yaml'),
             Loader=yaml.SafeLoader) for k, v in d.items()}
    return config


def generate_report_10repeated(data_name, file_name):
    # loop for the same file name with different seed number
    dfs = []
    for i in range(0, 10):
        df = pd.read_csv(config['output_scratch'] + data_name + '/' + str(i) + '_' + file_name, header=None)
        dfs.append(df)
    
    # calculate mean and std of each row among files
    df_res = pd.DataFrame()
    for i in range(len(dfs)):
        tmp = [df.iloc[i, :] for df in dfs]
        df_all = pd.DataFrame(tmp)
        df_train = df_all.iloc[:, 1:7].mean().round(decimals=4).astype(str).add(u"\u00B1" + \
             df_all.iloc[:, 1:7].std().round(decimals=4).astype(str)).to_frame(0).T
        df_test = df_all.iloc[:, 8:14].mean().round(decimals=4).astype(str).add(u"\u00B1" + \
            df_all.iloc[:, 8:14].std().round(decimals=4).astype(str)).to_frame(0).T
 
        algo_name = df_all.iloc[0, 0]
        df_tmp = df_train.join(df_test)
        df_tmp.insert(0, 'Algo', algo_name[1:])
        df_res = pd.concat([df_res, df_tmp])
        print(df_res)
    df_res.to_csv(config['output_scratch'] + data_name + '/' + 'report_' + file_name , header=False)
    return

if __name__ == "__main__":
    config = read_config()
    generate_report_10repeated('pos_tag_ws', '12classifier_POSBOW_res.csv')