"""
This script provides useful funcs to all other scripts
"""
import yaml
import pandas as pd


def read_config():
    # Read in config file
    config = {k: v for d in yaml.load(
        open('config.yaml'),
             Loader=yaml.SafeLoader) for k, v in d.items()}
    return config


def generate_report_10repeated_baseline(data_name, file_name):
    """_create mean and std from 10 separated csv files as a single file_

    Args:
        data_name (_str_): _name of the folder_
        file_name (_str_): _common name of the file_
    """
    # loop for the same file name with different seed number
    dfs = []
    for i in range(0, 10):
        df = pd.read_csv(config['output_scratch'] + data_name + '/' + str(i) + '_' + file_name, header=None)
        dfs.append(df)
    
    # calculate mean and std of each row among files
    df_res = pd.DataFrame()
    for i in range(0, 12):
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
    headers =  [" ", "ACC", "PRE", "REC", "MCC", "AUC", "F1", "ACC", "PRE", "REC", "MCC", "ACC", "F1"]
    df_res.to_csv(config['output_scratch'] + data_name + '/' + 'report_' + file_name , header=headers, index=False, encoding="utf-8-sig")
    
    return

def generate_report_10repeated_deepbaseline(data_name, file_name):
    """_create row of mean and std from a csv file as a row_

    Args:
        data_name (_str_): _name of the folder_
        file_name (_str_): _common name of the file_
    """
    df = pd.read_csv(config['output_scratch'] + data_name + '/' + file_name, header=None)
    
    df_train = df.iloc[:, 1:7].mean().round(decimals=4).astype(str).add(u"\u00B1" + \
        df.iloc[:, 1:7].std().round(decimals=4).astype(str)).to_frame(0).T
    df_test = df.iloc[:, 8:14].mean().round(decimals=4).astype(str).add(u"\u00B1" + \
        df.iloc[:, 8:14].std().round(decimals=4).astype(str)).to_frame(0).T
    df_tmp = df_train.join(df_test)
    #df_res = pd.concat([df_res, df_tmp])
    
    headers =  ["ACC", "PRE", "REC", "MCC", "AUC", "F1", "ACC", "PRE", "REC", "MCC", "ACC", "F1"]
    df_tmp.to_csv(config['output_scratch'] + data_name + '/' + 'report_' + file_name , index=False, header= headers, encoding="utf-8-sig")
    return

if __name__ == "__main__":
    config = read_config()
    #generate_report_10repeated_baseline('pos_tag_ws', '12classifier_POSBOWFLAT_res.csv')
    generate_report_10repeated_deepbaseline('CNN_WS', 'CNN_w2v_ws.csv')
    