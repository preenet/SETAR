import pandas as pd
import numpy as np
import src.utilities as utils
from matplotlib import pyplot as plt
plt.rcParams['font.family'] = 'tahoma'

# path config
import os

# get config file
config = utils.read_config()


# path for raw data
data_path_kt = os.path.join(config['data']['raw_kt'])
data_path_ws = os.path.join(config['data']['raw_ws'])
data_path_dict = os.path.join(config['data']['raw_dict'])

# output folder to be put in
out_path = os.path.join(config['output'])


def get_data():
    # load dataset
    df_kt = pd.read_csv(data_path_kt)
    df_ws = pd.read_csv(data_path_ws)

    y_kt = df_kt['vote'].astype('category').cat.codes
    y_ws = df_ws['targets'].astype('category').cat.codes
    print("unique label for kt:", y_kt.unique(), "unique label for ws:", y_ws.unique())

if __name__ == "__main__":
    print("Building features...")
    get_data()
    print("Finished building features!")