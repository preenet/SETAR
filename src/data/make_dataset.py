
"""
Loading original and making raw data 
@Authors: pree.t@cmu.ac.th
Todo:
* add google drive or one drive 
"""
import sys
import pandas as pd
import src.utilities as utils

def make_kt():
    print('Making khon-thai corpus...')
    return

def make_ws():
    print('Making wisesight corpus...')
    texts = []
    targets = []
    with open(config['data']['raw_ws_ori'] + '/' + 'neg.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('neg')

    with open(config['data']['raw_ws_ori'] + '/' + 'neu.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('neu')

    with open(config['data']['raw_ws_ori'] + '/' + 'pos.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('pos')

    with open(config['data']['raw_ws_ori'] + '/' + 'q.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('q')
            
    df_ws = pd.DataFrame({'texts': texts, 'targets': targets})
    print(df_ws.shape)
    df_ws.to_csv(config['data']['raw_ws'])
   
    return

def make_tt():
    print('Making thaitale corpus...')
    df_tt = pd.read_csv(config['data']['raw_tt_ori'] + 'tale_data.csv')
    print(df_tt.shape)
    df_tt.to_csv(config['data']['raw_tt'] + 'thaitale.csv')
    return

if __name__ == "__main__":
    # get config file
    config = utils.read_config()

    if (len(sys.argv) != 2):
        print("need argument")
        sys.exit(1)

    if sys.argv[1] in config['data']['name']: 
        data_name = sys.argv[1]

        if data_name == 'kt':
            make_kt()
        elif data_name == 'ws':
            make_ws()
        elif data_name == 'tt':
            make_tt()
    else:
        print("no such data name.")
        sys.exit(1)
print('Program terminate sucessfully!')