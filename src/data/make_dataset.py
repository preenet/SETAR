
"""
Loading original corupus and making a standard experiment raw dataset.
@Authors: pree.t@cmu.ac.th
Todo:
* add google drive or one drive 
"""
import sys
import pandas as pd
import src.utilities as utils

def make_kt():
    print('Making khon-thai corpus...')
    df_kt = pd.read_csv(config['data']['raw_kt_ori'])
    df_kt.rename(columns={'vote': 'target'}, inplace=True)
    print(df_kt.info)
    df_kt.to_csv(config['data']['raw_kt'])
    return

def make_ws():
    print('Making wisesight corpus...')
    texts = []
    targets = []
    with open(config['data']['raw_ws_ori'] + '/' + 'neg.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('neg')
    f.close()

    with open(config['data']['raw_ws_ori'] + '/' + 'neu.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('neu')
    f.close()

    with open(config['data']['raw_ws_ori'] + '/' + 'pos.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('pos')
    f.close()

    with open(config['data']['raw_ws_ori'] + '/' + 'q.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('q')
    f.close()
            
    df_ws = pd.DataFrame({'texts': texts, 'targets': targets})
    df_ws.rename(columns={'texts': 'text', 'targets': 'target'}, inplace=True)
    print(df_ws.info)
    df_ws.to_csv(config['data']['raw_ws'])
   
    return

def make_tt():
    print('Making thaitale corpus...')
    df_tt = pd.read_csv(config['data']['raw_tt_ori'])
    df_tt.rename(columns={'consensus': 'target'}, inplace=True)
    print(df_tt.info)
    df_tt.to_csv(config['data']['raw_tt'])
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
        print("Error: no such data name.")
        sys.exit(1)
print('Program terminate sucessfully!')