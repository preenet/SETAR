
"""
Loading and making raw data 
@Authors: pree.t@cmu.ac.th
Todo:   
* add google drive or one drive 
"""
import os
import pandas as pd
import src.utilities as utils

# get config file
config = utils.read_config()
data_path_ws_ori = os.path.join(config['data']['raw_ws_ori'])
data_path_ws = os.path.join(config['data']['raw_ws'])

print('making khon-thai corpus...')

print('making wisesight corpus...')
# we use the original wisesight corpus and reconstruct a new dataframe
texts = []
targets = []

with open(str(data_path_ws_ori) + '/' + 'neg.txt', encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())
        targets.append('neg')

with open(str(data_path_ws_ori) + '/' + 'neu.txt', encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())
        targets.append('neu')

with open(str(data_path_ws_ori) + '/' + 'pos.txt', encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())
        targets.append('pos')

with open(str(data_path_ws_ori) + '/' + 'q.txt', encoding='utf-8') as f:
    for line in f:
        texts.append(line.strip())
        targets.append('q')
        
df_ws = pd.DataFrame({'texts': texts, 'targets': targets})
print(df_ws.shape)
print("writing to disk..")
df_ws.to_csv(data_path_ws)
print('make dataset finished!')