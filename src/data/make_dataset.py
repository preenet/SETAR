
"""
Loading original corpus and making as standard experiment dataset.
@Authors: pree.t@cmu.ac.th
Todo:
* add google drive or one drive 
"""
import sys

import datasets
import pandas as pd
import src.utilities as utils
from src.feature.process_thai_text import process_text


def make_kt():
    print('Making khon-thai corpus...')
    df_kt = pd.read_csv(config['data']['raw_kt'])
    df_kt.rename(columns={'vote': 'target'}, inplace=True)
    print(df_kt.info)

    print("Pre-processing stage 2 with word tokenizing...")
    df_kt['processed'] = df_kt['text'].apply(str).apply(process_text)

    df_kt.to_csv(config['data']['processed_kt'])
    return

def make_ws():
    print('Making wisesight corpus...')
    texts = []
    targets = []
    with open(config['data']['raw_ws'] + '/' + 'neg.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('neg')
    f.close()

    with open(config['data']['raw_ws'] + '/' + 'neu.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('neu')
    f.close()

    with open(config['data']['raw_ws'] + '/' + 'pos.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('pos')
    f.close()

    with open(config['data']['raw_ws'] + '/' + 'q.txt', encoding='utf-8') as f:
        for line in f:
            texts.append(line.strip())
            targets.append('q')
    f.close()
            
    df_ws = pd.DataFrame({'texts': texts, 'targets': targets})
    df_ws.rename(columns={'texts': 'text', 'targets': 'target'}, inplace=True)
    print(df_ws.info)

    print("Pre-processing stage 2 with word tokenizing...")
    df_ws['processed'] = df_ws['text'].apply(process_text)

    df_ws.to_csv(config['data']['processed_ws'])
    return

def make_tt():
    print('Making thaitale corpus...')
    df_tt = pd.read_csv(config['data']['raw_tt'])
    df_tt.rename(columns={'Y_vote': 'target'}, inplace=True)
    print(df_tt.info)

    print("Pre-processing stage 2 with word tokenizing...")
    df_tt['processed'] = df_tt['text'].apply(str).apply(process_text)

    df_tt.to_csv(config['data']['processed_tt'])
    return

def make_wn():
    print('making wongnai corpus...')
    from datasets import list_datasets, load_dataset
    [item for item in list_datasets() if 'wongnai' in item]
    wongnai = load_dataset('wongnai_reviews')
    
    file = open(config['data']['processed_wn'], "w",  encoding="utf-8")
    file.write("text, target\n")
    for i, item in enumerate(wongnai['train'][:]['review_body']):
        text = item.replace(',', '')
        text = text.replace('\n', '')
        label = wongnai['train'][i]['star_rating']
        file.write(text + ',' + str(label)+ '\n')
        
    for i, item in enumerate(wongnai['test'][:]['review_body']):
        text = item.replace(',', '')
        text = text.replace('\n', '')
        label = wongnai['test'][i]['star_rating']
        file.write(text + ',' + str(label)+ '\n')

    file.close()
    
    print("Pre-processing stage 2 with word tokenizing...")
    df_wn = pd.read_csv(config['data']['processed_wn'])
    df_wn['processed'] = df_wn['text'].apply(str).apply(process_text)
    df_wn.to_csv(config['data']['processed_wn'])
    return

def make_to():
    print('making toxic tweet corpus...')
    from datasets import list_datasets, load_dataset
    [item for item in list_datasets() if 'thai_toxicity_tweet' in item]
    toxic_tweet = load_dataset('thai_toxicity_tweet')
    
    file = open(config['data']['processed_to'], "w",  encoding="utf-8")
    file.write("text, target\n")  
    
    for i, item in enumerate(toxic_tweet['train'][:]['tweet_text']):
        text = item.replace(',', '')
        text = text.replace('\n', '')
        label = toxic_tweet['train'][i]['is_toxic']
    
        # remove empty text, tweet not found, and non-label
        if not (text == "TWEET_NOT_FOUND" or text == ""):
            file.write(text + ',' + str(label)+ '\n')
    file.close()
    return
    
if __name__ == "__main__":
    # get config file
    config = utils.read_config()

    if (len(sys.argv) != 2):
        print("*Error: incorrect number of argument")
        sys.exit(1)

    if sys.argv[1] in config['data']['name']: 
        data_name = sys.argv[1]

        if data_name == 'kt':
            make_kt()
        elif data_name == 'ws':
            make_ws()
        elif data_name == 'tt':
            make_tt()
        elif data_name == 'wn':
            make_wn()
        elif data_name == 'to':
            make_to()
    else:
        print("*Error: no such data name.")
        sys.exit(1)
print('*Program terminate successfully!')
