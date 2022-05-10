
import src.utilities as utils
from src.feature.process_thai_text import process_text
from pythainlp.tag import pos_tag_sents
import pandas as pd
from ast import literal_eval

## test str to list from csv file.
config = utils.read_config()
print("hellow word")
df_ws = pd.read_csv(config['data']['processed_ws'], converters={'processed': pd.eval})

print(type(df_ws['processed'].iloc[555]))
pos_ws = pos_tag_sents(df_ws['processed'].tolist(), corpus='orchid_ud')



print("program terminated!")