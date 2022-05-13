import numpy as np
import joblib
import src.utilities as utils

from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import train_test_split

config = utils.read_config()


feat1, yall = joblib.load(config['data']['processed']+'text_bow1_kt.pkl')
feat2 = joblib.load(['data']['processed']+'text_bow2_kt.pkl')[0]
feat3 = joblib.load(['data']['processed']+'text_tfidf1_kt.pkl')[0]
feat4 = joblib.load(['data']['processed']+'text_tfidf2_kt.pkl')[0]
feat5 = joblib.load(['data']['processed']+'text_dict_bow1_kt.pkl')[0]
feat6 = joblib.load(['data']['processed']+'text_dict_bow2_kt.pkl')[0]
feat7 = joblib.load(['data']['processed']+'text_dict_tfidf1_kt.pkl')[0]
feat8 = joblib.load(['data']['processed']+'text_dict_tfidf2_kt.pkl')[0]
feat9 = joblib.load(['data']['processed']+'text_w2v_tfidf_kt.pkl')[0]
feat10 = joblib.load(['data']['processed']+'text_pos_bow1_kt.pkl')[0]

from sklearn.model_selection import train_test_split
[f1,ft1, f2, ft2, f3, ft3, f4, ft4, f5, ft5, f6, ft6, f7, ft7, f8, ft8, f9, ft9, f10, ft10, y, yt] = train_test_split(feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, yall, test_size=0.2, random_state=0)

all_feats = [('text_bow1_kt', f1, ft1),
               ('text_bow2_kt', f2, ft2),
               ('text_tfidf1_kt', f3, ft3),
               ('text_tfidf2_kt', f4, ft4),
               ('text_dict_bow1_kt', f5, ft5),
               ('text_dict_bow2_kt', f6, ft6),
               ('text_dict_tfidf1_kt', f7, ft7),
               ('text_dict_tfidf2_kt', f8, ft8),
               ('text_w2v_tifidf_kt', f9, ft9),
               ('text_pos_bow1_kt', f10, ft10),
            ]

for i in range(len(all_feats)):
    r = all_feats[i][1].shape[0]
    rt = all_feats[i][2].shape[0]
    y_dense = y.todense()
    y_arr = np.array(y_dense.reshape(r,))[0]
    yt_dense = yt.todense()
    yt_arr = np.array(yt_dense.reshape(rt,))[0]

    dump_svmlight_file(all_feats[i][1], y_arr, config['output'] + all_feats[i][0] + '_train.scl', zero_based=False)
    dump_svmlight_file(all_feats[i][2], yt_arr, config['output']+ all_feats[i][0] + '_test.scl', zero_based=False)