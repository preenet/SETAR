# -*- coding: utf-8 -*-
"""
# Implementation of existing pos-tagging techniques.  
# TODO: 1. test performance of each
@Authors: pree.t@cmu.ac.th
"""
import numpy as np
from sklearn.preprocessing import LabelBinarizer

__all__ = ['word_tag', 'tag', 'tag_emoj', 'flatten', 'onehot_label']

# concate word and tag with underscore (มัน_ADV)
def word_tag(tagged):
    tag_list = []
    for item in tagged:
        tag = ['_'.join(map(str, el)) for el in item]
        tag_list.append(' '.join(tag))
    return tag_list

# use only tag
def tag(tagged):
    tag_list = []
    for item in tagged:
        tmp = [el[1] for el in item]
        tag = ' '.join(map(str, tmp))
        tag_list.append(tag)
    return tag_list

# create custom tag for emoticon word
def tag_emoj(tagged):
    tag_list = []
    
    for i in range(len(tagged)):
        pl = [list(el) for el in tagged[i]]
        tag_list.append(pl)

    for item in tag_list:
        for i, tag in enumerate(item):
            for j, word in enumerate(tag):
                if "EMJ" in word:
                    item[i][1] = "EMOJI"
    return tag_list

# this approach convert nest-list to simply list of word follows by tag ['word', 'NOUN']
def flatten(tagged):
    tag_list = []
    for item in tagged:
        tag = list(sum(item, ()))
        tag_list.append(tag)
    return tag_list

# use one-hot-encoding for each word base on taggin scheme

def onehot_label(tagged):
    pos_tag_ud = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'EMOJI']
    pos_tag_ud_arr = np.array(pos_tag_ud)
    tag_list = []

    for item in tagged:
            lb = LabelBinarizer().fit(pos_tag_ud_arr)
            tmp = [el[1] for el in item]
            tag_only = ' '.join(map(str, tmp))
            tag_only = list(tag_only.split(" "))
            onehot_data = lb.transform(tag_only)
            
            tag_list.append( flatten(onehot_data) )
    return  tag_list