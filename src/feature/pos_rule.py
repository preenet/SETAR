# -*- coding: utf-8 -*-
"""
# Implementation of existing pos-tagging techniques.  
# TODO: 1. test performance of each
@Authors: pree.t@cmu.ac.th
"""


__all__ = ['word_tag', 'tag', 'tag_emoj', 'flatten']

# concate word and tag with underscore (มัน_ADV)
def word_tag(pos):
    tag_list = []
    for item in pos:
        tag = ['_'.join(map(str, el)) for el in item]
        tag_list.append(' '.join(tag))
    return tag_list

# use only tag
def tag(pos):
    tag_list = []
    for item in pos:
        tmp = [el[1] for el in item]
        tag = ' '.join(map(str, tmp))
        tag_list.append(tag)
    return tag_list

# create custom tag for emoticon word
def tag_emoj(pos):
    tag_list = []
    
    for i in range(len(pos)):
        pl = [list(el) for el in pos[i]]
        tag_list.append(pl)

    for item in tag_list:
        for i, tag in enumerate(item):
            for j, word in enumerate(tag):
                if "EMJ" in word:
                    item[i][1] = "EMOJI"
    return tag_list

# this approach convert nest-list to simply list of word follows by tag ['word', 'NOUN']
def flatten(text):
    res = list(sum(text, ()))
    return res