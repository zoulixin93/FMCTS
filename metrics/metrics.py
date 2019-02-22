#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: metrics
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/8,5:19 PM
#==================================

import numpy as np
import ipdb
from collections import Counter
import math
import copy as cp

def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum( r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg(item_value={},top_k_item=[]):
    score = [item_value[item] for item in top_k_item]
    ideal_list = list(sorted(item_value.values(),reverse=True))[:len(score)]
    i_dcg = dcg_at_k(ideal_list,len(ideal_list),method=1)
    dcg = dcg_at_k(score,len(score),method=1)
    return dcg/i_dcg

def alpha_dcg(tops, k = 10, alpha = 0.5, *args):
    items = []
    G = []
    for i,item in enumerate(tops[:k]):
        items+=item
        G.append(sum(map(lambda x: math.pow(alpha,x-1),dict(Counter(items)).values()))/math.log(i+2,2))
    return sum(G)

def generate_ideal_topic_list(topic_list,topic_num=3234):
    x = topic_list
    new_x = np.zeros((len(x),topic_num))
    for i,item in enumerate(x):
        for j in item:
            new_x[i,j] = 1
    raw_x = cp.deepcopy(new_x)
    score_x = cp.deepcopy(new_x)
    res = []
    while True:
        key = np.argmax(np.sum(score_x,1))
        res.append(x[key])
        if len(res) == len(x):
            break
        temp_x = np.repeat(np.reshape(raw_x[key,:],(1,-1)),len(x),0)
        new_x = new_x-np.multiply(raw_x,temp_x)*1.01
        score_x[np.nonzero(new_x<0)] = np.power(10,new_x[np.nonzero(new_x<0)]-1)
        score_x[key,:] = -1
    return res

def alpha_ndcg(item_tops={},top_k_item = [],topic_num = 10,alpha=0.5):
    ideal_list = generate_ideal_topic_list(list(item_tops.values()),topic_num)
    a_dcg = alpha_dcg([item_tops[item] for item in top_k_item], k = len(top_k_item), alpha = alpha)
    i_dcg = alpha_dcg(ideal_list, k=len(top_k_item), alpha=alpha)
    return a_dcg/i_dcg



