#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: statistical_utils
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:6/10/18
#==================================
from scipy.stats import stats
import numpy as np
from .zlog import log
def significant_test(sample_1,sample_2):
    """
    :param sample_1: list
    :param sample_2: list
    :return:
    """
    s_1 = np.asarray(sample_1)
    s_2 = np.asarray(sample_2)
    difference = s_1 - s_2
    t = (np.mean(difference)) / (difference.std(ddof=1) / np.sqrt(len(difference)))
    s = np.random.standard_t(len(difference), size=100000)
    p = np.sum(s < t) / float(len(s))
    log().info("There is a {} % probability that the paired samples stem from distributions with the same means.".format(
        2 * min(p, 1 - p) * 100))

def sigmoid(x,s):
    return s/(1+np.exp(-x/s))

def softmax(x, temperature = 0.1):
    x = np.asarray(x)
    return np.exp(x/temperature)/np.sum(np.exp(x/temperature))