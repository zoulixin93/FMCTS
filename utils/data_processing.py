#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: data_processing
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:6/12/18
#==================================

import numpy as np
import ipdb
import pandas as pd

def read_csv2data_frame(data_path = "",sep="\t",head=None,cols=[]):
    res = pd.read_csv(data_path,sep=sep,header=head,names=cols)
    return res