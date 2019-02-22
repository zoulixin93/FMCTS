#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: argments
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:18/10/19,上午4:32
#==================================

import numpy as np
import ipdb
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(*args):
    parser = argparse.ArgumentParser(description='pull or push from remote to home')
    parser.add_argument("-action",dest='action',type=str,default="pull",help='pull or push from the remote')
    parser.add_argument("-from",dest='from',type=str,default='./',help='the from directory')
    parser.add_argument("-to",dest='to',type=str,default='./',help='to which directory')
    args = parser.parse_args()
    return args
