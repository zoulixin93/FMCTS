#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: train_file
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/7,12:38 AM
#==================================

import numpy as np
import ipdb
from arguments import parse_arguments,initialize
import utils
import os
import time
from train import run_all,evaluate

if __name__ == '__main__':
    np.random.seed(int(time.time()))
    config = parse_arguments()
    config = initialize(config)
    if config.task =="train":
        run_all(config).run()
    elif config.task =="evaluate":
        evaluate(config).run()