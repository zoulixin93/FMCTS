#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: functions
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/8,4:42 PM
#==================================

import numpy as np
import ipdb

convert2int = lambda x:[eval(item) for item in x]
convkv2dict = lambda x:{key:value for key,value in x}
convert2str = lambda x:[str(item) for item in x]