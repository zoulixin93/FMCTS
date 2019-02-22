#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: time_analyses
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:8/22/18,10:38 AM
#==================================

import numpy as np
import ipdb
import time
from .zlog import log
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            log.info('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed