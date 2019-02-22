#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: base
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/9,2:59 PM
#==================================

import numpy as np
import ipdb
import tensorflow as tf
from utils.zlog import log
from .cells import *


class basic_model(object):
    GRAPHS = {}
    SESS = {}
    SAVER = {}

    def c_opt(self,learning_rate,name):
        if str(name).__contains__("adam"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif str(name).__contains__("adagrad"):
            optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif str(name).__contains__("sgd"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif str(name).__contains__("rms"):
            optimizer = tf.train.RMSPropOptimizer(learning_rate)
        elif str(name).__contains__("moment"):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.1)
        return optimizer

    @classmethod
    def create_model(cls, config, variable_scope = "target", trainable = True, graph_name="DEFAULT",task_index=0):
        jobs = config.jobs
        job = list(jobs.keys())[0]
        log.info("CREATE MODEL", config.model, "GRAPH", graph_name, "VARIABLE SCOPE", variable_scope,"jobs",jobs,"job",job,"task_index",task_index)
        cls.CLUSTER = tf.train.ClusterSpec(jobs)
        cls.SERVER = tf.train.Server(cls.CLUSTER, job_name=job, task_index=task_index,config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        if not graph_name in cls.GRAPHS:
            log.info("Adding a new tensorflow graph:",graph_name)
            cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            if not graph_name in cls.SESS:
                cls.SESS[graph_name] = tf.Session(cls.SERVER.target)
                cls.SAVER[graph_name] = tf.train.Saver(max_to_keep=1000)
            cls.SESS[graph_name].run(model.init)
        return {"graph": cls.GRAPHS[graph_name],
               "sess": cls.SESS[graph_name],
               "saver": cls.SAVER[graph_name],
               "model": model,"cluster":cls.CLUSTER,"server":cls.SERVER}

    @classmethod
    def create_model_without_distributed(cls, config, variable_scope = "target", trainable = True, graph_name="DEFAULT"):
        log.info("CREATE MODEL", config.model, "GRAPH", graph_name, "VARIABLE SCOPE", variable_scope)
        if not graph_name in cls.GRAPHS:
            log.info("Adding a new tensorflow graph:",graph_name)
            cls.GRAPHS[graph_name] = tf.Graph()
        with cls.GRAPHS[graph_name].as_default():
            model = cls(config, variable_scope=variable_scope, trainable=trainable)
            if not graph_name in cls.SESS:
                cls.SESS[graph_name] = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
                cls.SAVER[graph_name] = tf.train.Saver(max_to_keep=50)
            cls.SESS[graph_name].run(model.init)
        return {"graph": cls.GRAPHS[graph_name],
               "sess": cls.SESS[graph_name],
               "saver": cls.SAVER[graph_name],
               "model": model}

    def _update_placehoders(self):
        self.placeholders = {"none":{}}
        raise NotImplemented

    def _get_feed_dict(self,task,data_dicts):
        place_holders = self.placeholders[task]
        res = {}
        for key, value in place_holders.items():
            res[value] = data_dicts[key]
        return res

    def __init__(self, config, variable_scope = "target", trainable = True):
        print(self.__class__)
        self.config = config
        self.variable_scope = variable_scope
        self.trainable = trainable
        self.placeholders = {}
        self._build_model()

    def _build_model(self):
        with tf.variable_scope(self.variable_scope):
            self._create_placeholders()
            self._create_global_step()
            self._update_placehoders()
            self._create_inference()
            if self.trainable:
                self._create_optimizer()
            self._create_intializer()

    def _create_global_step(self):
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_intializer(self):
        with tf.name_scope("initlializer"):
            self.init = tf.global_variables_initializer()

    def _create_placeholders(self):
        raise NotImplementedError

    def _create_inference(self):
        raise NotImplementedError

    def _create_optimizer(self):
        raise NotImplementedError

    def build_cell(self,rnn_type,initializer,hidden,input_data,initial_state,name="test"):
        if rnn_type == "nlstm":
            cell = tf.contrib.rnn.MultiRNNCell([nlstm(hidden,name=name)]*self.config.RNN_LAYER)
            return tf.nn.dynamic_rnn(cell, input_data,
                                     initial_state=(tf.nn.rnn_cell.LSTMStateTuple(c=tf.zeros_like(initial_state[0]),
                                                                                  h=initial_state[0]),),
                                     dtype=tf.float32,
                                     time_major=True)
