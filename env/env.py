#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: env
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/8,1:44 PM
#==================================

import numpy as np
import ipdb
import os
from utils.functions import *
from metrics import metrics
import copy as cp

class env(object):
    def __init__(self,config):
        self.config = config
        self.init()

    def init(self):
        self.load_data()

    def load_data(self):
        self.data = {}
        self.cat1d = {}
        self.cat2d = {}
        with open(os.path.join(self.config.root_path,self.config.rating_path),"r") as f:
            for line in f.readlines():
                uid,iid,rating = convert2int(line.strip("\n").split("\t"))
                if uid not in self.data:
                    self.data[uid] = [(iid,rating)]
                else:
                    self.data[uid].append((iid,rating))
        with open(os.path.join(self.config.root_path,self.config.cat_path),"r") as f:
            for line in f.readlines():
                line = convert2int(line.strip("\n").split("\t"))
                self.cat1d[line[0]] = line[1]
                self.cat2d[line[0]] = line[2:]

    def reset4train(self):
        self.node_type = 0
        self.uid = np.random.choice(range(1,self.config.user_num))
        np.random.shuffle(self.data[self.uid])
        # items = self.data[self.uid][:int(len(self.data[self.uid])*self.config.used4train)]
        items = self.data[self.uid][:int(len(self.data[self.uid])*1.0)]
        num4state = np.random.choice(range(len(items)-self.config.top_k))
        used_as_profile = np.random.choice(range(len(items)),replace=False,size=(num4state,))
        self.profile = convkv2dict([items[item] for item in used_as_profile])
        self.candidate = convkv2dict([items[item] for item in range(len(items)) if item not in used_as_profile])
        self.trajectory = []
        self.cat_trajectory = []
        self.naive_candiate = cp.deepcopy(self.candidate)
        return self.uid,self.profile,list(set([self.cat1d[item] for item in self.candidate.keys()]))

    def reset4evaluate(self,uid):
        self.node_type = 0
        self.uid = uid
        self.accuracy = 0
        self.diversity = 0
        self.candidate = convkv2dict(self.data[self.uid][int(len(self.data[self.uid]) * self.config.used4train):])
        self.profile = convkv2dict(self.data[self.uid][:int(len(self.data[self.uid]) * self.config.used4train)])
        self.trajectory = []
        self.cat_trajectory = []
        self.naive_candiate = cp.deepcopy(self.candidate)
        return self.uid,self.profile,list(set([self.cat1d[item] for item in self.candidate.keys()]))

    def _update_node_type(self):
        self.node_type+=1
        self.node_type=self.node_type%2

    def get_node_type(self):
        return self.node_type

    def step(self,item):
        if self.node_type==0:
            self._update_node_type()
            self.cat_trajectory.append(item)
            return list([ii for ii in self.candidate.keys() if self.cat1d[ii]==item]), self.node_type-1, 0, False
        elif self.node_type==1:
            self._update_node_type()
            if item not in self.candidate:
                return "error recommendation"
            self.trajectory.append(item)
            terminal = False
            reward = 0
            if len(self.trajectory)>=self.config.top_k:
                terminal = True
                self.accuracy = self.config.accuracy(self.naive_candiate,self.trajectory)
                self.diversity = self.config.diversity({item:self.cat2d[item] for item in self.naive_candiate.keys()},
                                            self.trajectory,
                                            self.config.cat_n2_num)
                reward = 0.5 * self.accuracy+0.5 * self.diversity
            self.candidate.pop(item)
            return list(set([self.cat1d[item] for item in self.candidate.keys()])),self.node_type+1,reward,terminal

    def get_sensible_actions(self):
        if self.node_type==1:
            return list([ii for ii in self.candidate.keys() if self.cat1d[ii]==self.cat_trajectory[-1]]),self.node_type
        if self.node_type==0:
            return list(set([self.cat1d[item] for item in self.candidate.keys()])),self.node_type



