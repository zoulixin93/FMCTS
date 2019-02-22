#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: train
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/12,11:55 AM
#==================================

import numpy as np
import ipdb
from env import env
from utils.data_structure import type_memory
import os
from .mcts import MCTS
import copy as cp
from utils.zlog import log
import time
from multiprocessing import Process,Lock

def collecting_training_samples(config,mcts,env,temp):
    env.reset4train()
    terminal = False
    reward = 0
    data = []
    while not terminal:
        act,acts,act_probs = mcts.get_action(cp.deepcopy(env),
                                             temp,training=True)
        temp = [env.uid,env.profile,cp.deepcopy(env.trajectory),cp.deepcopy(env.cat_trajectory),env.node_type]
        candidate,node_type,reward,terminal = env.step(act)
        temp.extend([act,acts,act_probs,reward])
        data.append(temp)
    for item in data: item[-1] = reward
    print(reward)
    return data,reward


def train(config,env,task_index,gpu_index,lock=Lock):
    buffer = type_memory(2,config.memory_capacity)
    from function_approximation import rnn_model
    import utils
    np.random.seed(int(time.time()))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    # lock.acquire()
    pv_agent = rnn_model.create_model(config, task_index=task_index)
    # utils.save_model(pv_agent["saver"],
    #                  utils.create_saved_path(0, config.saved_model_path, config.model_id),
    #                  pv_agent["sess"],
    #                  pv_agent["model"].global_step)
    # lock.release()
    mcts = MCTS(pv_agent, config.c_puct, config.n_playout,config.discount_factor)
    for e in range(1,config.epoch + 1):
        average_reward = []
        for i in range(config.update_frequency):
            print(task_index,"collecting data")
            data,reward = collecting_training_samples(config,mcts,env,config.temperature/(e+0.000001))
            average_reward.append(reward)
            for item in data:
                if item[4]==0:
                    buffer.put(item,0)
                elif item[4]==1:
                    buffer.put(item,1)
        print(task_index,"finish collecting")
        log.info(str(e),"process",str(task_index),"collecting trajectory reward",np.mean(average_reward))
        batch = buffer.sample_batch(config.batch_size)
        lock.acquire()
        try:
            p1,v_1,p2,v_2=pv_agent["model"].optimize_model_batch(pv_agent["sess"],batch)
            log.info("\t".join([str(item) for item in [e,"process",task_index,"policy_1",p1,"value_1",v_1,"policy2",p2,"value_2",v_2]]))
            utils.save_model(pv_agent["saver"],
                             utils.create_saved_path(e, config.saved_model_path,config.model_id),
                             pv_agent["sess"],
                             pv_agent["model"].global_step)
        except:
            pass
        lock.release()

class run_all(object):
    def __init__(self,config):
        self.config = config
        self.env = env(self.config)

    def run(self):
        train_process = []
        l = Lock()
        for i in range(len(self.config.job_ports)):
            train_process.append(Process(target=train,args=(self.config,cp.deepcopy(self.env),i,i%4,l)))
            train_process[-1].start()
            time.sleep(3)
        for i in range(len(self.config.jobs)):
            train_process[i].join()

class evaluate(object):
    def __init__(self,config):
        self.config = config
        self.env = env(self.config)
        from function_approximation import rnn_model
        np.random.seed(int(time.time()))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
        self.pv_agent = rnn_model.create_model_without_distributed(self.config)

    def re_read_saved_model(self):
        saved_model = os.listdir(os.path.abspath(self.config.saved_model_path))
        saved_model = [int(item.split("_")[-1]) for item in saved_model if item.__contains__(self.config.model_id)]
        saved_model.sort()
        return saved_model

    def run(self):
        samples = []
        while True:
            for num in self.re_read_saved_model():
                if num in samples:
                    continue
                else:
                    try:
                        self.evaluate(num)
                        samples.append(num)
                    except:
                        print("load error, does not load the model")
            time.sleep(3)

    def evaluate(self,num):
        import tensorflow as tf
        path = os.path.join(self.config.saved_model_path, self.config.model_id + "_" + str(num))
        self.pv_agent['saver'].restore(self.pv_agent['sess'], tf.train.latest_checkpoint(path))
        r = []
        uidss = np.random.choice(range(1,self.config.user_num),(self.config.evaluate_num,),replace=False)
        for uid in uidss:
            try:
                self.env.reset4evaluate(uid)
                terminal = False
                reward = 0
                while not terminal:
                    _action_probs, next_state_value = self.pv_agent["model"].get_actions_probability_model(self.pv_agent["sess"], self.env)
                    act, probability = zip(*_action_probs)
                    action = act[np.argmax(probability)]
                    candidate, node_type, reward, terminal = self.env.step(action)
                r.append((reward,self.env.accuracy,self.env.diversity))
            except:
                pass
        print("####"*5)
        f1,a,d = zip(*r)
        log.info("evaluate",num, "average_reward", np.mean(f1),"accuracy",np.mean(a),"diversity",np.mean(d))
        pass




