#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: arguments
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:18/10/23,下午10:42
#==================================

import numpy as np
import ipdb
import argparse
import utils
import os
import tensorflow as tf
from metrics.metrics import *
# adding breakpoint with ipdb.set_trace()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(*args):
    parser = argparse.ArgumentParser(description='robust model based reinforcement learning for recommendation')
    parser.add_argument('-model',dest='model',type = str,default="debug",help='model_name')
    parser.add_argument('-data_name',dest='data_name',type = str,default="date_name",help='data name')
    parser.add_argument('-log_path',dest='log_path',type = str,default="./log",help='the log path')
    parser.add_argument('-root_path',dest='root_path',type = str,default="./data/ml-100k",help='root data path')
    parser.add_argument('-rating_path',dest='rating_path',type = str,default="rat.dat",help='rating data')
    parser.add_argument('-cat_path',dest='cat_path',type = str,default="cat.dat",help='category data')
    parser.add_argument('-item_num',dest='item_num',type = int,default=100,help="the item num")
    parser.add_argument('-user_num',dest='user_num',type = int,default=100,help="the user num")
    parser.add_argument('-rating_num',dest='rating_num',type = int,default=5,help="the number of rating")
    parser.add_argument('-cat_n1_num',dest='cat_n1_num',type = int,default=100,help="the catn1 num")
    parser.add_argument('-cat_n2_num',dest='cat_n2_num',type = int,default=100,help="the catn2 num")
    parser.add_argument('-top_k',dest='top_k',type = int,default=3,help="recommend how much")
    parser.add_argument('-used4train',dest='used4train',type =float,default=0.85,help="the percent for training")
    parser.add_argument('-accuracy',dest='accuracy',type =str,default="ndcg",help="the function for accuracy")
    parser.add_argument('-diversity',dest='diversity',type =str,default="alpha_ndcg",help="the function for ndcg")
    parser.add_argument('-latent_factor',dest='latent_factor',type =int,default=20,help="the latent factor number")
    parser.add_argument('-cell_type',dest='cell_type',type =str,default="nlstm",help="the nlstm")
    parser.add_argument('-rnn_layer',dest='rnn_layer',type =int,default=1,help="the rnn layer")
    parser.add_argument('-learning_rate',dest='learning_rate',type =float,default=0.01,help="the learning rate")
    parser.add_argument('-optimizer_name',dest='optimizer_name',type =str,default="sgd",help="the optimizer")
    parser.add_argument('-memory_capacity',dest='memory_capacity',type =int,default=10000,help="the optimizer")
    parser.add_argument('-discount_factor',dest='discount_factor',type =float,default=1.0,help="the discount factor")
    parser.add_argument('-c_puct',dest='c_puct',type =float,default=5.0,help="the cpuct")
    parser.add_argument('-n_playout',dest='n_playout',type =int,default=20,help="the playout number")
    parser.add_argument('-temperature',dest='temperature',type =float,default=1.0,help="the temperature")
    parser.add_argument('-update_frequency',dest='update_frequency',type =int,default=2,help="the update frequency")
    parser.add_argument('-batch_size',dest='batch_size',type =int,default=64,help="the batch size")
    parser.add_argument('-epoch',dest='epoch',type =int,default=100000,help="epoch")
    parser.add_argument('-evaluate_num',dest='evaluate_num',type =int,default=300,help="evaluate_user_num")
    parser.add_argument('-delete_previous',dest='delete_previous',type =str2bool,default="False",help="delete saved files")
    parser.add_argument('-saved_model_path',dest='saved_model_path',type =str,default="saved_model",help="saved_model")
    parser.add_argument('-job_ports',dest='job_ports',type =eval,default="[20000,20001,20002,20003,20004,20005,20006,20007]",help="saved_model")
    parser.add_argument('-task',dest='task',type =str,default="train",help="train")
    # parser.add_argument('-evaluate_num',dest='evaluate_num',type =int,default=300,help="evaluate_user_num")
    args = parser.parse_args()
    print(len(args.job_ports))
    args.jobs = {"local":["localhost:"+str(item) for item in args.job_ports]}
    args.key_words = [args.data_name,args.model,args.c_puct,args.n_playout,args.temperature]
    args.model_id = "_".join([str(item) for item in args.key_words]).replace(".","_")
    args.GPU_OPTION = tf.GPUOptions(allow_growth=True)
    args.RANDOM_SEED = 123
    args.RNN_LAYER = 1
    if args.accuracy =="ndcg":args.accuracy = ndcg
    if args.diversity =="alpha_ndcg": args.diversity = alpha_ndcg
    return args

def initialize(config,*args):
    if not os.path.isdir(config.log_path):
        os.mkdir(config.log_path)
    if config.delete_previous:
        print("delete previous log")
        os.system("rm "+os.path.join(config.log_path,config.model_id+"_"+config.task)+"*")
    if config.task == "train" and config.delete_previous:
        print("delete previous saved_model")
        os.system("rm -r "+os.path.join(config.saved_model_path,config.model_id)+"*")
    utils.log.set_log_path(os.path.join(config.log_path,config.model_id+"_"+config.task)+"_"+str(utils.get_now_time())+".log")
    utils.log.info('saving log file in '+utils.log().log_path)
    utils.log.structure_info("config for experiments",list(vars(config).items()))
    return config