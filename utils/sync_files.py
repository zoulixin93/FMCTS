#!/usr/bin/python
# encoding: utf-8

#==================================
#@file name: sync_files
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/6,1:00 PM
#==================================

import os

import numpy as np
import ipdb
import argparse
import os
import tensorflow as tf
# adding breakpoint with ipdb.set_trace()
import paramiko

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(*args):
    parser = argparse.ArgumentParser(description='pull or push from remote to home')
    parser.add_argument("-action",dest='action',type=str,default="none",help='pull or push from the remote')
    parser.add_argument("-source",dest='source',type=str,default='./',help='the from directory')
    parser.add_argument("-dest",dest='dest',type=str,default='./',help='to which directory')
    parser.add_argument("-host",dest='host',type=str,default='166.111.71.17',help='host name')
    parser.add_argument("-password",dest='password',type=str,default='zoulixin',help='password')
    parser.add_argument("-username",dest='username',type=str,default='zoulixin',help='username')
    parser.add_argument("-port",dest='port',type=str,default='22',help='port')
    args = parser.parse_args()
    return args

def join_path(path,file):
    if path[-1]=="/":
        return path+file
    else:
        return path+"/"+file

args = parse_arguments()
print(args.source)
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.WarningPolicy)
client.connect(args.host, port=args.port, username=args.username, password=args.password)
if args.action =="pull":
    print(args.source)
    stdin, stdout, stderr = client.exec_command("ls -lsh "+args.source)
    line = str(stdout.read(),encoding='utf-8').split("\n")
    for item in line:
        item = item.split(" ")
        if len(item)>2:
            if item[1][0]=='d':
                pass
            if item[1][0]=='-':
                print("scp "+args.username+"@"+args.host+":"+join_path(args.source,item[-1])+" "+os.path.join(args.dest,item[-1]))
                os.system("scp "+args.username+"@"+args.host+":"+join_path(args.source,item[-1])+" "+os.path.join(args.dest,item[-1]))

