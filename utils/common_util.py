#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: common_util
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2/22/18
#==================================
from datetime import datetime
import os
import numpy as np
import ipdb
import smtplib
import pickle
def get_now_time():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def dir_files(path):
    path = os.path.abspath(path)
    return list(map(lambda x:os.path.join(path,x),os.listdir(path)))

def obj_dic(d):
    top = type('new', (object,), d)
    seqs = tuple, list, set, frozenset
    for i, j in d.items():
        if isinstance(j, dict):
            setattr(top, i, obj_dic(j))
        elif isinstance(j, seqs):
            setattr(top, i,
                type(j)(obj_dic(sj) if isinstance(sj, dict) else sj for sj in j))
        else:
            setattr(top, i, j)
    return top

def text_message2phone(*info):
    from twilio.rest import Client
    info = [str(item) for item in info]
    temp = ",".join(info)
    # Find these values at https://twilio.com/user/account
    account_sid = "AC5546f0add6a15d278e5c590d77451b6c"
    auth_token = "39c6c470d30f9f7f98a0df00e769bdef"
    client = Client(account_sid, auth_token)
    phone_numbers = ["+8618801398829"]
    for item in phone_numbers:
        client.api.account.messages.create(
        to=item,
        from_="+18323613509",
        body=temp)

def saveobj2files(obj=[1],path=""):
    pickle.dump(obj, open(path, "wb"))

def loadobj4files(path=""):
    return pickle.load(open(path,"rb"))
