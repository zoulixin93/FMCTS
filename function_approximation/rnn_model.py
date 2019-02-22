#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: rnn_model
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/9,3:17 PM
#==================================

import numpy as np
import ipdb
from .base import basic_model
import tensorflow as tf
from utils.statistical_utils import softmax


def softmax_cross_entropy_with_logits_mask(t,p,m):
    p = tf.nn.softmax(p,-1)
    # cross_entropy = -m*(t*tf.log(p) + (1.0-t)*tf.log(1.0-p))
    cross_entropy = -m*(t*tf.log(p))
    return cross_entropy

class rnn_model(basic_model):
    def _create_placeholders(self):
        self.uid = tf.placeholder(tf.int32, (None,),name="uid")
        self.profile = tf.placeholder(tf.int32, (None,None),name='profile')
        self.profile_rating = tf.placeholder(tf.float32, (None,None),name="profile_rating")
        self.trajectory = tf.placeholder(tf.int32, (None,None),name="trajectory")
        self.t_i_t = tf.placeholder(tf.int32, (None,2),name="t_i_t")
        self.t_i_c_t = tf.placeholder(tf.int32, (None,2),name="t_i_c_t")
        self.cat_trajectory = tf.placeholder(tf.int32,(None,None),name="cat_trajectory")
        self.v_value = tf.placeholder(tf.float32, (None,),name="v_value")
        self.s1_policy = tf.placeholder(tf.float32, (None,self.config.cat_n1_num),name="s1_policy")
        self.s1_policy_mask = tf.placeholder(tf.float32, (None,self.config.cat_n1_num),name="s1_policy_mask")
        self.s2_policy = tf.placeholder(tf.float32, (None,self.config.item_num),name="s2_policy")
        self.s2_policy_mask = tf.placeholder(tf.float32, (None, self.config.item_num),name="s2_policy_mask")

    def _update_placehoders(self):
        self.placeholders["all"] = {"uid":self.uid,
                                    "profile":self.profile,
                                    "profile_rating":self.profile_rating,
                                    "trajectory":self.trajectory,
                                    "cat_trajectory":self.cat_trajectory,
                                    "t_i_t":self.t_i_t,
                                    "t_i_c_t":self.t_i_c_t,
                                    "v_value":self.v_value,
                                    "s1_policy":self.s1_policy,
                                    "s1_policy_mask":self.s1_policy_mask,
                                    "s2_policy":self.s2_policy,
                                    "s2_policy_mask":self.s2_policy_mask}
        predicts = ["uid","profile","profile_rating","trajectory","cat_trajectory","t_i_t","t_i_c_t"]
        self.placeholders["predict"]={item:self.placeholders["all"][item] for item in predicts}
        o_1 = ["uid","profile","profile_rating","trajectory",
               "cat_trajectory","t_i_t","t_i_c_t","s1_policy","s1_policy_mask","v_value"]
        self.placeholders["optimizer_1"] = {item:self.placeholders["all"][item] for item in o_1}
        o_2 = ["uid", "profile", "profile_rating", "trajectory",
               "cat_trajectory", "t_i_t", "t_i_c_t", "s2_policy", "s2_policy_mask","v_value"]
        self.placeholders["optimizer_2"] = {item: self.placeholders["all"][item] for item in o_2}

    def _create_inference(self):
        initializer = tf.random_uniform_initializer(0, 0.1, seed=self.config.RANDOM_SEED)
        item_feature = tf.Variable(tf.random_uniform([self.config.item_num, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='item_feature')
        cat_feature = tf.Variable(tf.random_uniform([self.config.cat_n1_num, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='cat_feature')
        user_feature = tf.Variable(tf.random_uniform([self.config.user_num, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='user_feature')
        # transfer the projection weights
        projection_weight = [tf.zeros((self.config.latent_factor,self.config.latent_factor))]+\
                            [tf.Variable(tf.random_uniform([self.config.latent_factor, self.config.latent_factor],
                                                     0, 0.1),trainable=self.trainable,name='projections_'+str(i))
                             for i in range(self.config.rating_num)]
        profile_embedding = tf.nn.embedding_lookup(item_feature,self.profile)
        for i in range(self.config.rating_num+1):
            temp = tf.tensordot(profile_embedding,projection_weight[i],axes=[[2],[0]])
            mask = tf.cast(self.profile_rating-i*1.0,tf.bool)
            mask = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, self.config.latent_factor])
            profile_embedding = tf.where(mask,profile_embedding,temp)

        profile_embedding = tf.reduce_sum(profile_embedding,1)
        profile_embedding = tf.nn.softmax(profile_embedding,1)
        trajectory_embedding = tf.nn.embedding_lookup(item_feature,self.trajectory)
        cat_trajectory_embedding = tf.nn.embedding_lookup(cat_feature,self.cat_trajectory)
        u_embedding = tf.nn.embedding_lookup(user_feature,self.uid)
        initial_state_traj = tf.layers.dense(inputs=tf.concat([u_embedding,profile_embedding],1),
                                              kernel_initializer=initializer,
                                              units=self.config.latent_factor,
                                              activation=tf.nn.relu,
                                              trainable=True)
        initial_state_cat = tf.layers.dense(inputs=tf.concat([u_embedding,profile_embedding],1),
                                              kernel_initializer=initializer,
                                              units=self.config.latent_factor,
                                              activation=tf.nn.relu,
                                              trainable=True)
        initial_state_traj = tuple([initial_state_traj for _ in range(self.config.rnn_layer)])
        initial_state_cat = tuple([initial_state_cat for _ in range(self.config.rnn_layer)])
        self.rnn_traj,rnn_state_traj = self.build_cell(self.config.cell_type,
                                                     initializer,
                                                     self.config.latent_factor,
                                                     trajectory_embedding,initial_state_traj,name="item_seq")
        rnn_traj = tf.gather_nd(self.rnn_traj,self.t_i_t)
        self.rnn_traj_cat,rnn_state_traj_cat = self.build_cell(self.config.cell_type,
                                                               initializer,
                                                               self.config.latent_factor,
                                                               cat_trajectory_embedding,initial_state_cat,name="cat_seq")
        rnn_traj_cat = tf.gather_nd(self.rnn_traj_cat,self.t_i_c_t)

        feature = tf.nn.relu(tf.concat([profile_embedding,u_embedding,rnn_traj,rnn_traj_cat],1))
        hidden = tf.layers.dense(inputs =feature,kernel_initializer=initializer,
                                                    units=self.config.latent_factor,
                                                    activation=tf.nn.relu,
                                                    trainable=True)
        self.p_s1_policy = tf.layers.dense(inputs=hidden,kernel_initializer=initializer,
                                           units=self.config.cat_n1_num,activation=None,trainable=True)
        self.p_s2_policy = tf.layers.dense(inputs=hidden,kernel_initializer=initializer,
                                           units=self.config.item_num,activation=None,trainable=True)
        self.p_v_value = tf.reshape(tf.layers.dense(inputs=hidden,kernel_initializer=initializer,
                                           units=1,activation=None,trainable=True),(-1,))

    def _create_optimizer(self):
        self.policy1_loss =tf.reduce_sum(softmax_cross_entropy_with_logits_mask(self.s1_policy,self.p_s1_policy,self.s1_policy_mask))
        self.value_loss = 10.0*tf.losses.mean_squared_error(self.v_value,self.p_v_value)
        self.policy2_loss =tf.reduce_sum(softmax_cross_entropy_with_logits_mask(self.s2_policy,self.p_s2_policy,self.s2_policy_mask))
        self.loss_1 = self.policy1_loss + self.value_loss
        self.loss_2 = self.policy2_loss + self.value_loss
        optimizer_1 = self.c_opt(self.config.learning_rate,self.config.optimizer_name)
        gvs = optimizer_1.compute_gradients(self.loss_1)
        self.optimizer_1 = optimizer_1.apply_gradients(gvs,global_step=self.global_step)

        optimizer_2 = self.c_opt(self.config.learning_rate,self.config.optimizer_name)
        gvs = optimizer_2.compute_gradients(self.loss_2)
        self.optimizer_2 = optimizer_2.apply_gradients(gvs,global_step=self.global_step)

    def policy_value_fn(self,sess,data):
        feed_dicts = self._get_feed_dict("predict", data)
        return sess.run([self.p_s1_policy,self.p_s2_policy,self.p_v_value], feed_dicts)

    def policy_value_with_type(self,sess,data,node_type=0):
        feed_dicts = self._get_feed_dict("predict", data)
        if node_type==0:
            return sess.run([self.p_s1_policy,self.p_v_value], feed_dicts)
        elif node_type==1:
            return sess.run([self.p_s2_policy,self.p_v_value], feed_dicts)

    def get_actions_probability_model(self,sess,env):
        # data = self.conv_env2dict(env)
        # sensible_action = env.get_sensible_actions()[0]
        # feed_dicts = self._get_feed_dict("predict", data)
        # if env.node_type==0:
        #     p_policy,v_value = sess.run([self.p_s1_policy, self.p_v_value], feed_dicts)
        # elif env.node_type==1:
        #     p_policy,v_value = sess.run([self.p_s2_policy, self.p_v_value], feed_dicts)
        # else:
        #     p_policy, v_value = None,None
        # p = [(item,p_policy[0][item]) for item in sensible_action]
        # return p, v_value[0]

        data = self.conv_env2dict(env)
        sensible_action = env.get_sensible_actions()[0]
        feed_dicts = self._get_feed_dict("predict", data)

        if env.node_type==0:
            p_policy,v_value = sess.run([self.p_s1_policy,self.p_v_value], feed_dicts)
        elif env.node_type==1:
            p_policy,v_value = sess.run([self.p_s2_policy,self.p_v_value], feed_dicts)
        else:
            p_policy,v_value = None,None
        p = [(item,p_policy[0][item]) for item in sensible_action]
        return p,v_value[0]

    def get_actions_probability(self,sess,env):
        # data = self.conv_env2dict(env)
        # sensible_action = env.get_sensible_actions()[0]
        # feed_dicts = self._get_feed_dict("predict", data)
        # if env.node_type==0:
        #     p_policy,v_value = sess.run([self.p_s1_policy, self.p_v_value], feed_dicts)
        # elif env.node_type==1:
        #     p_policy,v_value = sess.run([self.p_s2_policy, self.p_v_value], feed_dicts)
        # else:
        #     p_policy, v_value = None,None
        # all_score = softmax([p_policy[0][item] for item in sensible_action],200)
        # p = [(item,all_score[i]) for i,item in enumerate(sensible_action)]
        # return p, v_value[0]

        data = self.conv_env2dict(env)
        sensible_action = env.get_sensible_actions()[0]
        feed_dicts = self._get_feed_dict("predict", data)
        if env.node_type==0:
            p_policy = sess.run(self.p_s1_policy, feed_dicts)
        elif env.node_type==1:
            p_policy = sess.run(self.p_s2_policy, feed_dicts)
        else:
            p_policy = None

        all_score = softmax([p_policy[0][item] for item in sensible_action],30)
        p = [(item,all_score[i]) for i,item in enumerate(sensible_action)]
        return p,0.0

    # def get_actions_probability(self,sess,env):
    #     # data = self.conv_env2dict(env)
    #     sensible_action = env.get_sensible_actions()[0]
    #     # feed_dicts = self._get_feed_dict("predict", data)
    #     # if env.node_type==0:
    #     #     p_policy,v_value = sess.run([self.p_s1_policy, self.p_v_value], feed_dicts)
    #     # elif env.node_type==1:
    #     #     p_policy,v_value = sess.run([self.p_s2_policy, self.p_v_value], feed_dicts)
    #     # else:
    #     #     p_policy, v_value = None,None
    #     # prob = 1.0/len(sensible_action)
    #     p = [(item,0.1) for item in sensible_action]
    #     return p,0.0


    def optimize_model(self, sess, data, node_type=0):
        # ipdb.set_trace()
        if node_type==0:
            feed_dicts = self._get_feed_dict("optimizer_1", data)
            return sess.run([self.policy1_loss,self.value_loss,self.optimizer_1], feed_dicts)[:-1]
        if node_type==1:
            feed_dicts = self._get_feed_dict("optimizer_2", data)
            return sess.run([self.policy2_loss,self.value_loss,self.optimizer_2], feed_dicts)[:-1]

    def optimize_model_batch(self,sess,data_all):
        data = self.convert_batch_data(data_all[:self.config.batch_size],type=0)
        policy_loss1,value1_loss = self.optimize_model(sess,data,0)
        data = self.convert_batch_data(data_all[self.config.batch_size:],type=1)
        policy_loss2,value2_loss = self.optimize_model(sess,data,1)
        return policy_loss1,value1_loss,policy_loss2,value2_loss

    def convert_batch_data(self,data,type=0):
        uid = [item[0] for item in data]
        profile = np.transpose(self.convert_seq2matrix([list(item[1].keys()) for item in data])[0])
        profile_rating = np.transpose(self.convert_seq2matrix([list(item[1].values()) for item in data])[0])
        trajectory, t_i_t = self.convert_seq2matrix([[0]+item[2] for item in data])
        cat_trajectory, t_i_c_t = self.convert_seq2matrix([[0]+item[3] for item in data])
        policy = [item[5] for item in data]
        policy_mask = [item[6] for item in data]
        policy_probability = [item[7] for item in data]
        v_value = [item[-1] for item in data]
        data = {"uid": uid,
                "profile": profile,
                "profile_rating": profile_rating,
                "trajectory": trajectory,
                "t_i_t": t_i_t,
                "cat_trajectory": cat_trajectory,
                "t_i_c_t": t_i_c_t,
                "v_value": v_value}
        if type==0:
            policy,policy_mask = self.get_target_policy(policy,policy_mask,policy_probability,self.config.cat_n1_num)
            data["s1_policy"] = policy
            data["s1_policy_mask"] = policy_mask
        if type==1:
            policy,policy_mask = self.get_target_policy(policy,policy_mask,policy_probability,self.config.item_num)
            data["s2_policy"] = policy
            data["s2_policy_mask"] = policy_mask
        return data

    def get_target_policy(self,policy,policy_mask,policy_probability,dim):
        p = np.zeros((len(policy),dim))
        # mask = np.zeros((len(policy),dim))
        mask = np.ones((len(policy),dim))
        for i,item in enumerate(policy):
            p[i,item] = 1
            for j in policy_mask[i]:
                mask[i,j] = 1
        return p,mask

    def conv_env2dict(self,env):
        uid = [env.uid]
        trajectory = [[0]+env.trajectory]
        cat_trajectory = [[0]+env.cat_trajectory]
        profile = [list(env.profile.keys())]
        profile_rating = [list(env.profile.values())]
        trajectory,t_i_t = self.convert_seq2matrix(trajectory)
        cat_trajectory,t_i_c_t = self.convert_seq2matrix(cat_trajectory)
        data = {"uid":uid,
                "profile":np.transpose(self.convert_seq2matrix(profile)[0]),
                "profile_rating":np.transpose(self.convert_seq2matrix(profile_rating)[0]),
                "trajectory":trajectory,
                "t_i_t":t_i_t,
                "cat_trajectory":cat_trajectory,
                "t_i_c_t":t_i_c_t}
        return data

    def convert_seq2matrix(self,trajectory=[]):
        max_length = max([len(item) for item in trajectory])
        matrix = np.zeros((max_length,len(trajectory)))
        for x,xx in enumerate(trajectory):
            if len(xx)>0:
                for y,yy in enumerate(xx):
                    matrix[y,x] = yy
            else:continue
        target_index = list(zip([len(i) - 1 for i in trajectory], range(len(trajectory))))
        if sum([len(item) for item in trajectory])== 0:
            matrix = [[0]]*len(trajectory)
        return matrix,target_index


