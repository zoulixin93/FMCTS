#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: mcts
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/9,8:40 PM
#==================================

import numpy as np
import ipdb
import copy as cp
from utils.statistical_utils import softmax
from function_approximation import rnn_model
import os
from utils.time_analyse import timeit
from utils.zlog import log

class TreeNode(object):
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 1.0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value,discount_factor):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        self.update(leaf_value*discount_factor)
        if self._parent:
            self._parent.update_recursive(leaf_value,discount_factor)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits)) + np.random.random()/10000
        # print(self._Q,self._u)
        return self._Q + self._u


    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        if self._children == {}:
            return True
        else:
            return False

    def is_root(self):
        return self._parent is None

class MCTS(object):
    def __init__(self,policy_value_agent, c_puct, n_playout, discount_factor):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_agent
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._discount_factor = discount_factor

    def _playout(self, env):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        done = False
        reward = 0
        while True:
            if node.is_leaf():
                break
            # select next move.
            action,node = node.select(self._c_puct)
            sensible_actions,node_type,reward,done = env.step(action)
        _action_probs, next_state_value = self._policy["model"].get_actions_probability(self._policy["sess"],env)
        if done:
            node.update_recursive(reward, self._discount_factor)
        else:
            temp_env = cp.deepcopy(env)
            t = False
            _r = 0
            while not t:
                _a_p, n_v = self._policy["model"].get_actions_probability_model(self._policy["sess"], temp_env)
                _a, _p = zip(*_a_p)
                _a = _a[np.argmax(_p)]
                _c, _n_t, _r, t = temp_env.step(_a)
            node.update_recursive(_r,self._discount_factor)
        if not done:
            node.expand(_action_probs)

    def get_move_probs(self, env, temperature=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # current_step = env.config.top_k *2+5- (len(env.trajectory)+len(env.cat_trajectory))
        # _n_playout = int(min([len(env.candidate)*current_step,self._n_playout]))
        # log.info("start playout",_n_playout)
        for i in range(self._n_playout):
            env_copy = cp.deepcopy(env)
            self._playout(env_copy)
        # log.info("finish playout")
        # calculate the move probabilities based on visit counts at the root node
        # temperature = max([temperature, 1e-3])
        act_q = [(act, node._Q) for act,node in self._root._children.items()]
        acts,q = zip(*act_q)
        act_probs = softmax(np.asarray(q))
        return acts, act_probs

    def get_action(self, env, temperature=1e-3, training=False):
        acts, probs = self.get_move_probs(env, temperature)
        if training:    # if training, add noise for exploration
            # randomly choose the move in terms of action probabilities
            # and revise the action probabilities
            # move_probs[:] = 0.0
            # move_probs[move] = 1.0
            # reset the root node
            # action = np.random.choice(acts, p=probs)
            action = acts[np.argmax(probs)]
            self.update_with_move(action)
        else:
            action = acts[np.argmax(probs)]
            self.update_with_move(action)
        return action,acts,probs


    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"



