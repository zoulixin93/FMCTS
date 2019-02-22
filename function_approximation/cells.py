#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: cells
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/2/9,3:03 PM
#==================================
#!/usr/bin/python
# encoding: utf-8


#==================================
#@file name: rnn_cells
#@author: Lixin Zou
#@contact: zoulixin15@gmail.com
#@time:2019/1/2,4:07 PM
#==================================

import numpy as np
import ipdb
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import *
from tensorflow.contrib.model_pruning.python.layers import core_layers
_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_clipped_value = 100000000.0
class nlstm(tf.nn.rnn_cell.BasicLSTMCell):
    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units


    def build(self, inputs_shape):
        print("ntlstm cell")
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value-1
        h_depth = self._num_units
        # input gate
        self.W_xi = self.add_variable(self.name+"_w_xi",shape=(input_depth, h_depth))
        self.W_hi = self.add_variable(self.name+"_w_hi",shape=(h_depth, h_depth))
        self.bias_i = self.add_variable(self.name+"_bias_i",shape=(h_depth,))
        self.w_ci = self.add_variable(self.name+"_w_ci",shape=(h_depth,))
        # forget gate
        self.W_xf = self.add_variable(self.name+"_w_xf",shape=(input_depth, h_depth))
        self.W_hf = self.add_variable(self.name+"_w_hf",shape=(h_depth, h_depth))
        self.w_cf = self.add_variable(self.name+"_w_cf",shape=(h_depth,))
        self.bias_f = self.add_variable(self.name+"_bias_f",shape=(h_depth,))
        # cell
        self.W_xc = self.add_variable(self.name+"_w_xc",shape=(input_depth, h_depth))
        self.W_hc = self.add_variable(self.name+"_w_hc",shape=(h_depth, h_depth))
        self.bias_c = self.add_variable(self.name+"_bias_c",shape=(h_depth,))
        # output gate
        self.W_xo = self.add_variable(self.name+"_w_xo",shape=(input_depth, h_depth))
        self.W_ho = self.add_variable(self.name+"_w_ho",shape=(h_depth, h_depth))
        self.w_co = self.add_variable(self.name+"_w_co",shape=(h_depth,))
        self.bias_o= self.add_variable(self.name+"_bias_o",shape=(h_depth,))

        self.built = True

    def call(self, inputs, state):
        # time = tf.slice(inputs, [0, tf.shape(inputs)[1] - 1], [-1, 1], name=self.name+"_rnn_time")
        inputs = tf.slice(inputs, [0, 0], [-1, tf.shape(inputs)[1] - 1], name=self.name+"_rnn_input")
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        # cell clipping to avoid explostion
        c = clip_ops.clip_by_value(c, -1.0*_clipped_value, 1.0*_clipped_value)

        input_gate = sigmoid(
            math_ops.add(
                math_ops.add(
                    math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xi),
                        math_ops.mat_mul(h,self.W_hi)),
                    math_ops.multiply(c,self.w_ci)),
                self.bias_i))

        forget_gate = sigmoid(
            math_ops.add(
                math_ops.add(
                    math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xf),
                        math_ops.mat_mul(h,self.W_hf)),
                    math_ops.multiply(c,self.w_cf)),
                self.bias_f))
        new_c = math_ops.add(
            math_ops.multiply(forget_gate,c),
            math_ops.multiply(
                input_gate,math_ops.tanh(
                    math_ops.add(math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xc),
                        math_ops.mat_mul(h,self.W_hc)),
                        self.bias_c))))
        output_gate = sigmoid(
            math_ops.add(
                math_ops.add(
                    math_ops.add(
                        math_ops.mat_mul(inputs,self.W_xo),
                        math_ops.mat_mul(h,self.W_ho)),
                    math_ops.multiply(new_c,self.w_co)),
                self.bias_o))
        new_h = math_ops.multiply(output_gate,math_ops.tanh(new_c))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state


class memory_cell(nlstm):
    def build(self, inputs_shape):
        print("memory cell")
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        self.built = True

    def call(self, inputs, state):
        return inputs, state

