# -*- coding:utf-8 -*-
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn, rnn_cell

learning_rate = 0.001
test_str = '*i love you#'
length = len(test_str)
set_size = len(set(test_str))
list_str = list(set(test_str))
list_str.sort()
print (list_str)
print (length, set_size)
n_input = set_size
n_output = set_size
n_hidden = 20
n_steps = length-1

x = tf.placeholder(dtype=tf.float32, shape=[None, n_steps, n_input])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_output])
# 定义RNN结构
w = tf.Variable(tf.truncated_normal(shape=[n_hidden, n_output], dtype=tf.float32))
bias = tf.Variable(tf.truncated_normal(shape=[n_output], dtype=tf.float32))

def Rnn(x, w, bias):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], w) + bias
pred = Rnn(x, w, bias)
pred = tf.nn.softmax(pred)
cost = -tf.reduce_mean(y*tf.log(pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 产生数据
inputss = []
labels = []
j = 0
for k in range(100):
    num = k
    inputs = []
    for i in range(length-1):
        r = test_str[num%length]
        print(r, end='')
        # next_r = test_str[(i+1)%length]
        index = list_str.index(r)
        row = [0 for i in range(set_size)]
        row[index] = 1
        inputs.append(row)
        j = num
        num += 1
    inputss.append(inputs)
    next_r = test_str[(j+1)%length]
    print(next_r)
    row_label = [0 for i in range(set_size)]
    index_label = list_str.index(next_r)
    row_label[index_label] = 1
    labels.append(row_label)


final_inputs = inputss
final_inputs = np.array(final_inputs)
final_labels = labels
final_labels = np.array(final_labels)
print(final_inputs.shape, final_labels.shape)

# training
epochs = 1000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        sess.run(optimizer, feed_dict={x: final_inputs, y: final_labels})
        loss = sess.run(cost, feed_dict={x: final_inputs, y: final_labels})
        print(loss)
    print('Optimizer finished!!')

    # 测试
    test_inputs = []
    inputs = []
    test_str_new = ' love you#*'
    for i in range(length - 1):
        r = test_str_new[i % length]
        print(r, end='')
        # next_r = test_str[(i+1)%length]
        index = list_str.index(r)
        row = [0 for i in range(set_size)]
        row[index] = 1
        inputs.append(row)
    test_inputs.append(inputs)
    test_inputs = np.array(test_inputs)
    predict = sess.run(pred, feed_dict={x:test_inputs})
    predict = predict[0]
    predict = predict.tolist()
    max_index = predict.index(max(predict))
    print('\n' + list_str[max_index])
    print('Test finished!!!')
