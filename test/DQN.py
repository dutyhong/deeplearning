#-*- coding:utf-8 -*-
import random
from collections import deque

import gym
import tensorflow as tf
import numpy as np

gamma = 0.9
init_epsilon = 0.5
final_epsilon = 0.1
replay_size = 10000
batch_size = 32
ENV_NAME = "CartPole-v0"
EPISODE = 10000 #玩游戏玩一万次
TEST_EPISODE = 300
STEP = 300 #每次玩游戏最多多少步，没打到最后输赢就退出
learning_rate = 0.001
TEST = 10
##对于中间变量或者只在某一个函数中使用在别的函数中不用是用的变量都不用定义为类变量
class DQN(object):
    def __init__(self, env):
        self.replay_buffer = deque()
        self.epsilon = init_epsilon
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.create_q_network()
        self.create_training_method()
        #init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    #create q network
    def create_q_network(self):
        W1 = tf.Variable(tf.truncated_normal(shape=[self.state_dim, 20], mean=0, stddev=0.1))
        b1 = tf.Variable(tf.constant(value=0.01, dtype=tf.float32, shape=[20]))
        W2 = tf.Variable(tf.truncated_normal(shape=[20, self.action_dim], mean=0, stddev=0.1))
        b2 = tf.Variable(tf.constant(value=0.01, dtype=tf.float32, shape=[self.action_dim]))
        #input layer
        self.state_input = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim])
        layer1_out = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        #output_layer
        self.q_value = tf.matmul(layer1_out, W2) + b2

    #create training method
    def create_training_method(self):
        self.action_input = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim]) # one hot
        self.y_input = tf.placeholder(dtype=tf.float32, shape=[None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input-q_action))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    #定义perceive函数，存储每次选择一个action之后环境反馈回来的状态， 动作，当存储的数据大于batch size时训练一次dnn
    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer)>replay_size:
            self.replay_buffer.popleft()
        if len(self.replay_buffer)>batch_size:
            self.training_q_network()

    ##define the training network
    '''
        输入state_batch 是为了得到网络输出的q_vaule值 ，但是为了每次训练之训练某一个action的值，采取one hot
        所以还要输入action_batch 为了得到每个action的ont hot 编码 实际的label值事根据q_learning迭代公式计算而来
        通过DNN拟合的值就是这个值，所以训练的cost为y_batch-q_value
    '''
    def training_q_network(self):
        # from the minibatch get the training data
        mini_batch = random.sample(self.replay_buffer, batch_size)
        state_batch = [data[0] for data in mini_batch]
        action_batch = [data[1] for data in mini_batch]
        reward_batch = [data[2] for data in mini_batch]
        next_state_batch = [data[3] for data in mini_batch]
        # calculate y
        y_batch = []
        q_value_batch = self.q_value.eval(feed_dict={self.state_input:next_state_batch}) #根据公式计算的下一个状态的q值 然后去最大的值
        for i in range(0, batch_size):
            done = mini_batch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + gamma*np.max(q_value_batch[i]))
        self.optimizer.run(feed_dict={self.y_input:y_batch,self.state_input:state_batch, self.action_input:action_batch})

    #define the choose action fun
    # 采取explorer和exploetation均衡策略，更容易找到最优解
    def greedy_choose_action(self, state):
        q_value = self.q_value.eval(feed_dict={self.state_input:[state]})[0]
        # self.epsilon = self.epsilon - (init_epsilon - final_epsilon) / 10000
        if random.random()>self.epsilon:
            return np.argmax(q_value)
        else:
            return np.random.randint(0, self.action_dim-1)
    #只选择最大那个值，网络训练好之后选择 测试
    def max_choose_action(self, state):
        return np.argmax(self.q_value.eval(feed_dict={self.state_input:[state]})[0])

def main():
    # initl gym and dqn
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    for i in range(EPISODE):
        #initial state
        state = env.reset()
        for j in range(STEP):
            action = agent.greedy_choose_action(state)
            next_state, reward, done, _= env.step(action)
            agent.perceive(state,action, reward, next_state, done)
            if done:
                break
            state = next_state
            # Test every 100 episodes
        if EPISODE % 100 == 0:
            total_reward = 0
            for i in xrange(TEST):
                state = env.reset()
                for j in xrange(STEP):
                    env.render()
                    action = agent.max_choose_action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print 'episode: ', EPISODE, 'Evaluation Average Reward:', ave_reward
            if ave_reward >= 200:
                break
    # test
    for i in range(TEST_EPISODE):
        state = env.reset()
        total_reward = 0
        for j in range(STEP):
            env.render()
            action = agent.max_choose_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward = total_reward + reward
            if done:
                break
if __name__ == '__main__':
    main()