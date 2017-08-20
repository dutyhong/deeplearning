import tensorflow as tf
# import numpy as np
# # #import Tensor
# # m1 = tf.constant([[3.0,3.0]])
# # m2 = tf.constant([[2.],[2.]])
# # product = tf.matmul(m1,m2)
# # sess = tf.Session()
# # result = sess.run(product)
# # print result
# # sess = tf.InteractiveSession()
# # x = tf.Variable([1.0,2.0])
# # a = tf.constant([3.0,3.0])
# # x.initializer.run()
# # sub = tf.sub(x,a)
# # print sub.eval()
# t = [[1,2,3],
#      [4,5,6],
#      [7,8,9]]
# print tf.Tensor.shape
# #print t.rank()
# # sess.close()
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.mul(a, b)
tf.global_variables_initializer()
xx = tf.Variable(tf.truncated_normal([3,5], stddev=0.1))
# sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    yy = sess.run(xx)
    print(yy)
    # a1 = 4
    # b1 = 5
    # print "%f+%f = %f"%(4,5,sess.run(y,feed_dict={a:6,b:9}))
# sess.close()



