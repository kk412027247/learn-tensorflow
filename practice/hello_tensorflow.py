import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义tf的常量
hello = tf.constant('Hello TensorFlow')

sess = tf.compat.v1.Session()

print(sess.run(hello))
