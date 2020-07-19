import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

"""一个形如 y=wx+b的模型"""
x=tf.random.normal([4,784])
print(x)
net=layers.Dense(10)
"""a=net.built((4,784))"""
"""
def build(input_shape)
"""
print(net(x).shape)
print(net.kernel.shape)
print(net.bias.shape)