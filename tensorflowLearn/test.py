import tensorflow as tf
import numpy as np
a=tf.zeros([])
print("a:",a)
#tf.Tensor(0.0, shape=(), dtype=float32)
#这就是一个标量
b=tf.zeros([1])
print("b:",b)
#tf.Tensor([0.], shape=(1,), dtype=float32)
#理解为长度为1的list(或者说是数组)值为0
c=tf.zeros([2,2])
print("c:",c)
#tf.Tensor(
# [[0. 0.]
# [0. 0.]], shape=(2, 2), dtype=float32)
d=tf.zeros([2,2,3])
print("d:",d)
"""
[[[0. 0. 0.]
  [0. 0. 0.]]

 [[0. 0. 0.]
  [0. 0. 0.]]], shape=(2, 2, 3), dtype=float32)
注意 shape=(a,b,c)
a表示的是维度，b是行，c是列
    shape=(a,b)
a表示的行，b为列
    shape=(a,)
    a为维度（或者说是行）
"""