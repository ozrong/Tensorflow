"""
使用zeros，ones这种初始化的值都是一样的
"""
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
e=tf.ones_like(a)
"""相当于tf.ones(a.shape) 返回一个与a形状相同的tensor 然后值为1"""
print("e:",e)

"""--------------------------------------------------------------------"""
f=tf.fill([2,2],4)
"""shape=(2,2),填充为4"""
print("f:",f)
"""
f: tf.Tensor(
[[4 4]
[4 4]], shape=(2, 2), dtype=int32)
"""

