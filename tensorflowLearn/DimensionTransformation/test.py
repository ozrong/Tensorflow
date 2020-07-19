import tensorflow as tf
a=tf.random.normal([4,28,28,3])
print(a.ndim)
print(a.shape)
"""
a.ndim:a的维度
 4
(4, 28, 28, 3)
"""
"""================================================"""
"""tf.transpose 转置"""
b=tf.random.normal((4,3,2,1))
print(b.shape)
"""(4, 3, 2, 1)"""
print(tf.transpose(b).shape)
"""(1, 2, 3, 4)"""
print(tf.transpose(b,perm=[0,1,3,2]).shape)
"""
指定位置0位置放原来的b的0维度以此为退
(4, 3, 1, 2)
"""
"""=================================================="""
"""维度的增加"""
b=tf.random.normal([4,35,8])
print(b.shape)
"""shape=(4, 35, 8)"""
print(tf.expand_dims(b,axis=0).shape)
"""
expand_dims(b,axis=0)
给b增加一维，在位置0的前一个维度，（axis为正数则在该数的前面增加维度，维负数则在该位置的后面增加维度，如-1则在倒数第一的位置后面增加维度）
shape=(1, 4, 35, 8)
"""
c=tf.expand_dims(b, axis=-1)
print(c.shape)
"""shape=(4, 35, 8, 1)"""
"""维度的减少"""
print(tf.squeeze(c, axis=-1).shape)
"""
不能任意的减少维度
shape=(4, 35, 8)
"""
"""print(tf.squeeze(b, axis=-1).shape)错误"""