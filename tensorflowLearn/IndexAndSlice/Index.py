import tensorflow as tf
"""通用索引
索引方式单一
写起来麻烦[][]太多
"""
a=tf.ones([1,5,5,3])
"""
这是在tensor里面
[b,h,w,c]理解为b(batch)个数，h高(行)，w宽（列），c(channel通道)

在pytorth里面有所不同[b,c,h,w]
可以使用转置函数将tf的tensor转为pytorth的的顺序
tf.transpose(a,[0,3,1,2])
"""
print(a)
"""
tf.Tensor(
[[[[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]

  [[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]

  [[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]

  [[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]

  [[1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]
   [1. 1. 1.]]]], shape=(1, 5, 5, 3), dtype=float32)

Process finished with exit code 0
"""
print(a[0][0])
"""
tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]], shape=(5, 3), dtype=float32)
"""
print(a[0][0][0])
"""
tf.Tensor([1. 1. 1.], shape=(3,), dtype=float32)
"""
"""==========================================="""
"""numpy-style index   """
b=tf.random.normal([4,28,28,3])
print(b[1].shape)
print(b[1,2].shape)
"""相当于b[1][2]"""
print(b[1,2,4].shape)
"""相当于b[1][2][4]"""