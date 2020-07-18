

# 创建tensor

numpy,list  
zeros,ones  
fill  
random  
constant  
Application  

## 利用别的数据转换

a=tf.convert_to_tensor(np.ones([2,3]))  
"""  
output:2*3的全1矩阵  也就是[2,3]是shape      
tf.Tensor(      
[[1. 1. 1.]    
 [1. 1. 1.]], shape=(2, 3), dtype=float64)    
"""  
b=tf.convert_to_tensor(np.zeros([2,3]))  
"""  
output:全0  
tf.Tensor(  
[[0. 0. 0.]  
 [0. 0. 0.]], shape=(2, 3), dtype=float64)  
"""  
c=tf.convert_to_tensor([1,2])  
"""tf.Tensor([1 2], shape=(2,), dtype=int32)  
   注意：这里生成的是一个一维的 也就是说这儿的[1,2]就是数据  
""" 
## 直接生成
a=tf.zeros([])  
print("a:",a)  
"""tf.Tensor(0.0, shape=(), dtype=float32)  
这就是一个标量"""
b=tf.zeros([1])  
print("b:",b)  
"""tf.Tensor([0.], shape=(1,), dtype=float32)  
理解为长度为1的list(或者说是数组)值为0"""  
c=tf.zeros([2,2])  
print("c:",c)  
"""tf.Tensor(  

[[0. 0.]  
 
[0. 0.]], shape=(2, 2), dtype=float32)  

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

