# 创建tensor

numpy,list  
zeros,ones  
fill  
random  
constant  
Application 

## 1 利用别的数据转换（numpy,list ）

```
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
```

---

## 2 直接生成

### 2.1 zeros,one

```
import tensorflow as tf
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
```

### 2.2 fill

生成任意形状的tensor，用任意的值来填充

```
f=tf.fill([2,2],4)
"""shape=(2,2),填充为4"""
print("f:",f)
"""
f: tf.Tensor(
[[4 4]
[4 4]], shape=(2, 2), dtype=int32)
"""
```

## 3 初始化数据

### 3.1 正态分布，（可以指定均值，方差）

```
"""
tf.random_normal 函数
random_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)
参数：
shape：一维整数张量或 Python 数组.输出张量的形状.
mean：dtype 类型的0-D张量或 Python 值.正态分布的均值.
stddev：dtype 类型的0-D张量或 Python 值.正态分布的标准差.
dtype：输出的类型.
seed：一个 Python 整数.用于为分发创建一个随机种子.查看 tf.set_random_seed 行为.
name：操作的名称(可选)
正态分布也叫高斯分布
"""
a=tf.random.normal([2,2],mean=1,stddev=1)
b=tf.random.normal([2,2])
print(b)
"""=================================================="""
"""
截断正态分布
截断正态分布是截断分布（Truncated Distribution）的一种，那么截断分布是什么？截断分布，限制变量x取值范围（scope）的一种分布。例如，限制x取值在0到50之间，即{0 < x < 50}。 因此，根据限制条件的不同，截断正态分布可以分为：
限制取值上限， 例如，负无穷< x < 50
限制取值下限， 例如， 0 < x < 正无穷
上限下限取值都限制，例如， 0 < x < 50
"""
tf.random.truncated_normal([2,2],mean=0,stddev=1)
```

### 3.2 均匀分布

```
tf.random.uniform([2,2],minval=0,maxval=1)
"minval=0,maxval=1,0-1之间的均匀采样"
```

### 3.3 打撒分布

```
tf.random.shuffle(a)
"""就是打乱a的顺序（随机地将张量沿其第一维度打乱）
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
"""
```

## 4 例子

```
out=tf.random.uniform([4,10])
"""
out: tf.Tensor(
[[0.7518294  0.9819515  0.97938323 0.36362267 0.3670665  0.6927812
  0.09581697 0.07268369 0.79723597 0.05431318]
 [0.978883   0.18295932 0.63073134 0.5969789  0.59998    0.06354165
  0.00880182 0.59430647 0.9805695  0.16789973]
 [0.518355   0.38392365 0.40262496 0.5925077  0.7260604  0.05793762
  0.8229964  0.7762661  0.62592006 0.11666644]
 [0.9876168  0.5676081  0.5214232  0.98125017 0.68812597 0.06410491
  0.6918652  0.6894411  0.5921476  0.8976363 ]], shape=(4, 10), dtype=float32)
"""
print("out:",out)
y=tf.range(4)
print("y:",y)
"""
y: tf.Tensor([0 1 2 3], shape=(4,), dtype=int32)
"""
y=tf.one_hot(y,depth=10)
print("y_one_hot",y)
"""
y_one_hot tf.Tensor(
[[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]], shape=(4, 10), dtype=float32)
"""
loss=tf.keras.losses.mse(y,out)
"""mse:平方误差  返回每一个样本的平均平方误差"""
print("loss1:",loss)
"""
loss1: tf.Tensor([0.29126996 0.23629518 0.29722816 0.37364987], shape=(4,), dtype=float32)
"""
loss=tf.reduce_mean(loss)
"""所有样本的平均"""
print("loss(reduse_mean):",loss)
"""
loss(reduse_mean): tf.Tensor(0.2996108, shape=(), dtype=float32)
"""
```

### 形如y=xw+b的模型

```
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
```

