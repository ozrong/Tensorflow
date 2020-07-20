import tensorflow as tf
import  numpy as np
from tensorflow.keras import datasets
import os
#os.environ['IF_CPP_MIN_LOG_LEVEL']='2'

(x,y),_=datasets.mnist.load_data()
"""转换成tensor"""
x=tf.convert_to_tensor(x,dtype=tf.float32)/255.
"""把X的大小从0-255  转换到了0-1之间"""
y=tf.convert_to_tensor(y,dtype=tf.int32)

print(tf.reduce_min(x),tf.reduce_max(x))
"""查看X中的最小值,最大值"""

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
"""批量数据"""

train_iter=iter(train_db)
sample=next(train_iter)
"""
迭代器
术语中“可迭代的”指的是支持iter的一个对象
next函数：返回迭代器的下一个元素
iter函数：返回迭代器对象本身
"""
print("batch:",sample[0].shape)

"""
[b,784]->[b,256]->[b,10]
[dim_in,dim_out],[dim_out]
"""
w1=tf.Variable(tf.random.truncated_normal([784,256]))
b1=tf.Variable(tf.zeros([256]))
w2=tf.Variable(tf.random.truncated_normal([256,128]))
b2=tf.Variable(tf.zeros([128]))
w3=tf.Variable(tf.random.truncated_normal([128,10]))
b3=tf.Variable(tf.zeros([10]))
lr=0.001

"""模型"""
"""x[b,28,28]->[b,28*28]"""
for step,(x,y) in enumerate(train_db):
    x=tf.reshape(x,[-1,28*28])
    with tf.GradientTape() as tape:
        h1=x@w1+b1
        h1=tf.nn.relu(h1)
        h2=h1@w2+b2
        h2=tf.nn.relu(h2)
        out=h2@w3+b3
        """
        loss
        因为网络的输出是一个[b,10]
        标签是一个标量 所以把它变化为[b,10]

        """
        y_onehot = tf.one_hot(y, depth=10)
        loss1 = tf.square(y_onehot - out)
        loss = tf.reduce_mean(loss1)
    grads = tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
    #指定要对那个函数求导，以及对那些参数求导

    ###SDG
    w1.assign_sub(lr*grads[0])
    b1.assign_sub(lr*grads[1])
    w2.assign_sub(lr*grads[2])
    b2.assign_sub(lr*grads[3])
    w3.assign_sub(lr*grads[4])
    b3.assign_sub(lr*grads[5])

    """    
    w1=w1-lr*grads[0]
    b1=w1-lr*grads[1]
    w2=w1-lr*grads[2]
    b2=w1-lr*grads[3]
    w3=w1-lr*grads[4]
    b3=w1-lr*grads[5]"""

    if step%100==0:
        print(step,"loss:",float(loss))


