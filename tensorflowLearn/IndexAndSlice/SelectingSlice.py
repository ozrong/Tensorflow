import tensorflow as tf
a=tf.random.normal([4,35,8])
"""
把a看作4个班级，每个班级35个人，每个人8个学科
"""
"""
def gather(params: {sparse_read},数据源
              indices: Any,
              validate_indices: Any = None,
              axis: Any = None,维度
              batch_dims: int = 0,
              name: Any = None) -> Any
"""
tf.gather(a,axis=0,indices=[2,3])
"""
在第0个维度中（也就是4的那个维度）先收集2号班级 在3号班级
也就是收集第2个班到第3个班的所有
"""

tf.gather(a,axis=0,indices=[2,1,3,0])
"""
在班级维度中，先收集2号班级，然后1,3,0班级
"""
"""
多次组合就可以采集任意样子的数据
"""
"""====================================================="""
"""多维度操作
def gather_nd_v2(params: Any,
                 indices: Any,
                 batch_dims: int = 0,
                 name: Any = None) -> Any
"""
tf.gather_nd(a,[0])
"""取0号班级的的所有数据  shape=（35，8）"""

tf.gather_nd(a,[0,1])
"""取0号班级的1号学生的所有数据  shape=（8）"""

print(tf.gather_nd(a,[0,1,2]).shape)
"""取0号班级的1号学生的2号课程所有数据    shape=()"""
"""====================================================="""
"""再次复杂"""
tf.gather_nd(a,[[0,0],[1,1]])
"""取的是0号班级的0号学生的所有数据
   以及1号班级的1号学生的所有数据
   shape=(2,8)
"""
print(tf.gather_nd(a, [[[0, 0, 0], [1, 1, 1], [2, 2, 2]]]).shape)
"""
   取的是0号班级的0号学生0号课程的所有数据
   以及1号班级的1号学生的0号课程的所有数据
   shape=(1, 3)
"""
"""=============================================="""
"""
def boolean_mask_v2(tensor: Any,
                    mask: Any,
                    axis: Any = None,
                    name: str = "boolean_mask") -> Any
"""
tf.boolean_mask(a,mask=[True,True,False,False])
"""
表示在班级维度中
True表示对应的班级取，False表示不取
这里也就是0,1号班级取，2,3号班级不取
shape=(2,35,8)
"""
print(tf.boolean_mask(a, mask=[True, True, False, False, False, False, False, False], axis=2).shape)
"""
表示在课程维度取  （不写axis表示取第0个维度）
在所有数据中只取0,1号课程的数据
shape=(4,35,2)
"""
print(tf.boolean_mask(a, mask=[[],[]]).shape)