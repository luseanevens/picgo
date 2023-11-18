---
title: PR CIFAR10
date: 2020-03-11 9:22:23
tags:
- [tensorflow]
categories:
- [python]
---
# PR CIFAR10

记录一下神经网络分类从单CNN的60%调参到90%。

<!--more-->

## 环境配置

```bash
conda create --name ts18 python==2.7
conda info --env
conda activate ~
pip install -U pip
# 更新pip到20.3.1
pip install jupyter
jupyter notebook
# 报错 字符问题 强制英文
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe5 in position 4: ordinal not in range(128)
LANG = zn jupyter notebook
###################################################
pip install tensorflow==1.8
python 
print tensorflow.__version__
###################################################


cd Desktop& mkdir prpy27
cd prpy27
jupter notebook
control^c
conda deactivate
```

注：

- 数据集有5个batch和一个test集，共六万张图片
- 32*32的图像 十分类

## 下载数据集

- 使用[cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) for python，以Pickle文件存储，使用cPickle包

## 基础知识
### 前置
- mini - batch

  ​	神经元的作用就是不断的修正$w$使得损失函数最小，但是如果对每一个样本都计算一次损失函数，计算量过于庞大，mini-batch就是随机的从样本中选择随机样本样本进行训练。$\bold x$就是这样一个mini-batch的样本集。此时的mini-batch数据需要考虑：shuffle、batch_next的函数定义

### 神经网络

- 最小的神经元：逻辑斯蒂回归模型，使用sigmoid函数转化概率
- 多分类神经元：十分类需要十个神经元，要有十个输出，使用softmax来转化概率

## Tensorflow

- 实现单个神经元
  - 单个神经元结构：$h(\bold W^T \bold x+b)$ 
    - $h(x)$:`activation function`
    - $\bold W^T=[w_1,w_2,...,w_n] \qquad (1*n)$   
    - $\bold x = [ x_1, x_2,..., x_n]\qquad (n*1)$

```python
x = tf.placeholder(tf.float,[None,3072])
# 占位符 数据集导入x成为一个（样本个数*3072）的矩阵，矩阵中的每一行代表一张图的3072个维度
# None 表示未知 由输入的样本做作为输入量 mini-batch的要求
y= tf.placeholder(tf.int64,[None])
# 标签 （样本数*1）
tf.get_variable('w',[tf.get_shape(x)[-1],1],initializer=tf.random_normal_initializer(0,1))
'''
tf.get_varable('w',[tf.get_shape(x)[-1],10],
initializer=tf.random_normal_initializer(0,1)
)

'''
# 创建w get_variable ：若已存在定义的w 则使用已有的w 若不存在，则创建一个新的w
# tf.getshape(w)[-1] = 3072 为了增强泛化能力，只需修改x的维度，w的维度也会相应修正 
# tf.get_variable('name',[dim],initializer= )
# tf.random_normal_initializer 为正态分布
tf.get_variable('b',[1]，initializer=tf.constant_initializer(0.0))
# tf.constant_initializer(0.0) 常数 0.0

y_=tf.matmul(x,w)+b
# mat multipli 矩阵乘法 神经元计算最终要得到一个常数以带入激活函数，以此准则注意定义参数时的维度

p_y_1 = tf.nn.sigmoid(y_)
# 将数值转化为概率值
# p_y_1 [None *1] float32
# y [None] int32
y_reshaped = tf.reshape(y,(-1,1))
# y_reshaped [None,1] int32
y_reshaped_float=tf.cast(y_reshaped,float32)
# y_reshaped_float [None,1] float32 = p_y_1

# 平方差损失
loss = tf.reduce_mean(tf.square(y_reshaped_float32-p_y_1)
# 准确率
# p_y_1 \belong (0,1)
predict= p_y_1>0.5
# sigmoid 关于（0，0.5）对称
correct_prediction= tf.equal(tf.cast(prtdict,tf.int64),y_reshaped)
# correct_prediction= [1,0,1,1,0,0,0],取平均后得到准确率
accuary = tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

  
                      
```

定义梯度下降的方法：

```
with tf.name_scope('train_op')
	train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)
	# 使用库 使用动量梯度下降 
```





图像未做缩放时的准确率：

![image-20200609202218462](https://tva1.sinaimg.cn/large/007S8ZIlly1gfmb1tiknfj30se0r87ah.jpg)

在训练图像之前将图像归一化：`self.data= self.data/127-1`
$$
[0,255]/127=[0,2]\\\\
[0,2]-1=[-1,1]
$$
之后的精度：

![image-20200609203010733](https://tva1.sinaimg.cn/large/007S8ZIlly1gfmb9ytvu7j30pg0lejwp.jpg)

可以达到0.8以上，sigmoid的函数梯度变换集中在$[-6,6]$之间，当值很大时会导致梯度消失，归一化后，分类的效果会变好很多。

- 构建多神经元（无隐藏层）

```python
tf.getvarable('w',[tf,get_shape(x),10],
				initiallizer=tf.random_normal_initializer(0,1))
	
```



![image-20200609220706383](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfme2vo8vgj30pg0o6gqx.jpg)

- 加入卷积层与池化层

  - tensorflow中存在简化的定义w的方法，在前面使用`tf.get_varable('w',[tf.get_shape(x),10],initializer=)`可以一步写为：`tf.layers.dense(x,10)`定义全连接层，输入x会自动get到x的维度，10输出表示10分类。

    - `tf.layers`	还可以定义池化和卷积操作，为了优化神经网络，加入三层池化、卷积后进入全连接，

    - ```python
      conv1=tf.layers.conv2d(x,	
                             32，#输出通道 （提取几个特征）
                     				(3,3), #size
                    				 padding='same',	#保持图像大小
                    				 activation=tf.nn.rule,
                            name='conv1')
      pool1 = tf.layers.pooling2d(conv1,
                                 (2,2), #size
                                 (2,2),#stride
                                 name='pool1')
      ```

    - 这样就定义了一个卷积层和一个池化层,注意池化层和卷积层输入的是图像，对于cifar10数据集而言是需要reshape和transpose的。

- 最终准确率

![image-20200612231726407](https://tva1.sinaimg.cn/large/007S8ZIlgy1gfpwz1y68pj30s80qy44v.jpg)

## 问题总结

- `tf.get_varable('name',[n,m],initializer=)`是一个使用异常定义的函数函数，这个 函数很有意思的一点是tensorflow是一个以计算图为切入点的网络编程语言，他面向对象编程的思路是很明显的，使用with定义函数的好处就是如果已有过存储空间中的变量导致的问题，就会抛出异常，很有效的避免了错误的混淆，但是之后使用`tf.layers.dense`	等借口其表现形式就不明了了。

- jupyternotebook 存在内存污染的情况，可能是我内存不大够用
- 还是没有搞清楚卷积和池化层级对精度的影响



