import numpy as np
import torch
from torchvision.datasets import mnist

from torch import nn
from torch.autograd import Variable
import torchvision

#下载mnist数据集，因为我已经下载了，所以直接download=True
train_set=mnist.MNIST(
    './data',
    train=True,
    download=False)

test_set=mnist.MNIST(
    './data',
    train=False,
    download=False)

a_data,a_label=train_set[0]

#这时候的a_data是PIL库中的格式，我们需要把它转成ndarray的格式
a_data=np.array(a_data)
#print(a_data)
#输出如下,是每个元素 0~256 的28*28的图

#[[  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   3.  18.  18.  18. 126. 136. 175.  26. 166. 255. 247. 127.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.  30.  36.  94. 154. 170. 253. 253. 253. 253. 253. 225. 172. 253. 242. 195.  64.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.  49. 238. 253. 253. 253. 253. 253. 253. 253. 253. 251.  93.  82.  82.  56.  39.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.  18. 219. 253. 253. 253. 253. 253. 198. 182. 247. 241.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.  80. 156. 107. 253. 253. 205.  11.   0.  43. 154.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.  14.   1. 154. 253.  90.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 139. 253. 190.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  11. 190. 253.  70.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  35. 241. 225. 160. 108.   1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  81. 240. 253. 253. 119.  25.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  45. 186. 253. 253. 150.  27.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  16.  93. 252. 253. 187.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0. 249. 253. 249.  64.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  46. 130. 183. 253. 253. 207.   2.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  39. 148. 229. 253. 253. 253. 250. 182.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.  24. 114. 221. 253. 253. 253. 253. 201.  78.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.  23.  66. 213. 253. 253. 253. 253. 198.  81.   2.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.  18. 171. 219. 253. 253. 253. 253. 195.  80.   9.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.  55. 172. 226. 253. 253. 253. 253. 244. 133.  11.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0. 136. 253. 253. 253. 212. 135. 132.  16.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
# [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]]

#对于神经网络来说，我们第一层的输入就是28*28=784，所以我们需要将我们得到的数据进行交换，把它拉平成一维向量

def data_tf(x):
    x=np.array(x)
    x=(x-0.5)/0.5 # 标准化
    x=x.reshape((-1))#-1会把它所有的维度拉平成一维（所有其他的维度乘起来）
    return x


#重新导入数据集
train_set2=mnist.MNIST(
    './data',
    train=True,
    transform=torchvision.transforms.Compose(
         [torchvision.transforms.ToTensor(),
          data_tf]),
    #Compose将多个函数合并起来，对数据集进行改造
    #totensor将数据缩放到0~1
    #data_tf将数据变成标准正态分布
    download=False)

test_set2=mnist.MNIST(
    './data',
    train=False,
    transform=torchvision.transforms.Compose(
         [torchvision.transforms.ToTensor(),
          data_tf]),
    download=False)

#这样出来的数据就是每个784+label

#接下来创建mini-batch的迭代器
from torch.utils.data import DataLoader
train_data=DataLoader(train_set2,batch_size=64,shuffle=True)
test_data=DataLoader(test_set2,batch_size=128,shuffle=False)
#batch_size 批处理的数量
#shuffle 是否打乱

#a,a_label=next(iter(train_data))
#print(a.shape)
#64*784
#print(a_label.shape)
#784

#定义一个四层的神经网络
net=nn.Sequential(
    nn.Linear(28*28,400),
    nn.ReLU(),
    nn.Linear(400,200),
    nn.ReLU(),
    nn.Linear(200,100),
    nn.ReLU(),
    nn.Linear(100,10)
    )

print(net)

#Sequential(
#  (0): Linear(in_features=784, out_features=400, bias=True)
#  (1): ReLU()
#  (2): Linear(in_features=400, out_features=200, bias=True)
#  (3): ReLU()
#  (4): Linear(in_features=200, out_features=100, bias=True)
#  (5): ReLU()
#  (6): Linear(in_features=100, out_features=10, bias=True)
#)

#定义损失函数和优化函数
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.8)

#开始训练
losses=[]
accuracy=[]
eval_loss=[]
eval_accuracy=[]
 
for e in range(20):
    train_loss=0
    train_accuracy=0
    net.train()#现在的神经网络是训练模式（可加可不加，建议加上）

    for input,label in train_data:
        input=Variable(input)
        label=Variable(label)

        output=net(input)
        loss=loss_func(output,label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss=train_loss+loss.item()#记录误差

        #记录准确率
        _,prediction=output.max(1)
        #output是每一个元素分别属于十个数字的概率，max()后返回两个数，第一个是最大的那个数值，第二个是最大的数值所在的位置（行/列）
        #0表示每一行的最大的一个，1表示每一列的最大的一个
        #出来的也是tensor
        num_correct=(pred==label).sum().item()#所有对的数量
        acc=num_correct/input.shape[0]
        train_accuracy+=acc
   losses.append(train_loss/len(train_data))
   accuracy.append(ttrain_accuracy/len(train_data))
