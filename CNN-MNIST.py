import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms as tf
import torch.nn as nn
import torch.nn.functional as func

import matplotlib.pyplot as plt
import numpy as np
import sys

#数据格式转换
data_transforms=tf.Compose([
    tf.ToTensor(),
    tf.Normalize([0.5],[0.5])
])

batch_size=100
trainset=torchvision.datasets.MNIST(
    root='.\Datasets\RawData',
    train=True,
    download=True,
    transform=data_transforms
    )
testset=torchvision.datasets.MNIST(
    root='.\Datasets\RawData',
    train=False,
    download=True,
    transform=data_transforms
    )

train_loader =torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True)
test_loader =torch.utils.data.DataLoader(testset,batch_size=batch_size,shuffle=True)

def get_num_correct(preds, labels):  # get the number of correct times
    return preds.argmax(dim=1).eq(labels).sum().item()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cov1 = nn.Conv2d(
            in_channels=1, #灰度图片 通道固定为1
            out_channels=6,
            kernel_size=5
        )
        self.cov2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=5
        )
        #原图像28*28 卷积核5*5 池化2*2，两层卷积池化后特征图为4*4，输出通道数12
        #故全连接层第一个输入为12*4*4=192
        self.fc1 = nn.Linear(192,120)
        self.fc2 = nn.Linear(120,60)

        self.out = nn.Linear(60,10)

    
    def forward(self, t):
        
        #卷积层1
        t = self.cov1(t)
        #relu激活函数
        t = func.relu(t)
        #池化层 核大小2*2
        t = func.max_pool2d(t,kernel_size=2,stride=2) 

        #卷积层2
        t = self.cov2(t)
        #relu激活函数
        t = func.relu(t)
        #池化层 核大小2*2
        t = func.max_pool2d(t,kernel_size=2,stride=2)

        t = t.reshape(batch_size,192)

        #全连接层1
        t = self.fc1(t)
        t = func.relu(t)
        #全连接层2
        t = self.fc2(t)
        t = func.relu(t)
        #输出层
        t = self.out(t)

        return t


def train(epoch,lr,opt):
    network = CNN() #初始化网络
    loss_f = nn.CrossEntropyLoss() #损失函数采用交叉熵
    learning_rate = lr
    epochs = epoch

    if(opt=='Adam'):
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif(opt=='SGD'):
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    elif(opt=='NAG'):
        optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    else:
        print("参数有误")
        exit()

    train_loss = np.zeros(epochs)#初始化
    val_loss = np.zeros(epochs)

    print(f'CNN MaxEpochs = {epochs}, Learning_rate = {learning_rate}, Optimizer = {opt} ')

    for i in range(epochs):

        total_trainloss = 0
        total_train_correct = 0
        total_test_correct = 0
        total_valloss = 0
        train_cnt = 0
        val_cnt = 0

        for batch in train_loader:  #训练过程
            train_cnt += 1
            images,labels = batch
            preds = network (images)
            loss = loss_f(preds,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_trainloss +=loss.item()
            total_train_correct += get_num_correct(preds, labels)

        for batch in test_loader:  #验证集上精度，验证集共用测试集
            val_cnt+=1
            images, labels = batch
            preds = network(images)
            valloss = loss_f(preds, labels)

            total_valloss += valloss
            total_test_correct += get_num_correct(preds, labels)

        total_trainloss = total_trainloss/train_cnt
        total_valloss = total_valloss/val_cnt

        train_loss[i] = total_trainloss
        val_loss[i] = total_valloss

        print("epoch:", i, 
            f"training Accuracy:", "%.3f" %(total_train_correct/len(trainset)*100), "%", 
            "train_loss:", "%.3f" %total_trainloss,"val_loss:","%.3f"%total_valloss)

    torch.save(network,f'./model/cnn_{epochs}-{learning_rate}-{opt}.pth')

    #绘图
    x_epochs = np.arange(1,epochs+1,1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xticks(x_epochs)
    plt.title(f'CNN Lr={learning_rate}, Opt={opt}')
    plt.plot(x_epochs,train_loss,color='blue',label='train_loss')
    plt.plot(x_epochs,val_loss,color='orange',label='val_loss')
    plt.legend()
    plt.show()

    return network

def Evaluate(network):    
    #求解测试集上混淆矩阵
    CM = np.zeros((10,10))
    for batch in test_loader:
        image,label = batch
        pred=network(image).argmax(dim=1)
        
        for i in range(label.shape[0]):
            CM[label[i]][pred[i]]+=1

    #行求和，所有标签值为x的计数
    row = np.sum(CM,axis=1)
    #列求和，所有预测值为x的计数
    col = np.sum(CM,axis=0)

    #print(CM)

    classNames=np.arange(0,10,1)
    PR=np.zeros(10)
    RC=np.zeros(10)
    F1=np.zeros(10)
    tr=0

    for i in range(10):
        PR[i] = CM[i][i] / row[i]
        RC[i] = CM[i][i] / col[i]
        F1[i] = 2*PR[i]*RC[i]/(PR[i]+RC[i])
        tr = tr + CM[i][i]

    Acc = tr / np.sum(CM)
    Macro_PR = PR.mean()
    Macro_RC = RC.mean()
    Macro_F1 = F1.mean()

    Micro_PR = tr / np.sum(CM)
    Micro_RC = tr / np.sum(CM)
    Micro_F1 = 2*Micro_PR*Micro_PR/(Micro_PR+Micro_RC)


    str = "Class Name\t\tPrecision\t\tRecall\t\t\tF1-score\n"
    for i in range(10):
        str += f"{classNames[i]}   \t\t\t{format(PR[i],'.3f')}   \t\t{format(RC[i],'.3f')}" \
            f"   \t\t{format(F1[i],'.3f')}\n"
    str += f"Accuracy = {format(Acc,'.3f')}\n"
    str += f"Macro_PR = {format(Macro_PR,'.3f')}\n"
    str += f"Macro_RC = {format(Macro_RC,'.3f')}\n"
    str += f"Macro_F1 = {format(Macro_F1,'.3f')}\n"
    str += f"Micro_PR = {format(Micro_PR,'.3f')}\n"
    str += f"Micro_RC = {format(Micro_RC,'.3f')}\n"
    str += f"Micro_F1 = {format(Micro_F1,'.3f')}\n"

    return str


network = train(epoch=10,lr=0.005,opt='Adam') #opt='Adam'/'SGD'/'NAG'
report = Evaluate(network) #模型评估
print(report) #打印结果







