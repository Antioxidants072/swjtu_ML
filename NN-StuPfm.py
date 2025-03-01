import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import torchvision
from torchvision import transforms as tf
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('Datasets\RawData\Student_performance_data _.csv')
dataset.drop('StudentID', axis=1, inplace=True)

X=dataset.drop(['GPA','GradeClass'], axis=1)
y=dataset['GradeClass']

scaler = StandardScaler()
X = scaler.fit_transform(X)

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

train_x = np.array(train_x)
test_x = np.array(test_x)
train_y = np.array(train_y)
test_y = np.array(test_y)

train_x = torch.FloatTensor(train_x)
test_x = torch.FloatTensor(test_x)
train_y = torch.LongTensor(train_y)
test_y = torch.LongTensor(test_y)

#train_y=func.one_hot(train_y,num_classes=3)
#test_y=func.one_hot(test_y,num_classes=3)
#train_y = torch.FloatTensor(train_y)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 80)  # 全连接层
        self.fc2 = nn.Linear(80, 120)
        self.fc3 = nn.Linear(120, 40)
        self.out = nn.Linear(40,5)

    def forward(self, x):
        x = func.relu(self.fc1(x))  # 使用ReLU激活函数
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.out(x)
        return x



epoch = 300
learningrate = 0.01
model=Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
loss_recond = np.zeros(epoch)

for i in range(epoch):
    optimizer.zero_grad()
    out = model(train_x)
    
    loss = criterion(out, train_y)
    print(f'epoch={i}\ttrain_loss={loss}')
    loss_recond[i] = loss
    loss.backward()
    optimizer.step()

torch.save(model,f'./model/nn_{epoch}-{learningrate}-Adam.pth')

x_epochs = np.arange(1,epoch+1)
x_tick = np.arange(0,epoch+1,30)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.xticks(x_tick)
plt.title(f'NN Lr={learningrate}')
plt.plot(x_epochs,loss_recond,color='blue',label='train_loss')

plt.legend()
plt.show()

pred = model(test_x).argmax(dim=1)
test_y = np.array(test_y)

CM = np.zeros((5,5))
for i in range(pred.shape[0]):
    CM[test_y[i]][pred[i]]+=1



#行求和，所有标签值为x的计数
row = np.sum(CM,axis=1)
#列求和，所有预测值为x的计数
col = np.sum(CM,axis=0)

classNames=np.arange(0,5,1)
PR=np.zeros(5)
RC=np.zeros(5)
F1=np.zeros(5)
tr=0

for i in range(5):
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

str = "Neural Network\n"
str += "ClassName\t\tPrecision\t\tRecall\t\t\tF1-score\n"
for i in range(5):
    str += f"{classNames[i]}   \t\t\t{format(PR[i],'.3f')}   \t\t{format(RC[i],'.3f')}" \
        f"   \t\t{format(F1[i],'.3f')}\n"
str += f"Accuracy = {format(Acc,'.3f')}\n"
str += f"Macro_PR = {format(Macro_PR,'.3f')}\n"
str += f"Macro_RC = {format(Macro_RC,'.3f')}\n"
str += f"Macro_F1 = {format(Macro_F1,'.3f')}\n"
str += f"Micro_PR = {format(Micro_PR,'.3f')}\n"
str += f"Micro_RC = {format(Micro_RC,'.3f')}\n"
str += f"Micro_F1 = {format(Micro_F1,'.3f')}\n"

print(str)