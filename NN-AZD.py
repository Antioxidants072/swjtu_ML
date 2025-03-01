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

dataset = pd.read_csv('Datasets/RawData/alzheimers_disease_data.csv')
dataset.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

X=dataset.drop('Diagnosis', axis=1)
y=dataset['Diagnosis']

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
        self.fc1 = nn.Linear(32, 100)  # 全连接层
        self.fc2 = nn.Linear(100, 120)
        self.fc3 = nn.Linear(120, 40)
        self.out = nn.Linear(40,2)

    def forward(self, x):
        x = func.relu(self.fc1(x))  # 使用ReLU激活函数
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.out(x)
        return x


epoch = 200
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

pred = func.softmax(model(test_x)).argmax(dim=1)

TP,TN,FP,FN=0,0,0,0
for i in range(pred.shape[0]):
    if test_y[i]==1 and pred[i]==1:
        TP+=1
    if test_y[i]==1 and pred[i]==0:
        FN+=1
    if test_y[i]==0 and pred[i]==1:
        FP+=1
    if test_y[i]==0 and pred[i]==0:
        TN+=1

CM = f'TP={TP},TN={TN},FP={FP},FN={FN}'
print(CM)


PR = TP / (TP + FP)
RC = TP / (TP + FN)
F1 = 2 * PR * RC / (PR + RC)

str = "Neural Network\n"
str += "Precision\t\tRecall\t\tF1-score\n"

str += f"{format(PR,'.3f')}   \t\t{format(RC,'.3f')}" \
        f"   \t{format(F1,'.3f')}\n"

print(str)