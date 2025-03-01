import input_data
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
mnist = input_data.read_data_sets("Datasets/RawData/MNIST/raw", one_hot=False)

train_img = np.array(mnist.train.images)
train_label = np.array(mnist.train.labels)
test_img = np.array(mnist.test.images)
test_label = np.array(mnist.test.labels)

dt = DecisionTreeClassifier(criterion='entropy')
dt.fit(train_img,train_label)
pred = dt.predict(test_img)

#print(pred)
#print(test_label)

CM = np.zeros((10,10))
for i in range(pred.shape[0]):
    CM[test_label[i]][pred[i]]+=1

#行求和，所有标签值为x的计数
row = np.sum(CM,axis=1)
#列求和，所有预测值为x的计数
col = np.sum(CM,axis=0)

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

str = "DecisionTree\n"
str += "ClassName\t\tPrecision\t\tRecall\t\t\tF1-score\n"
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

print(str)



