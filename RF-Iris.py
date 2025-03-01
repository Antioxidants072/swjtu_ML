import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Datasets\RawData\Iris.csv')
data.drop('Id', axis=1, inplace=True)

train_x, test_x, train_y, test_y = train_test_split(data.drop('Species', axis=1), data['Species'], test_size=0.2, random_state=0,stratify=data['Species'])

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(train_x,train_y)

pred = rf.predict(test_x)

#print(pred)
#print(test_label)


test_y = np.array(test_y)

CM = np.zeros((3,3))
for i in range(pred.shape[0]):
    CM[test_y[i]][pred[i]]+=1

print(CM)

#行求和，所有标签值为x的计数
row = np.sum(CM,axis=1)
#列求和，所有预测值为x的计数
col = np.sum(CM,axis=0)

classNames=np.arange(0,3,1)
PR=np.zeros(3)
RC=np.zeros(3)
F1=np.zeros(3)
tr=0

for i in range(3):
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

str = "RandomForest\n"
str += "ClassName\t\tPrecision\t\tRecall\t\t\tF1-score\n"
for i in range(3):
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