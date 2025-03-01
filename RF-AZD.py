import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Datasets/RawData/alzheimers_disease_data.csv')
data.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)

train_x, test_x, train_y, test_y = train_test_split(data.drop('Diagnosis', axis=1), data['Diagnosis'], test_size=0.2, random_state=0)

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(train_x,train_y)

pred = rf.predict(test_x)

#print(pred)
#print(test_label)
test_y = np.array(test_y)

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

str = "RandomForest\n"
str += "Precision\t\tRecall\t\tF1-score\n"

str += f"{format(PR,'.3f')}   \t\t{format(RC,'.3f')}" \
        f"   \t{format(F1,'.3f')}\n"

print(str)
