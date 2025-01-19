
import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import numpy as np
import pandas as pd
import joblib

def get_acc(y, y_hat, i=None):
    if i is not None:
        length = len(i[0])
        acc = (abs(length - y_hat)) / length
        return acc
    acc = 100 - (100 * (abs((y-y_hat)/y)))
    return acc


x_test = pd.read_csv("src/FUSION/testData/x_test.csv", nrows=100)
x_test = x_test.iloc[:, 1:]
y_test = pd.read_csv("src/FUSION/testData/y_test.csv", nrows=100)
y_test = y_test.iloc[:, 1:]
fusion_model = load_model("src/FUSION/fusionModel.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer}, safe_mode=False)
scaler = joblib.load("src/FUSION/fusionStandard.pkl")
accuracy_list = []

for idx, i in enumerate(x_test.values.tolist()):
    prediction = fusion_model.predict(np.array([i]))
    prediction = [x[0] if isinstance(x[0], int) else x[0] for x in [i.tolist() for i in prediction]]
    prediction_trunc = prediction[0:12] + [prediction[14]] 
    for idx, i in enumerate(prediction_trunc):
        prediction_trunc[idx] = i[0]

    prediction_trunc = np.array(prediction_trunc).reshape(-1, 1)
    prediction_trunc = np.tile(prediction_trunc, (1, 13))
    prediction_trunc = scaler.inverse_transform(np.array(prediction_trunc))
    print(prediction_trunc)
    acc_list = []
    for id, pred in enumerate(prediction):
        try:
            pred = float(pred)
            y = float(y_test.values.tolist()[idx][id])
            acc_list.append(get_acc(y, pred))
        except (ValueError, TypeError):
            pred2 = np.argmax(pred)
            y = np.argmax(np.fromstring(y_test.values.tolist()[idx][id].replace("[", "").replace("]", ""), sep=" "))
            acc_list.append(get_acc(y, pred2, pred))
        
    accuracy_list.append(acc_list)
    
l = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
for i in accuracy_list:
    for idx, x in enumerate(i):
        l[idx].append(x)

print([sum(i)/len(i) for i in l])
    
#print(accuracy_list)
