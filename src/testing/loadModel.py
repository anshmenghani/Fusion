
import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import numpy as np
import pandas as pd

def get_acc(y, y_hat):
    epsilon = 1e-9
    try:
        return 100 - (100 * (abs(y-y_hat)/y))
    except ZeroDivisionError:
        return 100 - (100 * (abs(y-y_hat)/epsilon))


x_test = pd.read_csv("src/FUSION/testData/x_test.csv", nrows=100)
x_test = x_test.iloc[:, 1:]
y_test = pd.read_csv("src/FUSION/testData/y_test.csv", nrows=100)
y_test = y_test.iloc[:, 1:]
fusion_model = load_model("src/FUSION/fusionModel.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer}, safe_mode=False)
accuracy_list = []

for idx, i in enumerate(x_test.values.tolist()):
    prediction = fusion_model.predict(np.array([i]))
    acc_list = []
    for id, pred in enumerate(prediction):
        try:
            pred = float(pred)
            y = float(y_test.values.tolist()[idx][id])
        except (ValueError, TypeError):
            pred = np.argmax(pred)
            y = np.argmax(np.fromstring(y_test.values.tolist()[idx][id].replace("[", "").replace("]", ""), sep=" "))
        acc_list.append(get_acc(pred, y))
    accuracy_list.append(acc_list)
    
print([a[-1] for a in accuracy_list])
