# Returns Fusion accuracy by test sample 

import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import joblib
import numpy as np
import pandas as pd

def get_acc(y, y_hat, i=None):
    if i is not None:
        length = len(i)
        acc = 100 - (100 * ((abs(y - y_hat)) / length))
        return acc
    acc = 100 - (100 * (abs((y-y_hat)/y)))
    return acc


cols = ["AbsoluteBolometricMagnitude(Mbol)", "AbsoluteMagnitude(M)(Mv)", "AbsoluteBolometricLuminosity(Lbol)(log(W))", "Mass(M/Mo)", "AverageDensity(D/Do)", "CentralPressure(log(N/m^2))", "CentralTemperature(log(K))", "Lifespan(SL/SLo)", "SurfaceGravity(log(g)...log(N/kg))", "GravitationalBindingEnergy(log(J))", "BolometricFlux(log(W/m^2))", "Metallicity(log(MH/MHo))", "StarPeakWavelength(nm)"]
scaler = joblib.load("src/FUSION/fusionStandard.pkl")
x_test = pd.read_csv("src/FUSION/testData/x_test.csv", nrows=100)
x_test = x_test.iloc[:, 1:]
y_test = pd.read_csv("src/FUSION/testData/y_test.csv", nrows=100)
y_test = y_test.iloc[:, 1:]
fusion_model = load_model("src/FUSION/fusionModel.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer}, safe_mode=False)
accuracy_list = []

for idx, i in enumerate(x_test.values.tolist()):
    prediction = fusion_model.predict(np.array([i]))
    feat = [i.tolist() for i in prediction]
    trunc = feat[:12] + feat[14] 
    trunc2 = []
    for i in trunc:
        try:
            trunc2.append(i[0][0])
        except TypeError:
            trunc2.append(i[0])
    prediction = scaler.inverse_transform(np.array([trunc2]).reshape(1, -1)).flatten().tolist()
    prediction.insert(12, feat[12][0])
    prediction.insert(13, feat[13][0])
    prediction.insert(15, feat[15][0])

    acc_list = []
    for id, pred in enumerate(prediction):
        try:
            y = float(y_test.values.tolist()[idx][id])
            acc_list.append(get_acc(y, pred))
        except (ValueError, TypeError):
            pred2 = np.argmax(pred)
            y = np.argmax(np.fromstring(y_test.values.tolist()[idx][id].replace("[", "").replace("]", ""), sep=" "))
            acc_list.append(get_acc(y, pred2, pred))
        
    accuracy_list.append(acc_list)
    

def get_fusion_acc():
    return accuracy_list

