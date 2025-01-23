# Returns Fusion accuracy and Mean Absolute Percentage Error by test sample 

import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import joblib
import numpy as np
import pandas as pd

def get_acc(y, y_hat, i=None):
    if i:
        acc = 100 - (100 * ((abs(y - y_hat)) / i))
        return acc
    acc = 100 - (100 * (abs((y-y_hat)/abs(y))))
    return acc


cols = ["AbsoluteBolometricMagnitude(Mbol)", "AbsoluteMagnitude(M)(Mv)", "AbsoluteBolometricLuminosity(Lbol)(log(W))", "Mass(M/Mo)", "AverageDensity(D/Do)", "CentralPressure(log(N/m^2))", "CentralTemperature(log(K))", "Lifespan(SL/SLo)", "SurfaceGravity(log(g)...log(N/kg))", "GravitationalBindingEnergy(log(J))", "BolometricFlux(log(W/m^2))", "Metallicity(log(MH/MHo))", "StarPeakWavelength(nm)"]
scaler = joblib.load("src/FUSION/fusionStandard.pkl")
x_test = pd.read_csv("src/FUSION/testData/x_test.csv")
x_test = x_test.iloc[:, 1:]
y_test = pd.read_csv("src/FUSION/testData/y_test.csv")
y_test = y_test.iloc[:, 1:]
fusion_model = load_model("src/FUSION/fusionModel.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer})
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
        y = float(y_test.values.tolist()[idx][id])
        if id == 12:
            acc_list.append(get_acc(y, round(pred[0], 0), i=id))
            continue
        elif id == 13:
            acc_list.append(get_acc(y, round(pred[0], 0), i=id))
            continue
        elif id == 15:
            acc_list.append(get_acc(y, round(pred[0], 0), i=id))
            continue
        acc_list.append(get_acc(y, pred))
            
    accuracy_list.append(acc_list)
    

def get_fusion_acc():
    return np.array(accuracy_list)


def get_fusion_mapes():
    accs = get_fusion_acc()
    mapes = np.array([[100-i for i in x] for x in accs])
    return mapes


print(np.mean(get_fusion_acc(), axis=0).tolist(), np.mean(get_fusion_mapes(), axis=0).tolist(), sep="\n\n")
