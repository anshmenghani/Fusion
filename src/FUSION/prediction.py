from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import joblib
import numpy as np

'''
A simple script to load the Fusion model and allow predicitons to be made based on the input parameters: 
EffectiveTemperature(Teff)(K), Luminosity(L/Lo), Radius(R/Ro), Diameter(D/Do), Volume(V/Vo), 
SurfaceArea(SA/SAo), GreatCircleCircumference(GCC/GCCo), and GreatCircleArea(GCA/GCAo), which are 
calculated from two of the three: EffectiveTemperature(Teff)(K), Luminosity(L/Lo), and/or Radius(R/Ro).
This script will output model predicitons in a readable format. 
'''

fusion_model = load_model("/Users/anshmenghani/Documents/GitHub/Fusion/src/FUSION/fusionModel.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer})
scaler = joblib.load("/Users/anshmenghani/Documents/GitHub/Fusion/src/FUSION/fusionStandard.pkl")

def model(data: np.array) -> np.array:
    if data.ndim != 2:
        data = data.reshape(1, -1)
    predictions = np.empty(shape=(data.shape[0], 16))
    for idx, i in enumerate(data):
        prediction = fusion_model.predict(np.array([i]), verbose=1)
        feat = [i.tolist() for i in prediction]
        trunc = feat[:12] + feat[14] 
        trunc2 = []
        for x in trunc:
            try:
                trunc2.append(x[0][0])
            except TypeError:
                trunc2.append(x[0])
        prediction = scaler.inverse_transform(np.array([trunc2]).reshape(1, -1)).flatten().tolist()
        prediction.insert(12, np.argmax(feat[12][0]))
        prediction.insert(13, np.argmax(feat[13][0]))
        prediction.insert(15, np.argmax(feat[15][0]))

        predictions[idx] = prediction
    return predictions
