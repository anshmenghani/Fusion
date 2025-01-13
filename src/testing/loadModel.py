
import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import numpy as np

fusion_model = load_model("fusionModel1.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer}, safe_mode=False)
prediction = fusion_model.predict(np.array([[6041.846, 529.3467542, 20.9865, 20.9865, 9243.150979, 440.4331823, 20.9865, 440.4331823]]))
