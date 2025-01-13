
import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model

fusion_model = load_model("fusionModel1.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer}, safe_mode=False)
