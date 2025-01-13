from tensorflow.keras.saving import load_model

fusion_model = load_model("src/FUSION/fusionModel.keras", safe_mode=False)
