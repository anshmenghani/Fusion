from tkinter import *
from PIL import Image, ImageTk
import tkinter.ttk as ttk
import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import joblib

# Create the main window
root = Tk()
root.title("FUSION: Fundemental Stellar Interactions using Optimized Neural Networks")

style = ttk.Style()
style.configure("myStyle.TCombobox", highlightthickness=0)

# Add image file 
bg = ImageTk.PhotoImage(Image.open("/Users/anshmenghani/Documents/GitHub/Fusion/src/FUSION/jwstDeepField.png"))
  
# Show image using label 
label1 = Label(root, image=bg) 
label1.place(x = -300, y = 0) 

label = Label(root, text="FUSION", font=("Lexend", 24, "bold"))
label.pack(pady=10)

# Add a label
top_label = Label(root, text="Specify Paramaters of the Star", font=("Lexend", 15, "italic"))
top_label.pack(pady=10)

teff_label = Label(root, text="Effective Temperature: ")
teff_label.pack()
teff = Entry(root, highlightthickness=0)
teff.insert(0, "Kelvin") 
teff.pack(pady=10)

lum_label = Label(root, text="Luminosity: ")
lum_label.pack()
lum = Entry(root, highlightthickness=0)
lum.insert(0, "L/L⊙") 
lum.pack(pady=10)

rad_label = Label(root, text="Radius: ", highlightcolor="white")
rad_label.pack()
rad = Entry(root, highlightthickness=0)
rad.insert(0, "R/R⊙") 
rad.pack(pady=10)

def get_preds():
    top_label.destroy()
    teff_label.destroy()
    lum_label.destroy()
    rad_label.destroy()
    teff.destroy()
    lum.destroy()
    rad.destroy()
    get_text_button.destroy()
    teffs = Label(root, "Effective Temperature: ")

    cols = ["AbsoluteBolometricMagnitude(Mbol)", "AbsoluteMagnitude(M)(Mv)", "AbsoluteBolometricLuminosity(Lbol)(log(W))", "Mass(M/Mo)", "AverageDensity(D/Do)", "CentralPressure(log(N/m^2))", "CentralTemperature(log(K))", "Lifespan(SL/SLo)", "SurfaceGravity(log(g)...log(N/kg))", "GravitationalBindingEnergy(log(J))", "BolometricFlux(log(W/m^2))", "Metallicity(log(MH/MHo))", "StarPeakWavelength(nm)"]
    scaler = joblib.load("src/FUSION/fusionStandard.pkl")
    #x_test = pd.read_csv("src/FUSION/testData/x_test.csv")
    #x_test = x_test.iloc[:, 1:]
    #y_test = pd.read_csv("src/FUSION/testData/y_test.csv")
    #df1 = y_test["LuminosityClass"]
    #print(np.unique(df1.values.tolist()))
    #y_test = y_test.iloc[:, 1:]
    fusion_model = load_model("src/FUSION/fusionModel.keras", custom_objects={"DLR": DLR, "LambdaLayerClass": LambdaLayerClass, "LossRewardOptimizer": LossRewardOptimizer})
    t = 0
    accuracy_list = []

    i = input('Enter Prediction Inputs ["EffectiveTemperature(Teff)(K)", "Luminosity(L/Lo)", "Radius(R/Ro)", "Diameter(D/Do)", "Volume(V/Vo)", "SurfaceArea(SA/SAo)", "GreatCircleCircumference(GCC/GCCo)", "GreatCircleArea(GCA/GCAo)"]: ')
    i = [float(l) for l in i.split(",")]
    print(i)
    prediction = fusion_model.predict(np.array([i]), verbose=1)
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
    print('Prediction ["AbsoluteBolometricMagnitude(Mbol)", "AbsoluteMagnitude(M)(Mv)", "AbsoluteBolometricLuminosity(Lbol)(log(W))", "Mass(M/Mo)", "AverageDensity(D/Do)", "CentralPressure(log(N/m^2))", "CentralTemperature(log(K))", "Lifespan(SL/SLo)", "SurfaceGravity(log(g)...log(N/kg))", "GravitationalBindingEnergy(log(J))", "BolometricFlux(log(W/m^2))", "Metallicity(log(MH/MHo))", "SpectralClass", "LuminosityClass", "StarPeakWavelength(nm)", "StarType"]')
    print(prediction)

get_text_button = Button(root, text="Get Path Graph", command=get_preds)
get_text_button.pack(pady=10)

# Run the main loop
root.mainloop()
