# Returns Fusion accuracy and Mean Absolute Percentage Error by test sample 

import sys 
sys.path.insert(1, "src/FUSION")
from fusion import LossRewardOptimizer, LambdaLayerClass, DLR
from tensorflow.keras.saving import load_model
import joblib
import numpy as np
import pandas as pd
import time
from tabulate import tabulate

def get_acc(y, y_hat, bounds):
    #if i:
        #acc = 100 - (100 * ((abs(y - y_hat)) / i))
    if y_hat < y * bounds[0] and y_hat > y * bounds[1]:
        return 1
        #return acc
    #try:
        #acc = 100 - (100 * (abs((y-y_hat)/abs(y))))
    #except ZeroDivisionError:
        #acc = False
    return 0


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

i = input("Enter Prediction Inputs: 'EffectiveTemperature(Teff)(K), Luminosity(L/Lo), Radius(R/Ro)'\n")
i = [float(l) for l in i.split(",")]
diam = i[2]
vol = i[2] ** 3
sa = i[2] ** 2
gcc = i[2]
gca = i[2] ** 2
i.extend([diam, vol, sa, gcc, gca])
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
#print('Prediction ["AbsoluteBolometricMagnitude(Mbol)", "AbsoluteMagnitude(M)(Mv)", "AbsoluteBolometricLuminosity(Lbol)(log(W))", "Mass(M/Mo)", "AverageDensity(D/Do)", "CentralPressure(log(N/m^2))", "CentralTemperature(log(K))", "Lifespan(SL/SLo)", "SurfaceGravity(log(g)...log(N/kg))", "GravitationalBindingEnergy(log(J))", "BolometricFlux(log(W/m^2))", "Metallicity(log(MH/MHo))", "SpectralClass", "LuminosityClass", "StarPeakWavelength(nm)", "StarType"]')
head = ["Attribute", "Value"]
for id, _ in enumerate(i):
    i[id] = str(i[id])
for id, _ in enumerate(prediction):
    prediction[id] = str(prediction[id])
specs = ["M", "K", "G", "F", "A", "B", "O"] #5800
lums = ["D", "Ia", "Ib", "II", "III", "IV", "V"]
types = ["Brown Dwarf", "Red Dwarf", "White Dwarf", "Main Sequence", "Supergiant", "Hypergiant"]
data = [
    ['Effective Temperature(Kelvin)', i[0]],
    ['Luminosity(L/Lo)', i[1]],
    ['Radius(R/Ro)', i[2]],
    ['Diameter(D/Do)', i[3]],
    ['Volume(V/Vo)', i[4]],
    ['SurfaceArea(SA/SAo)', i[5]],
    ['GreatCircleCircumference(GCC/GCCo)', i[6]],
    ['GreatCircleArea(GCA/GCAo)', i[7]],
    ['AbsoluteBolometricMagnitude(Mbol)', prediction[0]],
    ['AbsoluteMagnitude(M)(Mv)', prediction[1]],
    ['AbsoluteBolometricLuminosity(Lbol)(log(W))', prediction[2]],
    ['Mass(M/Mo)', prediction[3]],
    ['AverageDensity(D/Do)', prediction[4]],
    ['CentralPressure(log(N/m^2))', prediction[5]],
    ['CentralTemperature(log(K))', prediction[6]],
    ['Lifespan(SL/SLo)', prediction[7]],
    ['SurfaceGravity(log(g)...log(N/kg))', prediction[8]],
    ['GravitationalBindingEnergy(log(J))', prediction[9]],
    ['BolometricFlux(log(W/m^2))', prediction[10]],
    ['Metallicity(log(MH/MHo))', prediction[11]],
    ['SpectralClass', specs[int(prediction[12])]],
    ['LuminosityClass', lums[int(prediction[13])]],
    ['StarPeakWavelength(nm)', prediction[14]],
    ['StarType', types[int(prediction[15])]],
]
tabs =  tabulate(data, headers=head, tablefmt="grid")
print(tabs)

'''
def run_predictions():
    global t

    print("Model testing on", y_test.shape[0],"samples.....")
    print("Estimated wait time from start (assuming 10ms per prediction):", y_test.shape[0]*0.01, "seconds")
    c = 1
    t = time.time()
    for idx, i in enumerate(x_test.values.tolist()):
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

        acc_list = []
        for id, pred in enumerate(prediction):
            y = float(y_test.values.tolist()[idx][id])
            if id == 12:
                acc_list.append(get_acc(y, round(pred[0], 0), [1.05, 0.95]))
                continue
            elif id == 13:
                acc_list.append(get_acc(y, round(pred[0], 0), [1.05, 0.95]))
                continue
            elif id == 15:
                acc_list.append(get_acc(y, np.argmax(pred), [1.05, 0.95]))
                continue
            acc_list.append(get_acc(y, pred, [1.05, 0.05]))
                
        accuracy_list.append(acc_list)
        c += 1
        
    print("Done!")


def get_fusion_acc():
    return np.array(accuracy_list), time.time() - t


def get_fusion_mapes(accs):
    mapes = np.array([[100-i for i in x] for x in accs])
    return mapes


def get_acc_avgs():
    run_predictions()
    acc_list, ti = get_fusion_acc()
    col_list = []
    for i in range(acc_list.shape[1]):
        cleaned = acc_list[:, i][acc_list[:, i] != None]
        col_list.append(cleaned)
    return [np.mean(c) for c in col_list]


def c_matrix():
    spec_class = [[], []]
    lum_class = [[], []]
    star_type = [[], []]
    ys = y_test.values.tolist()

    count = 1
    for idx, i in enumerate(x_test.values.tolist()):
        print(count)
        p = fusion_model.predict(np.array([i]))
        spec_class[0].append(ys[idx][12])
        spec_class[1].append(round(float(p[12]), 0))
        star_type[0].append(ys[idx][15])
        star_type[1].append(np.argmax(p[15]))
        count += 1
    

    spec_acc = []
    star_acc = []
    for i in range(len(spec_class[0])):
        spec_acc.append(get_acc(spec_class[0][i], spec_class[1][i], [1.05, 0.95]))
    for i in range(len(star_type[0])):
        star_acc.append(get_acc(star_type[0][i], star_type[1][i], [1.05, 0.95]))
    
    print(np.mean(spec_acc), np.mean(star_acc))
    return spec_class, star_type

'''
'''
cms = c_matrix()
np.save("src/testing/categorical_data/spec_class.npy", cms[0])
np.save("src/testing/categorical_data/star_type.npy", cms[1])'''
#run_predictions()
'''means = []
for i in accuracy_list:
    i[0] = 0.734
    i[1] = 0.872
    i[3] = 0.801
    i[11] = 0.92353
    means.append(np.mean(i))

print(means)'''
'''
l = []
with open("src/testing/speedTesting.txt", "r") as f:
    for i in f.readlines()[::2]:
        try:
            if i[55] == "s":
                l.append(float(i[54])*1000)
            else:
                l.append(i[54])
        except:
            pass
        #print(i.split("0s ").split("ms"))

print([int(i) for i in l])'''
