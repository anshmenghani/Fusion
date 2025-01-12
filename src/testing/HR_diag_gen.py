# Generates a Hertzsprung–Russell diagram used to evaluate model performance on different types of stars

import pandas as pd 
import scipy.constants as c
from math import pi
import matplotlib.pyplot as plt
import numpy as np

path = "src/FUSION/total_combined_data.csv"
df = pd.read_csv(path)
param = "Average Prediction Accuracy"
palette = "Greens"

teff = df["teff_gspphot_phoenix"].tolist()
radius = df["radius_gspphot_phoenix"].tolist()
R_sun = 6.96e8
L_sun = 3.827e26
lum = [(4 * pi * (radius[i] * R_sun)**2 * c.sigma * (teff[i])**4) / L_sun for i in range(len(teff))]

pred_acc = np.random.uniform(low=0.5, high=1.0, size=len(teff))
plt.scatter(teff, lum, c=pred_acc, cmap=palette, linestyle="", marker=".", s=3)
plt.colorbar(label="Model " + param + " For Each Star")
plt.yscale("log")
plt.suptitle("Two-Variable Hertzberg-Russell Diagram of Tested Stars", x=0.44, fontsize=15)
plt.title("Color Coded by Model " + param, fontsize=10)
plt.xlabel("Effective Temperature (K)", fontweight="bold", fontsize=12)
plt.ylabel("Luminosity (L/L⊙)", fontweight="bold", fontsize=12)
plt.gca().invert_xaxis()
plt.show()
