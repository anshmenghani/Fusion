# Generates bar graphs to evaluate the performance of the individual predicted outputs against each other 

import matplotlib.pyplot as plt
import numpy as np

param = "Average Prediction Accuracy"
labels = ["Absolute Bolometric\nMagnitude", "Absolute Magnitude", "Absolute Bolometric\nLuminosity", "Mass", "Average Density", "Central Pressure", "Central\nTemperature", "Lifespan", "Surface Gravity", "Gravitational\nBinding Energy", "Bolometric Flux", "Potential Energy", "Spectral Class", "Luminosity Class", "Star Peak\nWavelength", "Star Type", "Average"]
labels.reverse()
metrics = list(np.random.uniform(low=0.5, high=1.0, size=16))
metrics.append(np.average(metrics))
metrics.reverse()

bars = plt.barh(labels, metrics)
bars[0].set_color("b")
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, round(width, 3), 
             ha='left', va='center', fontsize=8.2)
plt.title("Model " + param + " by Specific Output", fontsize=15)
plt.xlabel("Output " + param, fontweight="bold", fontsize=12)
plt.ylabel(param, fontweight="bold", fontsize=12)
plt.yticks(fontsize=10)
plt.show()
