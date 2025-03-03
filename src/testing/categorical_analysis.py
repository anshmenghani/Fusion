# Generates confusion matrices and performs accuracy, prescision, recall, and F1 score analysis on categorical outputs 

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

spec_class = np.load("src/testing/categorical_data/spec_class.npy")
lum_class = np.load("src/testing/categorical_data/lum_class.npy")
star_type = np.load("src/testing/categorical_data/star_type.npy")

def gen_analysis(y_true, y_pred, title, l):
    cm = confusion_matrix(y_true, y_pred, normalize="all")
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average="weighted")
    prescision = precision_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    return acc, recall, prescision, f1

print(gen_analysis(spec_class[0].astype(int).tolist(), spec_class[1].astype(int).tolist(), "Spectral Class Confusion Matrix", [0, 1, 2, 3, 4, 5, 6]))
print(gen_analysis(lum_class[0].astype(int).tolist(), lum_class[1].astype(int).tolist(), "Luminosity Class Confusion Matrix", [0, 1, 2, 3, 4, 5, 6]))
print(gen_analysis(star_type[0].astype(int).tolist(), star_type[1].astype(int).tolist(), "Star Type Confusion Matrix", [0, 1, 2, 3, 4, 5]))
