# Gets Fusion model predictions and evaluates their speed and accuracy

import sys 
sys.path.insert(1, "src/FUSION")
import fusion
import numpy as np
import time

model, test_data = fusion.Fuse()
x_test = list(test_data[0])
y_test = list(test_data[1])


def get_acc(y, y_hat):
    y_hat = y_hat.inverse_transform(y_hat)
    return 100 - (100 * (abs(y-y_hat)/y))


def get_fusion_pred_data():
    predictions_acc = []
    times = []
    for idx in range(x_test):
        acc_list = []

        start = time.time()
        prediction = model.predict(np.array(x_test))
        end = time.time()
        times.append(end - start)
        
        prediction = list(prediction)
        for pidx, p in enumerate(prediction):
            real_y = y_test[idx][pidx]
            if type(real_y) == float or type(real_y) == int:
                acc_list.append(get_acc(real_y, p))
            else:
                max_val = np.argmax(p)
                z = list(np.zeros(len(real_y)))
                z[max_val] = 1.
                if real_y == z:
                    acc_list.append(1.)
                else:
                    acc_list.append(0.)

        acc_list.append(np.average(acc_list))
        predictions_acc.append(acc_list)

    return predictions_acc, times


def get_evaluation():
    model.evaluate(x_test, y_test)
