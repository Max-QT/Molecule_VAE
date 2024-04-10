import pandas as pd
import os
from matplotlib import pyplot as plt

def plot_hist(hist, name, start=0, end=None):
    plt.figure()
    if end == None:
        plt.plot(hist[name][start:])
        plt.plot(hist["val_"+name][start:])
        plt.title(name)
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
    else:
        plt.plot(hist[name][start:end])
        plt.plot(hist["val_" + name][start:end])
        plt.title(name)
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")

csv_path = 'models/server/train_6/history.csv'
start = 0

df = pd.read_csv(csv_path)
plot_hist(df, "x_pred_categorical_accuracy", start)
plot_hist(df, "x_pred_loss", start)
plot_hist(df, "z_mean_log_var_loss", start)
plot_hist(df, "reg_prop_pred_loss", start)
plot_hist(df, "loss", start)
plt.show()