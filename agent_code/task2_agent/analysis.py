import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def average_every(x, w):
    arr = np.nanmean(
        np.pad(
            x.astype(float),
            (0, 0 if x.size % w == 0 else w - x.size % w),
            mode="constant",
            constant_values=np.NaN,
        ).reshape(-1, w),
        axis=1,
    )
    return arr


BATCHES = 10

key = ["reward", "crates", "coins", "length", "useless_bombs"]
title = [
    "reward",
    "amount of crates destroyed",
    "amount of coins collected",
    "episode length",
    "amount of bombs destroying no crate",
]

# ensure analysis subfolder
if not os.path.exists("model/plots"):
    os.makedirs("model/plots")

with open("model/analysis_data.pt", "rb") as file:

    analysis_data = pickle.load(file)

    for i in range(len(key)):

        plt.plot(analysis_data[key[i]], "x", alpha=0.5, label="data points",markersize=2)
        if BATCHES != 1:
            arr = average_every(
                np.array(analysis_data[key[i]]),
                int(len(analysis_data[key[i]]) / BATCHES) + 1,
            )
            plt.plot(
                np.arange(len(arr)) * len(analysis_data[key[i]]) / len(arr)  + len(analysis_data[key[i]])/BATCHES/2,
                arr,
                label=f"average (batch size={int(len(analysis_data[key[i]])/BATCHES)})",
            )

        plt.xlabel("episode")
        plt.ylabel(title[i])
        plt.legend(loc="best")
        plt.title(title[i])
        plt.savefig(f"model/plots/{key[i]}.pdf")
        plt.show()
    # PLOT Qsum
    Qsum = analysis_data["Q_sum"]
    plt.plot(Qsum)
    plt.xlabel("episode")
    plt.ylabel("$\Sigma_{\{(s, a)\}}Q_{(s,a)}$")
    plt.title("sum of all Q values")
    plt.savefig("model/plots/Qsum.pdf")
    plt.show()
