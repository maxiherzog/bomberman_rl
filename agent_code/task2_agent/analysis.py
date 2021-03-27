import pickle
import numpy as np

import os

############################################
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["font.family"] = "STIXGeneral"

mpl.rcParams["axes.prop_cycle"] = cycler(
    "color", ["green", "tomato", "navy", "orchid", "yellowgreen", "goldenrod"]
)

############################################



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


BATCHES = 20

MODELS = []
parameters = []
for file in os.listdir("."):
    if os.path.isdir(file):
        if file.startswith("model_A"):
            model = file[6:]
            if not model.startswith("_") and model not in MODELS:
                MODELS.append(model)
                a, g = model.split("-")
                parameters.append((float(a[1:]), float(g[1:])))

key = ["reward", "crates", "coins", "length", "bombs", "useless_bombs"]
print(key)
title = [
    "Reward",
    "Amount of Crates Destroyed",
    "Amount of Coins Collected",
    "Episode Length",
    "Amount of Bombs Dropped",
    "Number of Bombs That Did Not Destroy Any Crates",
]

axis = [
    "$r$",
    "Amount of Crates Destroyed",
    "Amount of Coins Collected",
    r"$\tau$",
    "Amount of Bombs Dropped",
    "Number of Bombs That Did Not Destroy Any Crates",
]

for m, MODEL in enumerate(MODELS):
    print(MODEL)
    # ensure analysis subfolder
    if not os.path.exists(f"model_{MODEL}/plots"):
        os.makedirs(f"model_{MODEL}/plots")

    with open(f"model_{MODEL}/analysis_data.pt", "rb") as file:

        analysis_data = pickle.load(file)
        wins = None
        # print(analysis_data)
        if "win" in analysis_data.keys():

            wins = np.array(analysis_data["win"])
            print("winrate",  round(np.count_nonzero(wins)/len(wins)*100,3), "%")
        for i in range(len(key)):
            data = np.array(analysis_data[key[i]])
            if wins is not None:
                plt.plot(
                    np.arange(len(data))[~wins],
                    data[~wins],
                    "x",
                    alpha=0.2,
                    label="data points",
                    markersize=2,
                )
            else:
                plt.plot(
                    data, "x", alpha=0.2, label="data points", markersize=2
                )

            if BATCHES != 1:
                arr = average_every(
                    data,
                    int(len(data) / BATCHES) + 1,
                )
                plt.plot(
                    np.arange(len(arr)) * len(data) / len(arr)
                    + len(data) / BATCHES / 2,
                    arr,
                    label=f"average (batch size={int(len(analysis_data[key[i]])/BATCHES)})",
                )

            if wins is not None:
                plt.plot(
                    np.arange(len(data))[wins],
                    data[wins],
                    "x",
                    alpha=0.2,
                    label="data points wins",
                    markersize=2,
                )

            plt.xlabel("Episode")
            plt.ylabel(axis[i])
            plt.legend(loc="best")
            plt.title(title[i])
            plt.suptitle(fr" $\alpha={parameters[m][0]} $ $\gamma={parameters[m][1]}$")
            plt.savefig(f"model_{MODEL}/plots/{key[i]}.pdf")
            plt.show()
        # PLOT Qsum
        # name = ["", "_move", "_bomb", "_wait"]
        # title = ["mean", "moving", "placing a bomb", "waiting"]
        # for i in range(len(name)):
        #
        #
        #     if name[i]=="":
        #         Q = np.array(Q)/6
        #     plt.plot(Q, label=title[i])
        if "Q_sum" in analysis_data:
            Qsum = analysis_data["Q_sum"]
            plt.plot(Qsum)
            plt.xlabel("Episode")
            plt.ylabel("$\Sigma_{\{(s, a)\}}Q_{(s,a)}$")
            plt.title("Sum of All $Q$ Values")
            plt.savefig(f"model_{MODEL}/plots/Qsum.pdf")
            plt.show()

        # PLOT Q for one situation
        # ACTIONS = ["UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB"]
        # Qsit = np.asmatrix(analysis_data["Q_situation"])
        # print(Qsit)
        # for i in range(Qsit.shape[1]):
        #     plt.plot(Qsit[:,i], label=ACTIONS[i])
        # plt.xlabel("episode")
        # plt.ylabel("$\Sigma_{\{(s, a)\}}Q_{(s,a)}$")
        # plt.title("sum of all Q values for different actions for the given starting situation")
        # plt.legend()
        # plt.savefig("model/plots/Qsit.pdf")
        # plt.show()
