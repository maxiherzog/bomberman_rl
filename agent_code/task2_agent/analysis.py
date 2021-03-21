import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"]=200
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

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


BATCHES = 50  # 50

# with open(r"model.pt", "rb") as file:
#     Q = pickle.load(file)
#     print("Total entries:", len(Q.get_all()))

with open("analysis/rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    plt.plot(rewards, alpha=0.5, label="data points")
    if BATCHES != 1:
        arr = average_every(np.array(rewards), int(len(rewards) / BATCHES) + 1)
        plt.plot(np.arange(len(arr)) * len(rewards) / len(arr), arr, label=f"average (batch size={int(len(rewards)/BATCHES)})")

    plt.xlabel("episode")
    plt.ylabel("total reward obtained")
    plt.legend(loc="upper left")
    plt.title("Rewards")
    plt.savefig("analysis/rewards.pdf")
    plt.show()

with open("analysis/crates.pt", "rb") as file:
    crates = pickle.load(file)
    plt.plot(crates, alpha=0.5, label="data points")
    if BATCHES != 1:
        arr = average_every(np.array(crates), int(len(crates) / BATCHES) + 1)
        plt.plot(np.arange(len(arr)) * len(crates) / len(arr), arr, label=f"average (batch size={int(len(crates)/BATCHES)})")

    plt.xlabel("episode")
    plt.ylabel("amount of destroyed crates")
    plt.legend(loc="upper left")
    plt.title("Number of crates destroyed")
    plt.savefig("analysis/crates.pdf")
    plt.show()

with open("analysis/coins.pt", "rb") as file:
    coins = pickle.load(file)
    plt.plot(coins, alpha=0.5, label="data points")
    if BATCHES != 1:
        arr = average_every(np.array(coins), int(len(coins) / BATCHES) + 1)
        plt.plot(np.arange(len(arr)) * len(coins) / len(arr), arr, label=f"average (batch size={int(len(coins)/BATCHES)})")

    plt.xlabel("episode")
    plt.ylabel("amount of coins collected")
    plt.legend(loc="upper left")
    plt.title("Number of collected coins")
    plt.savefig("analysis/crates.pdf")
    plt.show()

try:
    with open("analysis/Q-dists.pt", "rb") as file:
        Qdists = pickle.load(file)
        plt.plot(Qdists)
        plt.xlabel("episode")
        plt.ylabel("$\Sigma_{\{(s, a)\}}Q_{(s,a)}$")
        plt.title("sum of Q values")
        plt.savefig("analysis/Qdist.pdf")
        plt.show()

except:
    pass
