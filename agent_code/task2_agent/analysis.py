import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"]=200

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


BATCHES = 30  # 50

# with open(r"model.pt", "rb") as file:
#     Q = pickle.load(file)
#     print("Total entries:", len(Q.get_all()))

with open("analysis/rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    if BATCHES != 1:
        arr = average_every(np.array(rewards), int(len(rewards) / BATCHES) + 1)
        plt.plot(np.arange(len(arr)) * len(rewards) / len(arr), arr)
    else:
        plt.plot(rewards)
    plt.xlabel("episode")
    plt.ylabel("reward averaged (batch size=" + str(BATCHES) + ")")
    plt.title("rewards")
    plt.savefig("analysis/rewards_averaged.pdf")
    plt.show()

    # plot without averaging
    plt.plot(rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("rewards")
    plt.savefig("analysis/rewards.pdf")
    plt.show()

with open("analysis/crates.pt", "rb") as file:
    crates = pickle.load(file)
    if BATCHES != 1:
        arr = average_every(np.array(crates), int(len(crates) / BATCHES) + 1)
        plt.plot(np.arange(len(arr)) * len(crates) / len(arr), arr)
    else:
        plt.plot(crates)
    plt.xlabel("episode")
    plt.ylabel("amount of destroyed crates averaged (batch size=" + str(BATCHES) + ")")
    plt.title("Number of crates destroyed (averaged)")
    plt.savefig("analysis/crates_averaged.pdf")
    plt.show()

    # plot without averaging
    plt.plot(crates)
    plt.xlabel("episode")
    plt.ylabel("amount of destroyed crates")
    plt.title("Number of crates destroyed")
    plt.savefig("analysis/crates.pdf")
    plt.show()

with open("analysis/coins.pt", "rb") as file:
    coins = pickle.load(file)
    if BATCHES != 1:
        arr = average_every(np.array(coins), int(len(coins) / BATCHES) + 1)
        plt.plot(np.arange(len(arr)) * len(coins) / len(arr), arr)
    else:
        plt.plot(coins)
    plt.xlabel("episode")
    plt.ylabel("amount of collected coins averaged (batch size=" + str(BATCHES) + ")")
    plt.title("Number of coins collected (averaged)")
    plt.savefig("analysis/coins_averaged.pdf")
    plt.show()

    # plot without averaging
    plt.plot(coins)
    plt.xlabel("episode")
    plt.ylabel("amount of collected coins")
    plt.title("Number of coins collected")
    plt.savefig("analysis/coins.pdf")
    plt.show()

try:
    with open("analysis/beta-dists.pt", "rb") as file:
        betadists = pickle.load(file)
        plt.plot(betadists)
        plt.xlabel("episode")
        plt.ylabel(r"distance to original $\beta$")
        plt.title(r"$\beta$ dist")
        plt.savefig("analysis/betadist.pdf")
        plt.show()

except:
    pass
