import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w
def average_every(x, w):
    arr = np.nanmean(np.pad(x.astype(float), ( 0, 0 if x.size % w == 0 else w - x.size % w ), mode='constant', constant_values=np.NaN).reshape(-1, w), axis=1)
    return arr

BATCHES = 50

with open("rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    arr = average_every(np.array(rewards),int(len(rewards)/BATCHES))
    plt.plot(np.arange(len(arr))*(len(rewards)/len(arr)), arr)
    plt.xlabel("episode")
    plt.ylabel("reward moving average")
    plt.title("rewards")
    plt.savefig("rewards.pdf")
    plt.show()
with open("Q-dists.pt", "rb") as file:
    Qdists = pickle.load(file)
    plt.plot(Qdists)
    plt.xlabel("episode")
    plt.ylabel("distance to original Q")
    plt.title("Q dist")
    plt.savefig("Qdist.pdf")
    plt.show()

    # Q = np.zeros((5, 14 * 2 + 1, 14 * 2 + 1, 2, 2))
    # Q[0, :, :14] = 1 # OBEN
    # Q[0, :, :14,  0, :] = 0
    # Q[2, :, -14:] = 1 # UNTEN
    # Q[2, :, -14:, 0, :] = 0
    # Q[3, :14, :] = 1 # LINKS
    # Q[3, :14,  :, :, 0] = 0
    # Q[1, -14:, :] = 1 # RECHTS
    # Q[1, -14:, :, :, 0] = 0
