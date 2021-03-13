import pickle
import numpy as np
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')/w



with open("rewards.pt", "rb") as file:
    rewards = pickle.load(file)
    plt.plot(moving_average(rewards,int(len(rewards)/100)))
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
