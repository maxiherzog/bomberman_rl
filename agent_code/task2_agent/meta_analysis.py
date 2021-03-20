import pickle
import numpy as np
import matplotlib.pyplot as plt

# compare distribution of in the following files
filenames = ["rewards"]


dists = []
names = ["blubblub"]
for name in filenames:
    with open("analysis/"+name+".pt", "rb") as file:
        distribution = pickle.load(file)

        dists.append(distribution)
        #names.append(name)
        print('>%s %.3f (%.3f)' % (name, np.mean(distribution), np.std(distribution)))

#plt.style.use('seaborn-whitegrid')
plt.grid(axis='y')
medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
plt.boxplot(dists, labels=names, showmeans=True, medianprops=medianprops)
plt.xlabel("Hyperparameter X")
plt.ylabel("reward")
plt.title("rewards distributions for different hyperparameters")
plt.savefig("analysis/rewards_comparison.pdf")
plt.show()
