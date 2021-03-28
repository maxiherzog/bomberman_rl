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

# compare distribution of in the following files
# filenames = ["rewards"]

dists = []

ALL_MODELS = True
MODELS = []

folder = "./"

if ALL_MODELS:
    print(os.listdir(f"{folder}"))
    for file in sorted(os.listdir(f"{folder}")):
        if os.path.isdir(folder+file):
            if file.startswith("model_"):
                model = file[6:]
                if not model.startswith("_") and model not in MODELS:
                    MODELS.append(model)
print(MODELS)
s = ""
labels = []
for name in MODELS:
    # ensure model subfolder
    if os.path.exists(f"{folder}model_{name}"):
        if os.path.isfile(f"{folder}model_{name}/test_results.pt"):
            with open(f"{folder}model_{name}/test_results.pt", "rb") as file:
                s += name + "_"
                results = pickle.load(file)
                #dist = (
                #    np.array(results["total_crates"]) - np.array(results["crates"])
                #) / np.array(results["total_crates"])
                try:
                    dist = np.array((results["points"]))
                    dists.append(dist)
                    winrate = np.count_nonzero(dist == 9) / len(dist)
                    ALPHA = name.split("-")[0]
                    GAMMA = name.split("-")[1]
                    labels.append(f"{ALPHA}\n{GAMMA}\n wr: {round(winrate * 100, 1)}%")
                    # names.append(name)
                    print(">%s %.3f (%.3f)" % (name, np.mean(dist), np.std(dist)))
                except:
                    pass

        else:
            print(f"FOUND NO MODEL NAMED {name}")
    else:
        print(f"FOUND NO MODEL NAMED {name}")
# plt.style.use('seaborn-whitegrid')
plt.grid(axis="y")
medianprops = dict(linestyle="-", linewidth=2.5, color="firebrick")
plt.boxplot(dists, labels=labels, showmeans=True, medianprops=medianprops)
plt.xlabel("Model")
# plt.ylabel("Crate Destroyed Percentage")
# plt.title("Crate Destroyed Percentage Distributions for Different Reward Models")

plt.ylabel("Number of Collected Coins")
plt.title("Number of Collected Coins for Different Reward Models")
# ensure meta/plots subfolder
if not os.path.exists(f"{folder}meta/plots"):
    os.makedirs(f"{folder}meta/plots")
plt.savefig(f"{folder}meta/plots/{s}coins_collected_boxplot.pdf")
plt.show()
