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

if ALL_MODELS:
    for file in os.listdir("."):
        if os.path.isdir(file):
            if file.startswith("model_"):
                model = file[6:]
                if not model.startswith("_") and model not in MODELS:
                    MODELS.append(model)
s = ""
labels = []
for name in MODELS:
    # ensure model subfolder
    if os.path.exists(f"model_{name}"):
        if os.path.isfile(f"model_{name}/test_results.pt"):
            with open(f"model_{name}/test_results.pt", "rb") as file:
                s += name + "_"
                results = pickle.load(file)
                dist = (
                    np.array(results["total_crates"]) - np.array(results["crates"])
                ) / np.array(results["total_crates"])
                dists.append(dist)
                winrate = np.count_nonzero(dist == 1) / len(dist)
                ALPHA = name.split("-")[0]
                GAMMA = name.split("-")[1]
                labels.append(f"{ALPHA}\n{GAMMA}\n wr: {round(winrate*100,1)}%")
                # names.append(name)
                print(">%s %.3f (%.3f)" % (name, np.mean(dist), np.std(dist)))
        else:
            print(f"FOUND NO MODEL NAMED {name}")
    else:
        print(f"FOUND NO MODEL NAMED {name}")
# plt.style.use('seaborn-whitegrid')
plt.grid(axis="y")
medianprops = dict(linestyle="-", linewidth=2.5, color="firebrick")
plt.boxplot(dists, labels=labels, showmeans=True, medianprops=medianprops)
plt.xlabel("Model")
plt.ylabel("Crate Destroyed Percentage")
plt.title("Crate Destroyed Percentage Distributions for Different Reward Models")

# ensure meta/plots subfolder
if not os.path.exists("meta/plots"):
    os.makedirs("meta/plots")
plt.savefig(f"meta/plots/{s}crate_percentage_boxplot.pdf")
plt.show()
