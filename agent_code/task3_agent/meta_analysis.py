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
# labels = []
for name in MODELS:
    # ensure model subfolder
    if os.path.exists(f"model_{name}"):
        if os.path.isfile(f"model_{name}/test_results.pt"):
            with open(f"model_{name}/test_results.pt", "rb") as file:
                s += name + "_"
                results = pickle.load(file)
                # dist = (
                #     np.array(results["total_crates"]) - np.array(results["crates"])
                # ) / np.array(results["total_crates"])
                dist = np.array(results["points"])
                print(len(dist))
                #print(dist)
                dists.append(dist)
                # winrate = np.count_nonzero(dist == 1) / len(dist)
                # labels.append(f"{name}, wr: {round(winrate*100,1)}%")
                # names.append(name)
                print(">%s %.3f (%.3f)" % (name, np.mean(dist), np.std(dist)))
        else:
            print(f"FOUND NO MODEL NAMED {name}")
    else:
        print(f"FOUND NO MODEL NAMED {name}")
# plt.style.use('seaborn-whitegrid')
plt.grid(alpha=0.2)
medianprops = dict(linestyle="-", linewidth=2.5, color="firebrick")
#plt.boxplot(dists, showmeans=True, medianprops=medianprops)
bins = range(min(dists[0]), max(dists[0]) + 1)
plt.hist(dists[0], bins=bins, label="our agent")
with open("../rule_based_agent/points.pt", "rb") as file:
    rule_based = pickle.load(file)
    plt.hist(rule_based, bins=bins, label="rule based agent",alpha=0.6)
plt.xlabel("Points")
plt.ylabel("Games")
plt.legend()
plt.title("Points")

# ensure meta/plots subfolder
if not os.path.exists("meta/plots"):
    os.makedirs("meta/plots")
plt.savefig(f"meta/plots/{s}crate_percentage_boxplot.pdf")
plt.show()
