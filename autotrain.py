# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 03:03:23 2021

@author: maxi
"""

import os
import subprocess

ROUNDS = 1000
# os.system('cd ../..')
# os.system(f'python main.py play --agents task2_agent --no-gui --n-rounds {ROUNDS}')
import sys
import gc



AGENT = "task2_agent"
# ALL_MODELS = True
ALPHAS = [0.02, 0.01]
GAMMAS = [0.5, 0.6, 0.7]
EPSILON_MAX = 0.025
EPSILON_MIN = 0.025
EPSILON_DECAY = 1

# if ALL_MODELS:
#     rootdir = f"agent_code/{AGENT}"
#     for file in os.listdir(rootdir):
#         if os.path.isdir(os.path.join(rootdir, file)):
#             if file.startswith("model_"):
#                 model = file[6:]
#                 if not model.startswith("_") and model not in MODELS:
#                     MODELS.append(model)
args = [
    "main.py",
    "play",
    "--agents",
    AGENT,
    "--no-gui",
    "--n-rounds",
    str(ROUNDS),
    "--train",
    "1"
]

sys.argv = args
command_str = "python "
for s in args:
    command_str += s+" "
print(command_str)


for gamma in GAMMAS:
    for alpha in ALPHAS:
        os.environ["AUTOTRAIN"] = "YES"
        os.environ["ALPHA"] = str(alpha)
        os.environ["GAMMA"] = str(gamma)
        os.environ["EPSILON_MIN"] = str(EPSILON_MIN)
        os.environ["EPSILON_MAX"] = str(EPSILON_MAX)
        os.environ["EPSILON_DECAY"] = str(EPSILON_DECAY)
        print("alpha", alpha, ";gamma", gamma)
        res = os.system(command_str)
        if res == 0:
            rootdir = f"agent_code/{AGENT}/model"
            # model ordner umbenennen!
            os.rename(rootdir, f"agent_code/{AGENT}/model_A{alpha}-G{gamma}")
            gc.collect()
        else:
            raise

os.environ["AUTOTRAIN"] = "NO"
