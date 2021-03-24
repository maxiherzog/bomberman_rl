# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 03:03:23 2021

@author: maxi
"""

import os

ROUNDS = 100
# os.system('cd ../..')
# os.system(f'python main.py play --agents task2_agent --no-gui --n-rounds {ROUNDS}')
import sys

AGENT = "task2_agent"
ALL_MODELS = True
MODELS = ["HANS"]  # ,"HANS_SYM", "PHILIPP"]

if ALL_MODELS:
    rootdir = f"agent_code/{AGENT}"
    for file in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir, file)):
            if file.startswith("model_"):
                model = file[6:]
                if not model.startswith("_") and model not in MODELS:
                    MODELS.append(model)
sys.argv = [
    "main.py",
    "play",
    "--agents",
    AGENT,
    "--no-gui",
    "--n-rounds",
    str(ROUNDS),
]

for i in range(len(MODELS)):
    os.environ["TESTING"] = "YES"
    os.environ["MODELNAME"] = MODELS[i]
    print(MODELS[i])
    exec(open("main.py").read())

os.environ["TESTING"] = "NO"
os.environ["MODELNAME"] = ""
