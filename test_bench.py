# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 03:03:23 2021

@author: maxi
"""

import os

ROUNDS = 1000
# os.system('cd ../..')
# os.system(f'python main.py play --agents task2_agent --no-gui --n-rounds {ROUNDS}')
import sys

MODELS = ["HANS"]  # ,"HANS_SYM", "PHILIPP"]


sys.argv = [
    "main.py",
    "play",
    "--agents",
    "task2_agent",
    "--no-gui",
    "--n-rounds",
    str(ROUNDS),
]

for i in range(len(MODELS)):
    os.environ["TESTING"] = "YES"
    os.environ["MODELNAME"] = MODELS[i]
    exec(open("main.py").read())

os.environ["TESTING"] = "NO"
os.environ["MODELNAME"] = ""
