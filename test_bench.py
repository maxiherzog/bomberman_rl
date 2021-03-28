# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 03:03:23 2021

@author: maxi
"""

import os
import gc
ROUNDS = 100
# os.system('cd ../..')
# os.system(f'python main.py play --agents task2_agent --no-gui --n-rounds {ROUNDS}')
import sys

AGENT = "task3_agent"
# ALL_MODELS = True
MODELS = ["TEST"]  # ,"HANS_SYM", "PHILIPP"]

# if ALL_MODELS:
#     rootdir = f"agent_code/{AGENT}"
#     for file in os.listdir(rootdir):
#         if os.path.isdir(os.path.join(rootdir, file)):
#             if file.startswith("model_A"):
#                 model = file[6:]
#                 if not model.startswith("_") and model not in MODELS:
#                     MODELS.append(model)
args = [
    "main.py",
    "play",
    "--my-agent",
    AGENT,
    "--no-gui",
    "--n-rounds",
    str(ROUNDS),
]
sys.argv = args
command_str = "python "
for s in args:
    command_str += s+" "
print(command_str)


for i in range(len(MODELS)):
    os.environ["TESTING"] = "YES"
    os.environ["MODELNAME"] = MODELS[i]
    print(MODELS[i])
    os.system(command_str)
    gc.collect()
    # if res == 0:
    #     rootdir = f"agent_code/{AGENT}/model"
    #     # model ordner umbenennen!
    #     os.rename(rootdir, f"agent_code/{AGENT}/model_A{alpha}-G{gamma}")
    #     gc.collect()
    # else:
    #     raise

os.environ["TESTING"] = "NO"
os.environ["MODELNAME"] = ""
