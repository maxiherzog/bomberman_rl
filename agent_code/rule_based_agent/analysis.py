# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 10:25:34 2021

@author: Philipp
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def moving_average(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')

fileDir = os.path.dirname(os.path.realpath('__file__'))
print(fileDir)

#For accessing the file inside a sibling folder.
filename_rewards = os.path.join(fileDir, '../task1_agent/rewards.pt')
filename_rewards = os.path.abspath(os.path.realpath(filename_rewards))

#For accessing the file inside a sibling folder.
filename_episodes = os.path.join(fileDir, '../task1_agent/episode_lengths.pt')
filename_episodes = os.path.abspath(os.path.realpath(filename_episodes))

#For accessing the file inside a sibling folder.
filename_len = os.path.join(fileDir, '../task1_agent/len.pt')
filename_len = os.path.abspath(os.path.realpath(filename_len))

#For accessing the file inside a sibling folder.
filename_Qdist = os.path.join(fileDir, '../task1_agent/Q-dists.pt')
filename_Qdist = os.path.abspath(os.path.realpath(filename_Qdist))
with open("steps.pt", "rb") as file1:
    with open(filename_episodes, "rb") as file2:
        with open(filename_len, "rb") as file3:
            len1 = pickle.load(file1)
            len2 = pickle.load(file2)
            len3 = pickle.load(file3)
            arr1 = moving_average(np.array(len1), 20)
            arr2 = moving_average(np.array(len2), 30)
            arr3 = moving_average(np.array(len3), 30)
            
            plt.plot(arr1, label = "rule-based agent", color="orange")
            plt.plot(arr2, label = "our agent", color="g")
            plt.plot(arr3, label = "our agent without Q", color="purple")
            
            plt.xlabel("episode")
            plt.ylabel("episode length")
            
            plt.legend()
            
            plt.savefig("comp.pdf")
            plt.show()
'''
with open(filename_rewards, "rb") as file1:
    rew = pickle.load(file1)
    arr1 = moving_average(np.array(rew), 30)
    
    plt.plot(arr1, label = "our agent", color="g")
    
    plt.xlabel("episode")
    plt.ylabel("rewards")
        
    plt.legend()
        
    plt.savefig("rewards.pdf")
    plt.show()

with open(filename_Qdist, "rb") as file1:
    Q = pickle.load(file1)
    arr1 = moving_average(np.array(Q), 30)
    
    plt.plot(arr1, label = "our agent", color="g")
    
    plt.xlabel("episode")
    plt.ylabel("Q-dist")
        
    plt.legend()
        
    plt.savefig("Qdist.pdf")
    plt.show()
'''