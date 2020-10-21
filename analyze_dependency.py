# -*- coding: utf-8 -*-

import pickle as pkl

import pandas as pd
from matplotlib import pyplot as plt

DISTANCES_SITES = {0: 93.02, 1: 739.80, 2: 537.42}


train_data = pd.read_excel('donnees_ironore.xlsx')

rms_val = pkl.load(open("rms_val.pkl", "rb"))
train_data["RMS"] = rms_val

for i in range(70):
    if i < 21:
        distance = DISTANCES_SITES[0]
    elif i < 45:
        distance = DISTANCES_SITES[1]
    else:
        distance = DISTANCES_SITES[2]
    train_data.loc[i, "Distance"] = distance

train_data = train_data[
    ["No Train", "RMS", "Distance", "Vitesse (mph)", "Poids (Tonnes)"]
]
train_data.columns = ["Train", "RMS", "Distance", "MPH", "Poids"]
train_data.drop(65)
train_data["MPH"] = train_data["MPH"].str.replace(" mph", "")
train_data["MPH"] = train_data["MPH"].str.replace(" mi/h", "")
train_data["MPH"] = train_data["MPH"].astype(float)
train_data = train_data[~pd.isna(train_data["MPH"])]

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_data)

plt.figure(figsize=(12, 8))
plt.scatter(train_data["Distance"], train_data["RMS"])
plt.xlabel("Distance")
plt.ylabel("RMS")
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(train_data["MPH"], train_data["RMS"])
plt.xlabel("MPH")
plt.ylabel("RMS")
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.scatter(train_data["Poids"], train_data["RMS"])
plt.xlabel("Poids")
plt.ylabel("RMS")
plt.grid()
plt.show()
