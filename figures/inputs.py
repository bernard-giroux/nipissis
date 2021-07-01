# -*- coding: utf-8 -*-

import pickle as pkl

import pandas as pd
from matplotlib import pyplot as plt

from catalog import catalog, Figure, Metadata

DISTANCES_SITES = {0: 93.02, 1: 739.80, 2: 537.42}


class Inputs(Metadata):
    def generate(self):
        data = pd.read_excel('donnees_ironore.xlsx')

        rms_val = pkl.load(open("rms_val.pkl", "rb"))
        data["RMS"] = rms_val

        for i in range(70):
            if i < 21:
                distance = DISTANCES_SITES[0]
            elif i < 45:
                distance = DISTANCES_SITES[1]
            else:
                distance = DISTANCES_SITES[2]
            data.loc[i, "Distance"] = distance

        data = data[
            ["No Train", "RMS", "Distance", "Vitesse (mph)", "Poids (Tonnes)"]
        ]
        data.columns = ["Train", "RMS", "Distance", "MPH", "Poids"]
        data.drop(65)
        data["MPH"] = data["MPH"].str.replace(" mph", "")
        data["MPH"] = data["MPH"].str.replace(" mi/h", "")
        data["MPH"] = data["MPH"].astype(float)
        data = data[~pd.isna(data["MPH"])]

        # with pd.option_context(
        #     'display.max_rows', None, 'display.max_columns', None,
        # ):
        #     print(data)

        for column in data.columns:
            self[column] = data[column].values


class Distance(Figure):
    Metadata = Inputs

    def plot(self, data):
        plt.scatter(data["Distance"], data["RMS"], s=12, c="k")
        plt.xlabel("Distance to railroad [m]")
        plt.ylabel("RMS amplitude [mm/s]")
        plt.suptitle("a)")
        plt.grid(True, which='major', color='k', alpha=.35)
        plt.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
        plt.minorticks_on()
        plt.tight_layout(rect=[0, 0, 1, 0.95])


class Velocity(Figure):
    Metadata = Inputs

    def plot(self, data):
        plt.scatter(data["MPH"], data["RMS"], s=12, c="k")
        plt.xlabel("Train velocity [mph]")
        plt.ylabel("RMS amplitude [mm/s]")
        plt.suptitle("b)")
        plt.grid(True, which='major', color='k', alpha=.35)
        plt.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
        plt.minorticks_on()
        plt.tight_layout(rect=[0, 0, 1, 0.95])


class Weight(Figure):
    Metadata = Inputs

    def plot(self, data):
        plt.scatter(data["Poids"], data["RMS"], s=12, c="k")
        plt.xlabel("Train weight [tonnes]")
        plt.ylabel("RMS amplitude [mm/s]")
        plt.suptitle("c)")
        plt.grid(True, which='major', color='k', alpha=.35)
        plt.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
        plt.minorticks_on()
        plt.tight_layout(rect=[0, 0, 1, 0.95])


catalog.register(Distance)
catalog.register(Velocity)
catalog.register(Weight)
