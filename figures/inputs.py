# -*- coding: utf-8 -*-

import pickle as pkl

import pandas as pd
from matplotlib import pyplot as plt
import proplot as pplt

from catalog import catalog, Figure, Metadata

DISTANCES_SITES = {0: 93.02, 1: 739.80, 2: 537.42}


class Inputs_(Metadata):
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


class Inputs(Figure):
    Metadata = Inputs_

    def plot(self, data):
        _, axs = pplt.subplots(
            [[1, 1, 2, 2], [0, 3, 3, 0]],
            figsize=[7.66, 7.66],
            sharey=True,
            sharex=False,
        )
        axs.format(
            ylabel="RMS amplitude (mm/s)",
            grid=True,
            gridminor=True,
        )
        labels = [
            "Distance to railroad (m)",
            "Train velocity (mph)",
            "Train weight (tons)",
        ]
        x = data["RMS"]
        for y, label, ax in zip(
            [data["Distance"], data["MPH"], data["Poids"]], labels, axs,
        ):
            ax.scatter(y, x, color="k", size=12)
            ax.format(abc=True, xlabel=label, ylim=[0, None])


catalog.register(Inputs)
