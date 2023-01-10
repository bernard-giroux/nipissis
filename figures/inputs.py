# -*- coding: utf-8 -*-

import pickle as pkl

import pandas as pd
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
        data.columns = ["Train", "Amplitude", "Distance", "Vitesse", "Poids"]
        data.drop(65)
        data["Poids"] *= 1000
        data["Vitesse"] = data["Vitesse"].str.replace(" mph", "")
        data["Vitesse"] = data["Vitesse"].str.replace(" mi/h", "")
        data["Vitesse"] = data["Vitesse"].astype(float)
        data["Vitesse"] = data["Vitesse"] * 1.609344
        data = data[~pd.isna(data["Vitesse"])]

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
        axs.format(ylabel=r"RMS amplitude ($\frac{\mathrm{mm}}{\mathrm{s}}$)")
        labels = [
            "Distance to railroad (m)",
            "Train velocity (km/h)",
            "Train weight (kg)",
        ]
        x = data["Amplitude"]
        for y, label, ax in zip(
            [data["Distance"], data["Vitesse"], data["Poids"]], labels, axs,
        ):
            ax.scatter(y, x, color="k", size=12)
            ax.format(abc=True, xlabel=label, ylim=[0, None])
        ticks = axs[2].get_xticks()
        axs[2].set_xticks(ticks[1::2])
        axs[2].format(xformatter='sci')


catalog.register(Inputs)
