# -*- coding: utf-8 -*-

import numpy as np
import proplot as pplt

from example_probability import ExampleProbabilities_
from catalog import catalog, Figure


class ExampleDistance_(ExampleProbabilities_):
    D = np.linspace(0, 1000, 24)
    V = np.full_like(D, 40)
    W = np.full_like(D, 32000)
    ONES = np.full_like(D, 1)


class ExampleDistance(Figure):
    Metadata = ExampleDistance_

    def plot(self, data):
        THRESHOLD = 250

        rms = data['rms']
        prob = data['prob_rms']
        idx_threshold = np.argmax(rms > THRESHOLD)
        prob = np.sum(prob[idx_threshold:], axis=0)
        distance = self.Metadata.D
        for d in [500, 740]:
            print(f"Probability at {d} meters:", prob[np.argmax(distance > d)])

        _, ax = pplt.subplots(figsize=[3.33, 3.33])

        ax.plot(distance, prob*100, c='k')
        ax.format(
            xlabel="Distance to railroad $d$ (m)",
            ylabel="Probability $p(y > y_t | \\bar{x}, D)$ (%)",
            xlim=[distance.min(), distance.max()],
            ylim=[0, 100],
        )


catalog.register(ExampleDistance)
