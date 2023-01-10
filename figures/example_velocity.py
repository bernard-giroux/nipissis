# -*- coding: utf-8 -*-

import numpy as np
import proplot as pplt

from example_probability import ExampleProbabilities_
from catalog import catalog, Figure

del catalog[-1]


class ExampleVelocities_(ExampleProbabilities_):
    V = np.linspace(0, 80, 24)
    D = np.full_like(V, 90)
    W = np.full_like(V, 20000*1000)
    ONES = np.full_like(V, 1)


class ExampleVelocities(Figure):
    Metadata = ExampleVelocities_

    def plot(self, data):
        THRESHOLD = 64

        rms = data['rms']
        prob = data['prob_rms']
        idx_threshold = np.argmax(rms > THRESHOLD)
        prob = np.sum(prob[idx_threshold:], axis=0)
        velocity = self.Metadata.V

        _, ax = pplt.subplots(figsize=[3.33, 3.33])

        ax.plot(velocity, prob*100, c='k')
        ax.format(
            xlabel="Velocity $v$ (km/h)",
            ylabel="Probability $p(y > y_t | \\bar{x}, D)$ (%)",
            xlim=[velocity.min(), velocity.max()],
            ylim=[0, 100],
        )


catalog.register(ExampleVelocities)
