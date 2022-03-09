# -*- coding: utf-8 -*-

import numpy as np
import proplot as pplt

from bayesian_inference import gaussian
from dependencies import Dependencies_
from catalog import catalog, Figure


class ExampleProbabilities_(Dependencies_):
    D = 90
    V = 30
    W = 20000
    ONES = 1

    @property
    def X(self):
        return np.array([self.D, self.V, self.W, self.ONES])

    def generate(self):
        super().generate()
        vars = self["vars"]
        self['rms'] = rms = np.linspace(0, 400, 48)

        vars = np.array(np.meshgrid(*vars, copy=False, indexing='ij'))

        if self.X.ndim == 1:
            mean = np.einsum('i,i...', self.X, vars[:-1])
        else:
            mean = np.einsum('ij,i...->j...', self.X, vars[:-1])
        mean = mean[None]
        noise = vars[-1]
        noise = noise[None]
        rms = np.expand_dims(rms, tuple(range(1, mean.ndim)))

        prob = gaussian(x=rms, mean=mean, std=noise)
        axes = (0, *range(self.X.ndim, prob.ndim))
        prob /= np.sum(prob, axis=axes, keepdims=True)

        self['prob_rms'] = prob


class ExampleProbabilities(Figure):
    Metadata = ExampleProbabilities_

    def plot(self, data):
        rms = data['rms']
        prob = data['prob_rms']
        prob = np.sum(prob, axis=tuple(range(1, prob.ndim)))
        prob = np.cumsum(prob[::-1])[::-1]

        _, ax = pplt.subplots(figsize=[3.33, 3.33])

        ax.plot(rms, prob, c='k')
        ax.format(
            xlabel="Amplitude threshold $y_t$ (mm/s)",
            ylabel="Probability $p(y > y_t | \\bar{x})$ (―)",
            xlim=[0, rms.max()],
            ylim=[0, 1],
        )


catalog.register(ExampleProbabilities)
