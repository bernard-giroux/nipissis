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
        self['rms'] = rms = np.linspace(0, 350, 48)

        vars = np.array(np.meshgrid(*vars, copy=False, indexing='ij'))

        if self.X.ndim == 1:
            mean = np.einsum('i,i...', self.X, vars[:-1])
        else:
            mean = np.einsum('ij,i...->j...', self.X, vars[:-1])
        noise = vars[-1]
        rms = np.expand_dims(rms, tuple(range(1, mean.ndim)))

        prob = []
        posterior = self["posterior"]
        for rms_ in rms:
            prob_ = gaussian(x=rms_, mean=mean, std=noise)
            prob_ *= posterior
            axes = tuple(range(self.X.ndim-1, prob_.ndim))
            prob.append(np.sum(prob_, axis=axes))
        prob = np.array(prob)
        prob /= np.sum(prob, axis=0, keepdims=True)

        self['prob_rms'] = prob


class ExampleProbabilities(Figure):
    Metadata = ExampleProbabilities_

    def plot(self, data):
        rms = data['rms']
        prob = data['prob_rms']
        prob = np.cumsum(prob[::-1])[::-1]

        _, ax = pplt.subplots(figsize=[3.33, 3.33])

        ax.plot(rms, prob*100, c='k')
        ax.format(
            xlabel="Amplitude threshold $y_t$ (mm/s)",
            ylabel="Probability $p(y > y_t | \\bar{x}, D)$ (%)",
            xlim=[0, rms.max()],
            ylim=[0, 100],
        )


catalog.register(ExampleProbabilities)
