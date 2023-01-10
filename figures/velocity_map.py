# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import proplot as pplt

from location import RIVER, Location, Location_, closest_distances
from example_probability import ExampleProbabilities_
from catalog import catalog

del catalog[-1]

RMS_THRESHOLD = 64
ADMISSIBLE_PROBS = [.25, .5, .75]


class VelocityMap_(Location_):
    def generate(self):
        super().generate()
        railway = self["railway"]
        railway = np.mean([railway[:-1], railway[1:]], axis=0)
        river = np.pad(RIVER.T, [[0, 0], [0, 1]])
        N = 1000
        x, xp = np.linspace(0, 1, N), np.linspace(0, 1, len(river))
        river = np.array(
            [
                np.interp(x, xp, river[:, 0]),
                np.interp(x, xp, river[:, 1]),
                np.full(N, 0),
            ]
        )
        river = river.T
        distances = closest_distances(railway, river)
        velocities = np.arange(0., 45., 5.)

        prob = []
        for d in distances:
            class Probability(ExampleProbabilities_):
                V = velocities
                D = np.full_like(V, d)
                W = np.full_like(V, 20000*1000)
                ONES = np.full_like(V, 1)

            prob_data = Probability()
            prob_data.generate()
            rms = prob_data['rms']
            prob_rms = prob_data['prob_rms']
            idx_threshold = np.argmax(rms > RMS_THRESHOLD)
            prob.append(np.sum(prob_rms[idx_threshold:], axis=0))
        prob = np.array(prob)

        self['distances_river'] = distances
        self['velocities'] = velocities
        self['prob'] = prob


class VelocityMap(Location):
    Metadata = VelocityMap_

    def plot(self, data, show_river=False):
        railway = data["railway"]
        prob = data["prob"]
        velocities = data['velocities']
        satellite = data["satellite"]

        fig, axs = pplt.subplots(
            ncols=len(ADMISSIBLE_PROBS),
            figsize=[7.66, 4.5],
            sharey=True,
            sharex=True,
        )

        lines = []
        satellite = satellite[10:-50, 38:-24]
        xlim = [705800, 708000]
        ylim = [5597200, 5601100]
        for ax, admissible_prob in zip(axs, ADMISSIBLE_PROBS):
            mask = prob > admissible_prob
            axis = 1
            # The following is taken from https://stackoverflow.com/a/47269413.
            idx_admissible = np.where(
                mask.any(axis=axis),
                mask.argmax(axis=axis),
                -1,
            )
            v = velocities[idx_admissible]

            # This visualization is inspired from https://matplotlib.org/stable
            # /gallery/lines_bars_and_markers/multicolored_line.html
            points = railway[:, :2]
            segments = np.array([points[:-1], points[1:]])
            segments = segments.swapaxes(0, 1)
            norm = plt.Normalize(velocities.min(), velocities.max())
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(v)
            lc.set_linewidth(4)
            lines = ax.add_collection(lc)

            if show_river:
                ax.plot(
                    *RIVER,
                    c='cyan',
                    label="River",
                    zorder=100,
                )
            ax.imshow(satellite, zorder=-1, extent=[*xlim, *ylim])
            ax.set_aspect('equal')
        for ax in axs:
            loc = ax.get_xticks()
            ax.set_xticks(loc[1::4].astype(int))
        loc = ax.get_yticks()
        axs[0].set_yticks(loc[1::2].astype(int))
        axs[0].set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")
        axs.format(
            abc=True,
            xlabel="Easting (UTM)",
            ylabel="Northing (UTM)",
            xlim=xlim,
            ylim=ylim,
            xformatter='{x:d}',
            yformatter='{x:d}',
            xrotation=90,
        )
        fig.colorbar(lines)


catalog.register(VelocityMap)
