# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import proplot as pplt

from bayesian_inference import (
    get_posterior, get_stats, plot_linear_dependency, plot_parameters,
)
from inputs import Inputs_
from catalog import catalog, Figure

del catalog[-1]


class Dependencies_(Inputs_):
    def generate(self):
        super().generate()

        rms = self["Amplitude"]
        distance = self["Distance"]
        velocity = self["Vitesse"]
        poids = self["Poids"]

        STEPS = 24  # NOTE Reduce step size to make computations faster.

        distance_dep = np.linspace(0.0, -.25, STEPS)
        velocity_dep = np.linspace(0.0, 4.5, STEPS)
        poids_dep = np.linspace(.0, 6E-6, STEPS)
        rms_0 = np.linspace(-40, 120, STEPS)
        rms_noise = np.logspace(1.3, 1.8, STEPS)
        vars = [distance_dep, velocity_dep, poids_dep, rms_0, rms_noise]

        posterior = get_posterior(
            vars, [distance, velocity, poids, np.ones_like(rms)], rms,
        )
        print(posterior.sum())
        _, _, vars_max, probs_mar, _, prob_null = get_stats(
            posterior, vars, null_dims=[1, 2],
        )
        print("Against H0:", 1/prob_null)
        print("Most probable model:", vars_max)
        _, _, _, _, _, prob_velocity = get_stats(
            posterior, vars, null_dims=[1],
        )
        print("Against H0 for velocity:", 1/prob_velocity)
        _, _, _, _, _, prob_weight = get_stats(
            posterior, vars, null_dims=[2],
        )
        print("Against H0 for weight:", 1/prob_weight)

        self["vars"] = vars
        self["posterior"] = posterior
        self["vars_max"] = vars_max
        self["probs_mar"] = probs_mar


class Dependencies(Figure):
    Metadata = Dependencies_

    def plot(self, data):
        QTY_VARS = 5
        _, axs = pplt.subplots(
            [
                [1, *[2]*QTY_VARS],
                [3, *range(4, 4+QTY_VARS)],
            ],
            ref=1,
            wratios=(1, *[1/QTY_VARS]*QTY_VARS),
            wspace=(None, *[.05]*(QTY_VARS-1)),
            figsize=[7.66, 7.66],
            sharey=False,
            sharex=False,
        )

        rms = data["Amplitude"]
        distance = data["Distance"]
        velocity = data["Vitesse"]
        weight = data["Poids"]
        a1, a2, a3, b, std = data["vars_max"]

        xs = [data["Distance"], data["Vitesse"], data["Poids"]]
        ys = [
            rms-a2*velocity-a3*weight,
            rms-a1*distance-a3*weight,
            rms-a1*distance-a2*velocity,
        ]
        as_ = [a1, a2, a3]
        xlabels = [
            "Distance to railroad $d$ (m)",
            "Train velocity $v$ (km/h)",
            "Train weight $w$ (kg)",
        ]
        ylabels = [
            "Contribution of distance to RMS amplitude \n"
            r"$y-\beta_v v-\beta_w w$ ($\frac{\mathrm{mm}}{\mathrm{s}}$)",
            "Contribution of velocity to RMS amplitude \n"
            r"$y-\beta_d d-\beta_w w$ ($\frac{\mathrm{mm}}{\mathrm{s}}$)",
            "Contribution of weight to RMS amplitude \n"
            r"$y-\beta_d d-\beta_v v$ ($\frac{\mathrm{mm}}{\mathrm{s}}$)",
        ]

        for ax, x, y, a, xlabel, ylabel, in zip(
            axs[:3], xs, ys, as_, xlabels, ylabels,
        ):
            plt.sca(ax)
            plot_linear_dependency(
                x,
                y,
                a=a,
                b=b,
                std=std,
                xlabel=xlabel,
                ylabel=ylabel,
            )

        vars = data["vars"]
        var_names = [
            r"$\beta_d$", r"$\beta_v$", r"$\beta_w$", r"$y_0$",
            r"$\sigma_\epsilon$",
        ]
        probs_mar = data["probs_mar"]
        plot_parameters(
            vars,
            var_names,
            probs_mar,
            axes=axs[3:],
            units=[
                r"\frac{\mathrm{mm}}{\mathrm{s} \cdot \mathrm{m}}",
                r"\frac{\mathrm{mm} \cdot \mathrm{h}}"
                r"{\mathrm{s} \cdot \mathrm{km}}",
                r"\frac{\mathrm{mm}}{\mathrm{s} \cdot \mathrm{kg}}",
                r"\frac{\mathrm{mm}}{\mathrm{s}}",
                r"\frac{\mathrm{mm}}{\mathrm{s}}",
            ],
        )
        axs[3].format(
            ylabel=(
                "Normalized marginal probability "
                "$\\frac{p(\\theta)}{p_{max}(\\theta)}$"
            ),
        )
        axs[3:].format(ylim=[0, 1], xmargin=.1)
        for ax in axs[3:]:
            ax.xaxis.label.set_fontsize(8)

        axs[-5].format(xticks=[0, -.15])
        axs[-4].format(xticks=[0, 3])
        axs[-3].format(xticks=[0, 4E-6], xformatter='sci')
        axs[-2].format(xticks=[0, 80])
        axs[-1].format(xscale='log', xticks=[3E1, 5E1])

        ticks = axs[2].get_xticks()
        axs[2].set_xticks(ticks[1::2])
        axs[2].format(xformatter='sci')

        axs[:4].format(abc=True)


catalog.register(Dependencies)
