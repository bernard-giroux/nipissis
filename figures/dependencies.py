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

        rms = self["RMS"]
        distance = self["Distance"]
        velocity = self["MPH"]
        poids = self["Poids"]

        STEPS = 24  # NOTE Reduce step size to make computations faster.

        distance_dep = np.linspace(0.0, -.25, STEPS)
        velocity_dep = np.linspace(0.0, 5., STEPS)
        poids_dep = np.linspace(.0, .005, STEPS)
        rms_0 = np.linspace(-40, 120, STEPS)
        rms_noise = np.logspace(1.2, 1.7, STEPS)
        vars = [distance_dep, velocity_dep, poids_dep, rms_0, rms_noise]

        posterior = get_posterior(
            vars, [distance, velocity, poids, np.ones_like(rms)], rms,
        )
        print(posterior.sum())
        _, prob_max, _, vars_max, probs_mar, _, prob_null = get_stats(
            posterior, vars, null_dims=[1, 2],
        )
        print("Against H0:", prob_max / prob_null)
        print("Most probable model:", vars_max)
        _, _, _, _, _, _, prob_velocity = get_stats(
            posterior, vars, null_dims=[1],
        )
        print("Against H0 for velocity:", prob_velocity / prob_null)
        _, _, _, _, _, _, prob_weights = get_stats(
            posterior, vars, null_dims=[2],
        )
        print("Against H0 for weight:", prob_weights / prob_null)

        self["vars"] = vars
        self["posterior"] = posterior
        self["vars_max"] = vars_max
        self["probs_mar"] = probs_mar


# class DistanceDependency(Figure):
#     Metadata = Dependencies
#
#     def plot(self, data):
#         rms = data["RMS"]
#         distance = data["Distance"]
#         velocity = data["MPH"]
#         poids = data["Poids"]
#         a1, a2, a3, b, std = data["vars_max"]
#         plot_linear_dependency(
#             distance,
#             rms-a2*velocity-a3*poids,
#             a=a1,
#             b=b,
#             std=std,
#             xlabel="Distance to railroad $d$ [m]",
#             ylabel="Contribution of distance to RMS amplitude \n"
#                    "$y-\\beta_v v-\\beta_w w$ [mm/s]",
#         )
#
#
# class VelocityDependency(Figure):
#     Metadata = Dependencies
#
#     def plot(self, data):
#         rms = data["RMS"]
#         distance = data["Distance"]
#         velocity = data["MPH"]
#         poids = data["Poids"]
#         a1, a2, a3, b, std = data["vars_max"]
#         plot_linear_dependency(
#             velocity,
#             rms-a1*distance-a3*poids,
#             a=a2,
#             b=b,
#             std=std,
#             xlabel=r"Train velocity $v$ [mph]",
#             ylabel="Contribution of velocity to RMS amplitude \n"
#                    "$y-\\beta_d d-\\beta_w w$ [mm/s]",
#         )
#
#
# class WeightDependency(Figure):
#     Metadata = Dependencies
#
#     def plot(self, data):
#         rms = data["RMS"]
#         distance = data["Distance"]
#         velocity = data["MPH"]
#         poids = data["Poids"]
#         a1, a2, a3, b, std = data["vars_max"]
#         plot_linear_dependency(
#             poids,
#             rms-a1*distance-a2*velocity,
#             a=a3,
#             b=b,
#             std=std,
#             xlabel="Train weight $w$ [tonnes]",
#             ylabel="Contribution of weight to RMS amplitude \n"
#                    "$y-\\beta_d d-\\beta_v v$ [mm/s]",
#         )
#
#
# class Parameters(Figure):
#     Metadata = Dependencies
#
#     def plot(self, data):
#         vars = data["vars"]
#         var_names = [
#             r"$\beta_d$", r"$\beta_v$", r"$\beta_w$", r"$y_0$",
#             r"$\sigma_\epsilon$",
#         ]
#         probs_mar = data["probs_mar"]
#         plot_parameters(
#             vars,
#             var_names,
#             probs_mar,
#             units=[
#                 "\\frac{mm}{s \\cdot m}",
#                 "\\frac{mm}{m}",
#                 "\\frac{mm}{s \\cdot tons}",
#                 "\\frac{mm}{s}",
#                 "\\frac{mm}{s}",
#             ],
#         )


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
            wspace=(None, *[0]*(QTY_VARS-1)),
            figsize=[7.66, 7.66],
            sharey=False,
            sharex=False,
        )
        axs.format(
            grid=True,
            gridminor=True,
        )

        rms = data["RMS"]
        distance = data["Distance"]
        velocity = data["MPH"]
        weight = data["Poids"]
        a1, a2, a3, b, std = data["vars_max"]

        xs = [data["Distance"], data["MPH"], data["Poids"]]
        ys = [
            rms-a2*velocity-a3*weight,
            rms-a1*distance-a3*weight,
            rms-a1*distance-a2*velocity,
        ]
        as_ = [a1, a2, a3]
        xlabels = [
            "Distance to railroad $d$ (m)",
            "Train velocity $v$ (mph)",
            "Train weight $w$ (tons)",
        ]
        ylabels = [
            "Contribution of distance to RMS amplitude \n"
            "$y-\\beta_v v-\\beta_w w$ (mm/s)",
            "Contribution of velocity to RMS amplitude \n"
            "$y-\\beta_d d-\\beta_w w$ (mm/s)",
            "Contribution of weight to RMS amplitude \n"
            "$y-\\beta_d d-\\beta_v v$ (mm/s)",
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
                "\\frac{mm}{s \\cdot m}",
                "\\frac{mm}{m}",
                "\\frac{mm}{s \\cdot tons}",
                "\\frac{mm}{s}",
                "\\frac{mm}{s}",
            ],
        )
        axs[3].format(
            ylabel=(
                "Normalized marginal probability "
                "$\\frac{p(\\theta)}{p_{max}(\\theta)}$"
            )
        )

        # axs[-1].get_xaxis().set_ticklabels([])
        axs[-1].set_xscale('log')
        # axs[-1].set_xticks([2E1, 6E1], [2E1, 6E1])

        ticks = axs[2].get_xticks()
        axs[2].set_xticks(ticks[1::2])

        axs[:4].format(abc=True)


catalog.register(Dependencies)
