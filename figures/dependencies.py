# -*- coding: utf-8 -*-

import numpy as np

from bayesian_inference import (
    get_posterior, get_stats, plot_linear_dependency, plot_parameters,
)
from inputs import Inputs
from catalog import catalog, Figure


class Dependencies(Inputs):
    def generate(self):
        super().generate()
        print(self.keys())

        rms = self["RMS"][:]
        distance = self["Distance"][:]
        velocity = self["MPH"][:]
        poids = self["Poids"][:]

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


class DistanceDependency(Figure):
    Metadata = Dependencies

    def plot(self, data):
        rms = data["RMS"][:]
        distance = data["Distance"][:]
        velocity = data["MPH"][:]
        poids = data["Poids"][:]
        a1, a2, a3, b, std = data["vars_max"][:]
        plot_linear_dependency(
            distance,
            rms-a2*velocity-a3*poids,
            a=a1,
            b=b,
            std=std,
            xlabel="Distance à la voie ferrée $d$ [m]",
            ylabel="Contribution de la distance aux vibrations \n"
                   "$y-\\beta_v v-\\beta_w w$ [mm/s]",
            title="a)",
            savepath="fig/distance_dependency",
        )


class VelocityDependency(Figure):
    Metadata = Dependencies

    def plot(self, data):
        rms = data["RMS"][:]
        distance = data["Distance"][:]
        velocity = data["MPH"][:]
        poids = data["Poids"][:]
        a1, a2, a3, b, std = data["vars_max"][:]
        plot_linear_dependency(
            velocity,
            rms-a1*distance-a3*poids,
            a=a2,
            b=b,
            std=std,
            xlabel=r"Vitesse des trains $v$ [mph]",
            ylabel="Contribution de la vitesse aux vibrations \n"
                   "$y-\\beta_d d-\\beta_w w$ [mm/s]",
            title="b)",
            savepath="fig/velocity_dependency",
        )


class WeightDependency(Figure):
    Metadata = Dependencies

    def plot(self, data):
        rms = data["RMS"][:]
        distance = data["Distance"][:]
        velocity = data["MPH"][:]
        poids = data["Poids"][:]
        a1, a2, a3, b, std = data["vars_max"][:]
        plot_linear_dependency(
            poids,
            rms-a1*distance-a2*velocity,
            a=a3,
            b=b,
            std=std,
            xlabel="Poids des trains $w$ [tonnes]",
            ylabel="Contribution du poids aux vibrations \n"
                   "$y-\\beta_d d-\\beta_v v$ [mm/s]",
            title="c)",
            savepath="fig/weight_dependency",
        )


class Parameters(Figure):
    Metadata = Dependencies

    def plot(self, data):
        vars = data["vars"][:]
        var_names = [
            r"$\beta_d$", r"$\beta_v$", r"$\beta_w$", r"$y_0$",
            r"$\sigma_\epsilon$",
        ]
        probs_mar = data["probs_mar"][:]
        plot_parameters(
            vars,
            var_names,
            probs_mar,
            units=[
                "\\frac{mm}{s \\cdot m}",
                "\\frac{mm}{m}",
                "\\frac{mm}{s \\cdot tonnes}",
                "\\frac{mm}{s}",
                "\\frac{mm}{s}",
            ],
            title="d)",
            savepath="fig/params",
        )


catalog.register(DistanceDependency)
catalog.register(VelocityDependency)
catalog.register(WeightDependency)
catalog.register(Parameters)
