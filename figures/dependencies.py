# -*- coding: utf-8 -*-

import numpy as np

from bayesian_inference import (
    get_posterior, get_stats, plot_linear_dependency, plot_parameters,
)
from analyze_dependency import train_data

variables = train_data[["RMS", "Distance", "MPH", "Poids"]]

rms, distance, velocity, poids = variables.values.T

STEPS = 24  # NOTE Reduce step size to make computations faster.

distance_dep = np.linspace(0.0, -.25, STEPS)
velocity_dep = np.linspace(0.0, 5., STEPS)
poids_dep = np.linspace(.0, .005, STEPS)
rms_0 = np.linspace(-40, 120, STEPS)
rms_noise = np.logspace(1.2, 1.7, STEPS)
vars = [distance_dep, velocity_dep, poids_dep, rms_0, rms_noise]
var_names = [
    r"$\beta_d$", r"$\beta_v$", r"$\beta_w$", r"$y_0$",
    r"$\sigma_\epsilon$",
]

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
a1, a2, a3, b, std = vars_max

# Maximum 24-channel mean RMS amplitude [mm/s].
# Mean over 8 minutes.
# Maximum over train passage.

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
