"""Bayesian linear regression by complete enumeration.

This file is modified from Simon Jérome, Dufour-Beauséjour Sophie (2020)
RS2_bayesian_regression.
https://github.com/CloudyOverhead/RS2_bayesian_linear_regression.

This algorithm was developed for data analysis associated with
S. Dufour-Beauséjour, M Bernier, J. Simon, S. Homayouni, V. Gilbert,
Y. Gauthier, J. Tuniq, A. Wendleder and A. Roth (2020) Tenuous Correlation
between Snow Depth or Sea Ice Thickness and C- or X-Band Backscattering in
Nunavik Fjords of the Hudson Strait. Remote Sensing. 13(4), 768.
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def get_posterior(vars, xs, y):
    *as_, std = vars
    n = len(as_) + 2

    for i, a in enumerate(as_):
        new_shape = np.ones(n, dtype=int)
        new_shape[i] = -1
        as_[i] = a.reshape(new_shape)

    new_shape = np.ones(n, dtype=int)
    new_shape[-2] = -1
    std = std.reshape(new_shape)

    new_shape = np.ones(n, dtype=int)
    new_shape[-1] = -1
    xs = [x.reshape(new_shape) for x in xs]
    y = y.reshape(new_shape)

    posterior = gaussian(
        x=y,
        mean=np.sum([a * x for a, x in zip(as_, xs)], axis=0),
        std=std,
    )
    posterior = np.prod(posterior, axis=-1)

    return posterior


def get_stats(posterior, vars, null_dims, print_=True):
    # Maximum posterior probability
    argmax = np.argmax(posterior)
    prob_max = posterior.flatten()[argmax]

    # Mean and std of each parameter's marginal distribution at prob_max
    unravel_argmax = list(np.unravel_index(argmax, posterior.shape))
    vars_max = [var[unravel_argmax[i]] for i, var in enumerate(vars)]
    probs_mar = marginal_distributions(unravel_argmax, posterior)
    std_mar = [weighted_std(var, prob) for var, prob in zip(vars, probs_mar)]
    # prob_uniform = posterior.mean()

    prob_null = get_prob_null(posterior, vars, null_dims)

    return (
        argmax,
        prob_max,
        unravel_argmax,
        vars_max,
        probs_mar,
        std_mar,
        prob_null,
    )


def get_prob_null(posterior, vars, null_dims):
    for dim in sorted(null_dims)[::-1]:
        posterior = np.moveaxis(posterior, dim, -1)
        idx_null = np.argmin(np.abs(vars[dim]))
        posterior = posterior[..., idx_null]
    return posterior.max()


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    return np.sqrt(variance)


def marginal_distributions(argmax, posterior):
    distributions = []
    for i in range(posterior.ndim):
        axis = tuple(j for j in range(posterior.ndim) if i != j)
        distributions.append(np.sum(posterior, axis=axis))
    return distributions


def gaussian(x, mean, std):
    exponent = -((x-mean)/std)**2 / 2
    denominator = np.sqrt(2*np.pi) * std
    return np.exp(exponent) / denominator


def gaussian_fill_between(a, b, std, xlim=None, ylim=None):
    if xlim is None:
        xlim = plt.xlim()
    if ylim is None:
        ylim = plt.ylim()

    # Alternative representation.
    # for s, a in zip(range(1, 4), [.25, .125, .0625]):
    #     plt.fill_between(
    #         p,
    #         line-s*snow_noise,
    #         line+s*snow_noise,
    #         alpha=a,
    #         color="tab:blue",
    #     )
    #     plt.imshow(gaussian_fill_between)

    extent = [*xlim, *ylim]
    x, y = np.meshgrid(np.linspace(*xlim, 1000), np.linspace(*ylim, 1000))
    fill_between = gaussian(y, a*x + b, std)

    tab_blue = mpl.colors.to_rgb(mpl.colors.BASE_COLORS["k"])
    alpha = np.linspace(0, .3, 1000)
    cmap = mpl.colors.ListedColormap([[*tab_blue, a] for a in alpha])
    plt.imshow(
        fill_between,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap,
    )


def plot_linear_dependency(x, y, a, b, std, xlabel="", ylabel="", title="",
                           savepath=""):
    plt.figure(figsize=(12, 8))
    plt.scatter(x, y, s=12, c="k")
    extend = x.max() - x.min()
    x_line = np.linspace(x.min()-extend, x.max()+extend, 2)
    line = a*x_line + b
    gaussian_fill_between(a, b, std)
    plt.autoscale(False)
    plt.plot(x_line, line, ls='--', c="k")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    plt.grid(True, which='major', color='k', alpha=.35)
    plt.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
    plt.minorticks_on()
    if savepath:
        plt.savefig(savepath)
    plt.show()
    plt.close()


def plot_parameters(vars, var_names, probs_mar, units=None, title="",
                    savepath=""):
    plt.figure(figsize=(12, 8))
    gs = mpl.gridspec.GridSpec(1, len(var_names))
    axes = list()
    for i in np.arange(len(var_names)):
        axes.append(plt.subplot(gs[i]))

    axes[0].set_ylabel(
        "Probabilité marginale normalisée "
        "$\\frac{p(\\theta)}{p_{max}(\\theta)}$"
    )

    for i, (var, name) in enumerate(zip(vars, var_names)):
        ax = axes[i]
        width = np.diff(var)
        probs_mar_ = probs_mar[i] / probs_mar[i].max()
        print(name, 2*weighted_std(var, probs_mar_))
        ax.bar(var, probs_mar_, [*width, width[-1]], color=[.3, .3, .3])
        ax.grid(True, which='major', color='k', alpha=.1)
        ax.grid(True, which='minor', linestyle='--', color='k', alpha=.1)
        ax.minorticks_on()
        if units is not None:
            unit = units[i]
            name = f"{name} $\\left[{unit}\\right]$"
        ax.set_xlabel(name)
        if i == len(vars) - 1:
            ax.get_xaxis().set_ticklabels([])
            ax.set_xscale('log')
            ax.set_xticks([2E1, 6E1], [2E1, 6E1])
        if i > 0:
            ax.get_yaxis().set_ticklabels([])
        ax.tick_params(direction='in', which="both", right=1, top=0)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95 if title else 1.0])
    if savepath:
        plt.savefig(savepath)
    plt.show()
    plt.close()
