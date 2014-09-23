usage="""a module that will house fancy visualization routines that interact with bayesburst data products"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams.update({"text.usetex":True})

import numpy as np
import healpy as hp

#=================================================
# General purpose plotting functions
#=================================================
ax_pos1       = [0.15, 0.1, 0.6, 0.6]
ax_right_pos = [0.75, 0.1, 0.2, 0.6]
ax_top_pos   = [0.15, 0.7, 0.6, 0.2]

ax_pos = [0.15, 0.1, 0.7, 0.8]
#=================================================
def scat_hist(x, y, fig_tuple=None, marker="o", markerfacecolor="none", markeredgecolor="b", markersize=5, normed=False, log=True, n_per_bin=10):
        """ scatter x vs y with projected histograms """
        if fig_tuple == None:
                fig = plt.figure()
                ax = fig.add_axes(ax_pos1)
                ax_right = fig.add_axes(ax_right_pos)
                ax_top = fig.add_axes(ax_top_pos)
                fig_tuple = (fig, ax, ax_right, ax_top)
        else:
                fig, ax, ax_right, ax_top = fig_tuple

        _min = min(min([_ for _ in x if _>-np.infty]),min([_ for _ in y if _>-np.infty]))
        _max = max(max([_ for _ in x if _<np.infty]),max([_ for _ in y if _<np.infty]))

        _max = _max*(1.1)**(_max/abs(_max))
        if _max == 0: _max = 0.1
        _min = _min*(1.1)**(-_min/abs(_min))
        if _min == 0: _min = -0.1

        n_bins = min(int(len(x)/n_per_bin), 50)

        if log:
                bins = np.logspace(np.log10(_min), np.log10(_max), n_bins+1)
                ax.loglog(x, y, marker=marker, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor, linestyle="none")
                ax_right.hist(y, bins=bins, histtype="step", normed=normed, orientation="horizontal")
                ax_top.hist(x, bins=bins, histtype="step", normed=normed, orientation="vertical")
                ax_right.set_yscale("log")
                ax_top.set_xscale("log")
                ax_right.set_xlim(xmin=0)
                ax_top.set_ylim(ymin=0)
        else:
                bins = np.linspace(_min, _max, n_bins+1)
                ax.plot(x, y, marker=marker, markersize=markersize, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor, linestyle="none")
                ax_right.hist(y, bins=bins, histtype="step", normed=normed, orientation="horizontal")
                ax_top.hist(x, bins=bins, histtype="step", normed=normed, orientation="vertical")
                ax_right.set_xlim(xmin=0)
                ax_top.set_ylim(ymin=0)
        if normed:
                ax_right.set_xlabel("pdf")
                ax_top.set_ylabel("pdf")
        else:
                ax_right.set_xlabel("count")
                ax_top.set_ylabel("count")

        plt.setp(ax_right.get_yticklabels(), visible=False)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        return fig_tuple

###
def dual_hist(x, fig_tuple=None, n_per_bin=10, n_samples=1001, log=False, label=None, color="b"):
        """ overlays cumulative with differential histogram """
        if fig_tuple == None:
                fig = plt.figure()
                ax = fig.add_axes(ax_pos)
                ax1 = ax.twinx()
                fig_tuple = (fig, ax, ax1)
        else:
                fig, ax, ax1 = fig_tuple

        _min = min([_ for _ in x if _>-np.infty])
        _max = max([_ for _ in x if _<np.infty])

        _max = _max*(1.1)**(_max/abs(_max))
        if _max == 0: _max = 0.1
        _min = _min*(1.1)**(-_min/abs(_min))
        if _min == 0: _min = -0.1

        N = 1.0*len(x)
        n_bins = min(int(N/n_per_bin), 50)
        if log:
                bins = np.logspace(np.log10(_min), np.log10(_max), n_bins+1)
                samples = np.logspace(np.log10(_min), np.log10(_max), n_samples)
        else:
                bins = np.linspace(_min, _max, n_bins+1)
                samples = np.linspace(_min, _max, n_samples)

        ax.hist(x, bins=bins, histtype="step", label=label, color=color)
        ax.set_ylim(ymin=0)
        cum = np.zeros_like(samples)
        for _ in x:
                cum += _<=samples
        cum /= N
        std = (cum*(1-cum)/N)**0.5
        ax1.fill_between(samples, cum-std, cum+std, alpha=0.25, color=color)
        ax1.plot(samples, cum, label=label, color=color)

        if log:
                ax.set_xscale("log")
                ax1.set_xscale("log")

        ax1.yaxis.tick_right()

        return fig_tuple

###
def square(ax, ax_right=None, ax_top=None):
        """ make plots square """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        _min = min(xmin, ymin)
        _max = max(xmax, ymax)
        ax.set_xlim(xmin=_min, xmax=_max)
        ax.set_ylim(ymin=_min, ymax=_max)
        if ax_right!=None:
                ax_right.set_ylim(ymin=_min, ymax=_max)
        if ax_top!=None:
                ax_top.set_xlim(xmin=_min, xmax=_max)

#=================================================
# plotting functions for specific data types
#=================================================
### WRITE ME

"""
mollweide projections
	different color-map options
	inj, est position markers
	cos_dtheta tracer
	area shader (searched_area)
	contours

strain visualizations at a single position
	mle estimate
	mle +gaussian errors
	MAP estimate? +error bars
	at geocenter and in each detector datastream

interactive mollweide with strain 


else?
"""
	
