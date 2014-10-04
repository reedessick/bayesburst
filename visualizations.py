usage="""a module that will house fancy visualization routines that interact with bayesburst data products"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

plt.rcParams.update({"text.usetex":True})

import numpy as np
import healpy as hp

import injections

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

###
def project(network, freqs, h, theta, phi, psi, data, figtuple=None, dh=None, units="$1/\sqrt{\mathrm{Hz}}$"):
	"""
	projects the strain "h" onto the network and overlays it with "data" at each detector
	if supplied, "dh" is the upper/lower bounds for h and will be shaded on the plots
	"""
	n_ifo = len(network)
	n_freqs = len(freqs)

	n_axs = n_ifo+1
	if figtuple:
		fig, axs = figtuple
		if len(axs) != n_axs:
			raise ValueError, "inconsistent number of ifos between network and figtuples"

	if np.shape(data) != (n_freqs, n_ifo):
		raise ValueError, "bad shape for data"

	if np.shape(h) != (n_freqs):
		raise ValueError, "bad shape for h"

	if dh!=None and np.shape(dh) != (n_freqs, 2):
		raise ValueError, "bad shape for dh"

	fig = plt.figure()
	axs = []
	
	iax = 1
	### geocenter h
	ax = plt.subplot(n_ax, 1, iax)
	ax.plot(freqs, h.real, color="b", label="$\mathbb{R}\{h\}$")
	if np.any(h.imag):
		ax.plot(freqs, h.imag, color="r", label="$\mathrm{I}\{h\}$")
	if dh!=None:
		ax.fill_between(freqs, dh[:,0].real, dh[:,1].real, color="b", alpha=0.25)
		if np.any(dh.imag):
			ax.fill_between(freqs, dh[:,0].imag, dh[:,1].imag, color="r", alpha=0.25)

	axs.append( ax )
	iax+=1	

	### inject h into the network
	inj = injections.inject(network, h, theta, phi, psi=psi)
	if dh != None:
		dinj0 = np.transpose(injections.inject(network, dh[:,0], theta, phi, psi=psi))
		dinj1 = np.transpose(injections.inject(network, dh[:,1], theta, phi, psi=psi))
	else:
		dinj0 = dinj1 = [None]*n_ifo
	
	### iterate and plot
	for (name, i, di0, di1, d) in zip(network.detector_name_list, np.transpose(inj), dinj0, dinj1, np.transpose(data)):
		ax = plt.subplot(n_ax, 1, iax)

		ax.plot(freqs, i.real, color="b", label="$\mathbb{R}\{F*h\}$")
		if np.any(i.imag):
			ax.plot(freqs, i.imag, color="r", label="$\mathbb{I}\{F*h\}$")

		if di0!=None:
			ax.fill_between(freqs, di0, di1, color="b", alpha=0.25)
			if np.any(di0.imag):
				ax.fill_between(freqs, di0, di1, color="r", alpha=0.25)

		ax.plot(freqs, d.real, color="c", label="$\mathbb{R}{d_{\mathrm{%s}}}$"%name)
		if np.any(d.imag):
			ax.plot(freqs, d.imag, color="m", label="$\mathbb{I}{d_{\mathrm{%s}}}"%name)

		ax.set_ylabel("data from %s [%s]"%(name, units))

		axs.append( ax )
		iax += 1

	return (fig, axs)






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
	
