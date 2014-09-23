#!/usr/bin/python

usage = """ a script that generates statistical tests of the algorithm, such as coverage plots """

#=================================================

import os
import sys
sys.path.append("/home/reed/LIGO/BAYESburst/bayesburst/")

import numpy as np
import healpy as hp

import utils
import detector_cache

import visualizations as vis

import time
import pickle

from optparse import OptionParser

#=================================================

parser = OptionParser(usage=usage)

parser.add_option("", "-v", "--verbose", default=False, action="store_true")

parser.add_option("-n", "--num-inj", default=100, type="int")
parser.add_option("-N", "--max-trials", default=np.infty, type="float")

parser.add_option("", "--num-proc", default=1, type="int")
parser.add_option("", "--max-proc", default=1, type="int")
parser.add_option("", "--max-array-size", default=100, type="int")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

opts, args = parser.parse_args()

if opts.tag:
        opts.tag = "_"+opts.tag

if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)

num_inj = opts.num_inj
max_trials = opts.max_trials

num_proc = opts.num_proc
max_proc = opts.max_proc
max_array_size = opts.max_array_size

#=================================================
### set up stuff for hPrior
a = 4
xmin = 1e-24
xmax = 1e-20
npts = 1001

vmin = 10*xmin**2
vmax = 0.1*xmax**2

n_gaus_per_decade = 5 ### approximate scaling found empirically to make my decomposition work well
n_gaus = int(round((np.log10(vmax**0.5)-np.log10(vmin**0.5))*n_gaus_per_decade, 0))
print "n_gaus :", n_gaus
variances = np.logspace(np.log10(vmin), np.log10(vmax), n_gaus)

n_pol = 2
#n_pol = 1

n_freqs = 101
freqs = np.linspace(100, 300, n_freqs)
df = freqs[1]-freqs[0]
seglen = df**-1

### set up stuff for angprior
nside_exp = 5
nside = 2**nside_exp
npix = hp.nside2npix(nside)
pixarea = hp.nside2pixarea(nside, degrees=True)
prior_type="uniform"

### set up stuff for ap_angprior
network = utils.Network([detector_cache.LHO, detector_cache.LLO], freqs=freqs, Np=n_pol)
#network = utils.Network([detector_cache.LHO, detector_cache.LLO, detector_cache.Virgo], freqs=freqs, Np=n_pol)

n_ifo = len(network.detectors)

### set up stuff for posterior
freq_truth = np.ones_like(freqs, bool)

### set up stuff for model selection
log_bayes_thr = 10

### plotting options
log_dynamic_range = 100

#=================================================
# INJECTIONS
#=================================================
import injections

to=0.0
phio=0.0
fo=200
tau=0.010
q=2**0.5*np.pi*fo*tau ### the sine-gaussian's q, for reference

min_snr = 10
min_hrss = 5e-23

waveform_func = injections.sinegaussian_f
waveform_args = {"to":to, "phio":phio, "fo":fo, "tau":tau, "alpha":np.pi/2}

print "injections.pareto_hrss"
to=time.time()
theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj = injections.pareto_hrss(network, a, waveform_func, waveform_args, min_hrss=min_hrss, min_snr=min_snr, num_inj=num_inj, max_trials=max_trials, verbose=opts.verbose)
print "\t", time.time()-to

print "plotting hrss distribution"
to=time.time()
figname = "%s/hrss%s.png"%(opts.output_dir, opts.tag)
fig, ax, ax1 = vis.dual_hist(hrss_inj, n_per_bin=10, n_samples=1001, log=True, label=None, color="b")
ax.set_xlabel("$h_{rss}$")
ax.set_ylabel("count")
ax.set_yscale("log")
ax1.set_ylabel("fraction of events")
ax.grid(True, which="both")
fig.savefig(figname)
print "\t", time.time()-to
vis.plt.close(fig)

print "plotting snr distribution"
to=time.time()
figname = "%s/snr%s.png"%(opts.output_dir, opts.tag)
fig, ax, ax1 = vis.dual_hist(np.sum(snrs_inj**2,axis=1)**0.5, n_per_bin=10, n_samples=1001, log=True, label=None, color="b")
ax.set_xlabel("$\\rho_{network}$")
ax.set_ylabel("count")
ax.set_yscale("log")
ax1.set_ylabel("fraction of events")
ax.grid(True, which="both")
fig.savefig(figname)
print "\t", time.time()-to
vis.plt.close(fig)

#=================================================
# PRIORS
#=================================================
import priors

print "hPrior"
to=time.time()
pareto_means, pareto_covariance, pareto_amps = priors.pareto(a, n_freqs, n_pol, variances)
hprior_obj = priors.hPrior(freqs, pareto_means, pareto_covariance, amplitudes=pareto_amps, n_gaus=n_gaus, n_pol=n_pol)
print "\t", time.time()-to

print "angPrior"
to=time.time()
angprior_obj = priors.angPrior(nside_exp, prior_type=prior_type)
print "\t", time.time()-to

#=================================================
# POSTERIORS
#=================================================
import posteriors

print "posterior.__init__()"
to=time.time()
posterior_obj = posteriors.Posterior(network=network, hPrior=hprior_obj, angPrior=angprior_obj, seglen=seglen, data=None)
print "\t", time.time()-to

print "posterior.set_theta_phi()"
to=time.time()
posterior_obj.set_theta_phi()
print "\t", time.time()-to

print "posterior.set_A_mp"
to=time.time()
posterior_obj.set_A_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
print "\t", time.time()-to

print "posterior.set_B_mp"
to=time.time()
posterior_obj.set_B_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
print "\t", time.time()-to

print "posterior.set_P_mp"
to=time.time()
posterior_obj.set_P_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
print "\t", time.time()-to

#=================================================
# GENERATE POSTERIORS
#=================================================
import model_selection
import stats

### place holders for resulting data
lbc_log_posteriors = np.empty((num_inj,npix),float)
lbc_models = np.zeros((num_inj,n_freqs),bool)
lbc_log_bayes = np.empty((num_inj,),float)

p_value = np.empty((num_inj,),float)
searched_area = np.empty((num_inj,),float)
cos_dtheta = np.empty((num_inj,),float)

for inj_id in xrange(num_inj):

	print "inj_id = %d"%inj_id

	print "\tpulling out injection parameters"
	to=time.time()
	### antenna pattern jazz
	theta = theta_inj[inj_id]
	phi = phi_inj[inj_id]
	psi = psi_inj[inj_id]

	### singal amplitude
	hrss = hrss_inj[inj_id]
	snrs = snrs_inj[inj_id]
	snr_net = np.sum(snrs**2)**0.5
	print "\t\t", time.time()-to	

	### generate data stream
	print "\tgenerating data"
	to=time.time()
	data = injections.inject(network, waveform_func(freqs, hrss=hrss, **waveform_args), theta, phi, psi=psi)
	print "\t\t", time.time()-to

	print "\tset_data"
	to=time.time()
	posterior_obj.set_data(data)
	print "\t\t", time.time()-to

	print "\tset_dataB"
	to=time.time()
	posterior_obj.set_dataB()
	print "\t\t", time.time()-to

	print "\tlog_posterior_elements"
	to=time.time()
	log_posterior_elements, n_pol_eff = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=False)
	print "\t\t", time.time()-to

	print "\tlog_bayes_cut"
	to=time.time()
	lbc_models[inj_id,:], lbc_log_bayes[inj_id] = model_selection.log_bayes_cut_mp(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, num_proc=num_proc, max_proc=max_proc, joint_log_bayes=True)
	print "\t\t", time.time()-to

	print "\tlog_posterior"
	to=time.time()
	lbc_log_posteriors[inj_id,:] = posterior_obj.log_posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, lbc_models[inj_id,:], normalize=True)
	print "\t\t", time.time()-to

	print "\tposterior by hand"
	to=time.time()
	posterior = np.exp(lbc_log_posteriors[inj_id,:]) ### find posterior
	print "\t\t", time.time()-to
	
	#=================================================
	# PLOT POSTERIOR
	#=================================================
	print "\tplotting posterior"
	figname = "%s/inj=%d%s.png"%(opts.output_dir, inj_id, opts.tag)
	to=time.time()
	posterior_obj.plot(figname, posterior=posterior/pixarea, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(lbc_models[inj_id,:]),lbc_log_bayes[inj_id],snr_net), unit="prob/deg$^2$", inj=(theta, phi), est=stats.estang(posterior, nside=nside))
	print "\t\t", time.time()-to

	#=================================================
	# SAVE POSTERIOR
	#=================================================
	print "\twriting posterior to file"
	fits = "%s/inj-%d%s.fits"%(opts.output_dir, inj_id, opts.tag)
	to=time.time()
        hp.write_map(fits, posterior)
	print "\t\t", time.time()-to

	#=================================================
	# COMPUTE STATISTICS
	#=================================================
	print "\tp_value"
	to=time.time()
	p_value[inj_id] = stats.p_value(posterior, theta, phi, nside=nside)
	print "\t\t", time.time()-to

	print "\tsearched_area"
	to=time.time()
	searched_area[inj_id] = stats.searched_area(posterior, theta, phi, nside=nside, degrees=True)
	print "\t\t", time.time()-to

	print "\tcos_dtheta"
	to=time.time()
	cos_dtheta[inj_id] = stats.est_cos_dtheta(posterior, theta, phi)
	print "\t\t", time.time()-to

	### write a tool that will compute moments of frequency given a model (and log_posterior_elements?)

#=================================================
# ENSEMBLE AVERAGES
#=================================================
print "plotting p_value"
figname = "%s/p_value%s.png"%(opts.output_dir, opts.tag)
to=time.time()
fig, ax, ax1 = vis.dual_hist(p_value, n_per_bin=10, n_samples=1001, log=False, label=None, color="b")
ax.set_xlabel("bayesian confidence level")
ax.set_ylabel("probability density")
ax1.set_ylabel("fraction of events")
ax.grid(True, which="both")
fig.savefig(figname)
print "\t", time.time()-to
vis.plt.close(fig)

print "plotting searced_area"
figname = "%s/searched_area%s.png"%(opts.output_dir, opts.tag)
to=time.time()
fig, ax, ax1 = vis.dual_hist(searched_area, n_per_bin=10, n_samples=1001, log=True, label=None, color="b")
ax.set_xlabel("searched area [deg$^2$]")
ax.set_ylabel("probability density")
ax1.set_ylabel("fraction of events")
ax.grid(True, which="both")
fig.savefig(figname)
print "\t", time.time()-to
vis.plt.close(fig)

print "plotting cos_dtheta"
figname = "%s/cos_dtheta%s.png"%(opts.output_dir, opts.tag)
to=time.time()
fig, ax, ax1 = vis.dual_hist(cos_dtheta, n_per_bin=10, n_samples=1001, log=False, label=None, color="b")
ax.set_xlabel("$\cos(\delta\\theta)$")
ax.set_ylabel("probability density")
ax1.set_ylabel("fraction of events")
ax.grid(True, which="both")
fig.savefig(figname)
print "\t", time.time()-to
vis.plt.close(fig)


print """WRITE:
	coverage plots:
		localization confidence regions
		moments of frequencies
		moments of amplitudes
	else?
"""
