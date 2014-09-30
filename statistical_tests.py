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

parser.add_option("", "--zero-data", default=False, action="store_true")
parser.add_option("", "--zero-noise", default=False, action="store_true")

parser.add_option("", "--skip-plots", default=False, action="store_true")
parser.add_option("", "--skip-fits", default=False, action="store_true")

parser.add_option("", "--grid", default=False, action="store_true")

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

xmin = 1e-25
xmax = 1e-19
#npts = 1001

vmin = 10*xmin**2
vmax = 0.1*xmax**2

n_gaus_per_decade = 5 ### approximate scaling found empirically to make my decomposition work well
#n_gaus_per_decade = 2 ### approximate scaling found empirically to make my decomposition work well
n_gaus = int(round((np.log10(vmax**0.5)-np.log10(vmin**0.5))*n_gaus_per_decade, 0))
print "n_gaus :", n_gaus
variances = np.logspace(np.log10(vmin), np.log10(vmax), n_gaus)

#n_gaus = 1
#variances = 10*np.array([1e-24])**2 ### single-gaussian
### expect minimum hrss to be around 2.8e-23 for SNR~10 at design sensitivity
### we expect there to be roughly 5 bins, 2 polarizations
### this means we have a correction factor of sqrt(2*n_freqs*n_pol-1) ~ 3 in the maximum of the marginalized prior
### let's set the variance to be (1e-23)**2/10
#variance = np.array([2.83e-24 * 10 ])**2 / 20  ### single-gaussian1
#variance = np.array([5e-24 * 10 ])**2 / 10 ### single-gaussian2

'''
vmin = (2.83e-24 * 10)**2 / 20 ### high_confidence
vmax = (1e-20)**2

n_gaus_per_decade=5
n_gaus = int(round((np.log10(vmax)*0.5-np.log10(vmin)*0.5)*n_gaus_per_decade, 0))
variacnes = np.logspace(np.log10(vmin), np.log10(vmax), n_gaus)
'''

'''
n_gaus=1
fixed_hrss = 2e-23
variances = np.array([fixed_hrss])**2 / 20
'''

n_pol = 2
#n_pol = 1

#n_freqs = 51
#freqs = np.linspace(175, 225, n_freqs)
n_freqs = 101
freqs = np.linspace(100, 300, n_freqs)
df = freqs[1]-freqs[0]
seglen = df**-1

### set up stuff for angprior
nside_exp = 6
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
log_bayes_thr = 0
#log_bayes_thr = -1

n_bins = 7

min_n_bins = 1
max_n_bins = 15
dn_bins = 1

### plotting options
log_dynamic_range = 100

#=================================================
# INJECTIONS
#=================================================
import injections

log=False

to=0.0
phio=0.0
fo=200
tau=0.010
#tau=0.100
q=2**0.5*np.pi*fo*tau ### the sine-gaussian's q, for reference

min_snr = 15
min_hrss = 6e-24

waveform_func = injections.sinegaussian_f
waveform_args = {"to":to, "phio":phio, "fo":fo, "tau":tau, "alpha":np.pi/2}

if not opts.zero_data:
	print "injections.pareto_hrss"
	to=time.time()
	theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj = injections.pareto_hrss(network, a, waveform_func, waveform_args, min_hrss=min_hrss, min_snr=min_snr, num_inj=num_inj, max_trials=max_trials, verbose=opts.verbose)
	print "\t", time.time()-to

	#print "injections.min_snr"
	#to=time.time()
	#theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj = injections.min_snr(network, waveform_func, waveform_args, hrss=fixed_hrss, min_snr=min_snr, num_inj=num_inj, max_trials=max_trials, verbose=opts.verbose)
	#print "\t", time.time()-to

	print "plotting hrss distribution"
	to=time.time()
	figname = "%s/hrss%s.png"%(opts.output_dir, opts.tag)
	fig, ax, ax1 = vis.dual_hist(hrss_inj, n_per_bin=10, n_samples=1001, log=log, label=None, color="b")
	ax.set_xlabel("$h_{rss}$")
	ax.set_ylabel("count")
	if log:
		ax.set_yscale("log")
	ax1.set_ylabel("fraction of events")
	ax1.grid(opts.grid, which="both")
	ax.set_xlim(xmin=min_hrss)
	ax1.set_xlim(xmin=min_hrss)
	fig.savefig(figname)
	print "\t", time.time()-to
	vis.plt.close(fig)

	print "plotting snr distribution"
	to=time.time()
	figname = "%s/snr%s.png"%(opts.output_dir, opts.tag)
	fig, ax, ax1 = vis.dual_hist(np.sum(snrs_inj**2,axis=1)**0.5, n_per_bin=10, n_samples=1001, log=log, label=None, color="b")
	ax.set_xlabel("$\\rho_{network}$")
	ax.set_ylabel("count")
	if log:
		ax.set_yscale("log")
	ax1.set_ylabel("fraction of events")
	ax1.grid(opts.grid, which="both")
	ax.set_xlim(xmin=min_snr)
	ax1.set_xlim(xmin=min_snr)
	fig.savefig(figname)
	print "\t", time.time()-to
	vis.plt.close(fig)

	### save injected parameters
	import pickle
	inj_params_pklname = "%s/inj-params%s.pkl"%(opts.output_dir, opts.tag)
	file_obj = open(inj_params_pklname, "w")
	pickle.dump(waveform_args, file_obj)
	pickle.dump(theta_inj, file_obj)
	pickle.dump(phi_inj, file_obj)
	pickle.dump(psi_inj, file_obj)
	pickle.dump(hrss_inj, file_obj)
	pickle.dump(snrs_inj, file_obj)
	file_obj.close()

#=================================================
# PRIORS
#=================================================
import priors

print "hPrior"
to=time.time()
pareto_means, pareto_covariance, pareto_amps = priors.pareto(a, n_freqs, n_pol, variances)
hprior_obj = priors.hPrior(freqs, pareto_means, pareto_covariance, amplitudes=pareto_amps, n_gaus=n_gaus, n_pol=n_pol, byhand=True)
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
posterior_obj.set_A_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=True)
print "\t", time.time()-to

print "posterior.set_B_mp"
to=time.time()
posterior_obj.set_B_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
print "\t", time.time()-to

print "posterior.set_P_mp"
to=time.time()
posterior_obj.set_P_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=True)
print "\t", time.time()-to

#=================================================
# GENERATE POSTERIORS
#=================================================
import model_selection
import stats

fitsnames = []

if not opts.zero_data:
	p_value = np.empty((num_inj,),float)
	searched_area = np.empty((num_inj,),float)
	cos_dtheta = np.empty((num_inj,),float)

for inj_id in xrange(num_inj):

	print "inj_id = %d"%inj_id

	if opts.zero_data:
		data_inj = np.zeros((n_freqs, n_ifo), complex)
		inj_ang=None
		snr_net=0.0
	else:
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
		data_inj = injections.inject(network, waveform_func(freqs, hrss=hrss, **waveform_args), theta, phi, psi=psi)
		print "\t\t", time.time()-to

		injang = (theta, phi)

	if opts.zero_noise:
		noise = np.zeros((n_freqs, n_ifo), complex)
	else:
		print "\tdrawing noise"
		to=time.time()
		noise = network.draw_noise()
		print "\t\t", time.time()-to

	data = data_inj + noise

	print "\tset_data"
	to=time.time()
	posterior_obj.set_data(data)
	print "\t\t", time.time()-to

	print "\tset_dataB"
	to=time.time()
	posterior_obj.set_dataB()
	print "\t\t", time.time()-to

	print "\tlog_posterior_elements_mp"
	to=time.time()
	log_posterior_elements, n_pol_eff = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.detinvP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), diagnostic=False, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
	print "\t\t", time.time()-to

#	print "\tlog_bayes_cut_mp"
#	print "\tvariable_bandwidth(log_bayes_cut_mp)"
#	print "\tvariable_bandwidth_mp"
#	print "\tfixed_bandwidth_mp"
	print "\twaterfill"
	to=time.time()
#	model, lb = model_selection.log_bayes_cut_mp(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, joint_log_bayes=True)

#	lbc_model = model_selection.log_bayes_cut_mp(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, joint_log_bayes=False)
#	_max_n_bins = min(max_n_bins, np.sum(lbc_model+num_proc))
#	model, lb = model_selection.variable_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, lbc_model, min_n_bins=min_n_bins, max_n_bins=_max_n_bins, dn_bins=dn_bins)

#	model, lb = model_selection.variable_bandwidth_mp(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)

#	model, lb = model_selection.fixed_bandwidth_mp(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, n_bins=n_bins, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)

	model, lb = model_selection.waterfill(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth)
	print "\t\t", time.time()-to

	print ""
#	print "\tlogB_thr=",log_bayes_thr
#	print "\tn_lbc_bins=",np.sum(lbc_model)
	print "\tn_bins=",np.sum(model)
	print "\tlogBayes=",lb
	print "\thrss=",hrss
	print "\tsnr=",snr_net
	print ""

	print "\tlog_posterior"
	to=time.time()
	log_posterior = posterior_obj.log_posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, model, normalize=True)
	print "\t\t", time.time()-to

#	lbc_log_posteriors[inj_id,:] = lbc_log_posterior

	print "\tposterior by hand"
	to=time.time()
	posterior = np.exp(log_posterior) ### find posterior
	print "\t\t", time.time()-to
	
	#=================================================
	# PLOT POSTERIOR
	#=================================================
	if not opts.skip_plots:
		estang = stats.estang(posterior, nside=nside)
		print "\tplotting posterior"
		figname = "%s/inj=%d%s.png"%(opts.output_dir, inj_id, opts.tag)
		logfigname = "%s/inj=%d-log%s.png"%(opts.output_dir, inj_id, opts.tag)
		to=time.time()
		posterior_obj.plot(figname, posterior=posterior/pixarea, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(model),lb,snr_net), unit="prob/deg$^2$", inj=injang, est=estang)
		posterior_obj.plot(logfigname, posterior=log_posterior-np.log(pixarea), title="log(posterior)\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(model),lb,snr_net), unit="log(prob/deg$^2$)", inj=injang, est=estang)
		print "\t\t", time.time()-to

	#=================================================
	# SAVE POSTERIOR
	#=================================================
	if not opts.skip_fits:
		print "\twriting posterior to file"
		fits = "%s/inj-%d%s.fits"%(opts.output_dir, inj_id, opts.tag)
		to=time.time()
        	hp.write_map(fits, posterior)
		print "\t\t", time.time()-to

		fitsnames.append(fits)

	#=================================================
	# COMPUTE STATISTICS
	#=================================================
	if not opts.zero_data:
		print "\tp_value"
		to=time.time()
		p_value[inj_id] = stats.p_value(posterior, theta, phi, nside=nside)
		print "\t\t", p_value[inj_id]
		print "\t\t", time.time()-to

		print "\tsearched_area"
		to=time.time()
		searched_area[inj_id] = stats.searched_area(posterior, theta, phi, nside=nside, degrees=True)
		print "\t\t", searched_area[inj_id]
		print "\t\t", time.time()-to

		print "\tcos_dtheta"
		to=time.time()
		cos_dtheta[inj_id] = stats.est_cos_dtheta(posterior, theta, phi)
		print "\t\t", cos_dtheta[inj_id]
		print "\t\t", time.time()-to

	### write a tool that will compute moments of frequency given a model (and log_posterior_elements?)

#=================================================
# ENSEMBLE AVERAGES
#=================================================
if not opts.zero_data:
	print "plotting p_value"
	figname = "%s/p_value%s.png"%(opts.output_dir, opts.tag)
	to=time.time()
	fig, ax, ax1 = vis.dual_hist(p_value, n_per_bin=10, n_samples=1001, log=False, label=None, color="b")
	ax.set_xlabel("bayesian confidence level")
	ax.set_ylabel("probability density")
	ax1.set_ylabel("fraction of events")
	ax1.grid(opts.grid, which="both")
	ax1.plot([0,1],[0,1], 'k-') ### reference line for pp plots
	ax.set_xlim(xmin=0, xmax=1)
	ax1.set_xlim(xmin=0, xmax=1)
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
	ax1.grid(opts.grid, which="both")
	ax.set_xlim(xmin=pixarea, xmax=4*180**2/np.pi)
	ax1.set_xlim(xmin=pixarea, xmax=4*180**2/np.pi)
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
	ax1.grid(opts.grid, which="both")
	ax.set_xlim(xmin=1, xmax=-1)
	ax1.set_xlim(xmin=1, xmax=-1)
	fig.savefig(figname)
	print "\t", time.time()-to
	vis.plt.close(fig)


print """WRITE:
	coverage plots:
		localization confidence regions
		moments of frequencies
		moments of amplitudes
	else?

	RUN TESTS WITH OTHER MODEL SELECTION ALGORITHMS?
		=> check pp-plots, etc for each algorithm
		=> check localization performance for each algorithm
"""
