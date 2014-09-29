#!/usr/bin/python

usage = """pipeline.py [--options] 
a script meant to run the engine contained in posteriors.py and priors.py"""

import ConfigParser
import os
import numpy as np
import pickle

import utils
import detector_cache
import priors
import posteriors
import model_selection

from optparse import OptionParser

#=================================================
parser=OptionParser(usage=usage)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-c", "--config", default="./config.ini", type="string", help="config file")

parser.add_option("-g", "--gps", default=0, type="float", help="central time of this analysis")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

parser.add_option("", "--time", default=False, action="store_true")

opts, args = parser.parse_args()

if opts.tag:
	opts.tag = "_" + opts.tag

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

if opts.time:
	opts.verbose=True
	import time

#=================================================
### load config file
#=================================================
if opts.verbose: 
	print "loading config from ", opts.config
	if opts.time:
		to = time.time()

config = ConfigParser.SafeConfigParser()
config.read(opts.config)

max_proc=config.getint("general", "max_proc")
max_array_size=config.getint("general", "max_array_size")

n_pol=config.getint("general", "num_pol")

if opts.time:
	print "\t", time.time()-to

#=================================================
### fft parameters
#=================================================
if opts.verbose: 
	print "setting up fft paramters"
	if opts.time:
		to = time.time()

seglen = config.getfloat("fft","seglen")
df = 1.0/seglen
fs = config.getfloat("fft","fs")
freqs=np.arange(0, fs/2, df)
n_freqs = len(freqs)

if opts.time:
	print "\t", time.time()-to

#=================================================
### build network
#=================================================
if opts.verbose:
	print "building network"
	if opts.time:
		to = time.time()

ifos = eval(config.get("network","ifos"))
n_ifo = len(ifos)
network = utils.Network([detector_cache.detectors[ifo] for ifo in ifos], freqs=freqs, Np=n_pol)

if opts.time:
	print "\t", time.time()-to

#=================================================
### estimate PSD and load it into the network 
#=================================================
if opts.verbose:
	print "generating PSD's"
	if opts.time:
		to = time.time()

if config.has_option("psd_estimation","cache"):
	psd_cache = eval(config.get("psd_estimation", "cache"))
	for ifo in ifos:
		psd_filename = psd_cache[ifo]
		if opts.verbose:
			print "\treading PSD for %s from %s"%(ifo, psd_filename)
		psd_freqs, asd = np.transpose(np.loadtxt(psd_filename)) ### read in the psd from file
		network.detectors[ifo].set_psd(asd**2, freqs=psd_freqs) ### update the detector
else:
	raise StandardError, "don't know how to estimate psd's yet, so we must have a cache..."

if opts.time:
	print "\t", time.time()-to

freq_truth = np.ones_like(freqs, bool) ### put this here in case we want to throw out lines, etc

#=================================================
### build angprior
#=================================================
if opts.verbose:
	print "building angPrior"
	if opts.time:
		to = time.time()

nside_exp=config.getint("angPrior","nside_exp")
prior_type=config.get("angPrior","prior_type")
angPrior_kwargs={}
if prior_type=="uniform":
	pass
elif prior_type=="antenna_pattern":
	angPrior_kwargs["frequency"]=config.getfloat("angPrior","frequency")
	angPrior_kwargs["exp"]=config.getfloat("angPrior","exp")
else:
	raise ValueError, "prior_type=%s not understood"%prior_type

angprior = priors.angPrior(nside_exp, prior_type=prior_type, network=network, **angPrior_kwargs)

if opts.time:
	print "\t", time.time()-to

#=================================================
### build hprior
#=================================================
if opts.verbose:
	print "building hPrior"
	if opts.time:
		to = time.time()

pareto_a=config.getfloat("hPrior","pareto_a")
n_gaus_per_dec=config.getfloat("hPrior","n_gaus_per_dec")
log10_min=np.log10(config.getfloat("hPrior","min"))
log10_max=np.log10(config.getfloat("hPrior","max"))
n_gaus = max(1, int(round((log10_max-log10_min)*n_gaus_per_dec,0)))

variances=np.logspace(log10_min, log10_max, n_gaus)**2

pareto_means, pareto_covariance, pareto_amps = priors.pareto(pareto_a, n_freqs, n_pol, variances)

hprior = priors.hPrior(freqs, pareto_means, pareto_covariance, amplitudes=pareto_amps, n_gaus=n_gaus, n_pol=n_pol)

if opts.time:
	print "\t", time.time()-to

#=================================================
### build posterior_obj
#=================================================
if opts.verbose: 
	print "building posterior"
posterior = posteriors.Posterior(network=network, hPrior=hprior, angPrior=angprior, seglen=seglen)
posterior_num_proc=config.getint("posterior","num_proc")

### set things
if opts.verbose:
	print "set_theta_phi"
	if opts.time:
		to=time.time()
posterior.set_theta_phi()
if opts.time:
	print "\t", time.time()-to

if opts.verbose:
	print "set_AB_mp"
	if opts.time:
		to = time.time()
posterior.set_AB_mp(num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size)
if opts.time:
	print "\t", time.time()-to

if opts.verbose:
	print "set_P_mp"
	if opts.time:
		to = time.time()
posterior.set_P_mp(num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size)
if opts.time:
	print "\t", time.time()-to

#=================================================
### load noise
#=================================================
if config.has_option("noise","zero"):
	if opts.verbose:
		print "zeroing noise data"
	noise = np.zeros((n_freqs, n_ifo), complex)
elif config.has_option("noise","gaussian"):
        ### simulate gaussian noise
	if opts.verbose:
		print "drawing gaussian noise"
		if opts.time:
			to = time.time()
        noise = network.draw_noise()
	if opts.time:
		print "\t", time.time()-to
else:
        ### read noise from frames?
	if opts.verbose:
		print "\treading noise from file"
		if opts.time:
			to=time.time()

        raise StandardError, "don't know how to read noise from frames yet..."

	if opts.time:
		print "\t", time.time()-to

#=================================================
### load injection
#=================================================
if config.has_option("injection","zero"):
	if opts.verbose:
		print "zeroing injected data"
	inj = np.zeros((n_freqs, n_ifo), complex)

elif config.has_option("injection","dummy"):
	import injections

	fo = 200
	to = opts.gps + seglen/2.0
	phio = 0.0
	tau = 0.010
	hrss = 6e-23
	alpha = np.pi/2

	theta = np.pi/4
	phi = 3*np.pi/4
	psi = 0.0
	if opts.verbose:
		print """injecting SineGaussian with parameters
	fo = %f
	to = %f
	phio = %f
	tau = %f
	hrss = %fe-23
	alpha = %f
	
	theta = %f
	phi = %f
	psi = %f"""%(fo, to, phio, tau, hrss*1e23, alpha, theta, phi, psi)

		if opts.time:
			to = time.time()
	h = injections.sinegaussian_f(freqs, to=to, phio=phio, fo=fo, tau=tau, hrss=hrss, alpha=alpha)
	inj = injections.inject( network, h, theta, phi, psi)
else:
	raise StandardError, "don't know how to read injections yet..."

if config.has_option("injection","factor"):
	inj *= config.getfloat("injection","factor")

#=================================================
### set data, dataB
#=================================================
if opts.verbose:
	print "setting data"
	if opts.time:
		to=time.time()

posterior.set_data( noise+inj )
posterior.set_dataB()

if opts.time:
	print "\t", time.time()-to

#=================================================
### compute log_posterior_elements
#=================================================
if opts.verbose:
	print "computing log_posterior_elements"
	if opts.time:
		to = time.time()

log_posterior_elements, n_pol_eff = posterior.log_posterior_elements_mp(posterior.theta, posterior.phi, psi=0.0, invP_dataB=(posterior.invP, posterior.detinvP, posterior.dataB, posterior.dataB_conj), A_invA=None, diagnostic=False, num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size)

if opts.time:
	print "\t", time.time()-to

#=================================================
### model_selection
#=================================================
selection=config.get("model_selection","selection")

if opts.verbose:
	print "model selection with ", selection
	if opts.time:
		to = time.time()
if selection=="waterfill":
	model, log_bayes = model_selection.waterfill(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, freq_truth)
else:
	raise ValueError, "selection=%s not understood"%selection

log_posterior = posterior.log_posterior(posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, model, normalize=True)

if opts.time:
	print "\t", time.time()-to

#=================================================
### save output
#=================================================
pklname = "%s/pipeline%s.pkl"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "saving data to ", pklname
	if opts.time:
		to = time.time()

pkl_obj = open(pklname, "w")
pickle.dump(posterior, pkl_obj)
pickle.dump(log_posterior_elements, pkl_obj)
pickle.dump(model, pkl_obj)
pickle.dump(log_bayes, pkl_obj)
pickle.dump(log_posterior, pkl_obj)
pkl_obj.close()

if opts.time:
	print "\t", time.time()-to
