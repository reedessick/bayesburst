#!/usr/bin/python

usage = """a script that tests the basic functionality of our modules"""

#=================================================

import os
import sys
sys.path.append("/home/reed/LIGO/BAYESburst/bayesburst/")

import numpy as np
import healpy as hp

import utils
import detector_cache

import time
import pickle

from optparse import OptionParser

#=================================================

parser = OptionParser(usage=usage)

parser.add_option("", "--hPrior", default=False, action="store_true")
parser.add_option("", "--angPrior", default=False, action="store_true")
parser.add_option("", "--ap_angPrior", default=False, action="store_true")
parser.add_option("", "--posterior", default=False, action="store_true")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

opts, args = parser.parse_args()

if opts.posterior:
	opts.hPrior = opts.angPrior = True

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

#=================================================
# set up
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

n_freqs = 50
freqs = np.linspace(100, 200, n_freqs)
df = freqs[1]-freqs[0]
seglen = df**-1

### set up stuff for angprior
nside_exp = 5
prior_type="uniform"

### set up stuff for ap_angprior
network = utils.Network([detector_cache.LHO, detector_cache.LLO], freqs=freqs, Np=n_pol)
#network = utils.Network([detector_cache.LHO, detector_cache.LLO, detector_cache.Virgo], freqs=freqs, Np=n_pol)

n_ifo = len(network.detectors)

### set up stuff for posterior

n_proc = 5
max_proc = 1
max_array_size=100

freq_truth = np.ones_like(freqs, bool)

### define injection data
import injections

to=0.0
phio=0.0
fo=200
tau=0.010
hrss=2e-22

h = injections.sinegaussian_f(freqs, to, phio, fo, tau, hrss, alpha=np.pi/2)

theta_inj =   np.pi/4
phi_inj   = 3*np.pi/2

data = injections.inject(network, h, theta_inj, phi_inj, psi=0.0)
#data = np.random.random((n_freqs, n_ifo))

#=================================================
### filenames
tag = "_%d-%d-%d%s"%(n_freqs, n_gaus, nside_exp, opts.tag)

hfigname="%s/hprior%d%s.png"%(opts.output_dir, n_gaus_per_decade, tag)

posterior_figname = "%s/posterior%s.png"%(opts.output_dir, tag)
logposterior_figname="%s/log-posterior%s.png"%(opts.output_dir, tag)

posterior_pklname = "%s/posterior%s.pkl"%(opts.output_dir, tag)
posterior_filename = "%s/posterior%s.fits"%(opts.output_dir, tag)

angfigname = "%s/angprior%s.png"%(opts.output_dir, tag)

ap_angfigname = "%s/ap_angprior%s.png"%(opts.output_dir, tag)

diag_figname=opts.output_dir+"/%s-%s"+tag+".png"
logdiag_figname=opts.output_dir+"/log-%s-%s"+tag+".png"

#=================================================
# PRIORS
#=================================================
import priors

if opts.hPrior:
	print "pareto_amplitudes"
	to=time.time()
	pareto_amps = priors.pareto_amplitudes(a, variances)
	print "\t", time.time()-to

	print "pareto"
	to=time.time()
	pareto_means, pareto_covariance, pareto_amps = priors.pareto(a, n_freqs, n_pol, variances)
	print "\t", time.time()-to

	print "hPrior"
	to=time.time()
	hprior_obj = priors.hPrior(freqs, pareto_means, pareto_covariance, amplitudes=pareto_amps, n_gaus=n_gaus, n_pol=n_pol)
	print "\t", time.time()-to

	print "hPrior.plot"
	to=time.time()
	hprior_obj.plot(hfigname, xmin=xmin, xmax=xmax, npts=npts)#, ymin=1e0)
	print "\t", time.time()-to

#=================================================
if opts.angPrior:
	print "angPrior"
	to=time.time()
	angprior_obj = priors.angPrior(nside_exp, prior_type=prior_type)
	print "\t", time.time()-to

	print "angPrior.angprior"
	to=time.time()
	angprior = angprior_obj.angprior()
	print "\t", time.time()-to

	print "angPrior.__call__"
	to=time.time()
	p = angprior_obj(np.pi/2, np.pi)
	print "\t", time.time()-to

	print "angPrior.plot"
	to=time.time()
	angprior_obj.plot(angfigname, inj=(theta_inj, phi_inj))
	print "\t", time.time()-to

#=================================================
if opts.ap_angPrior:

	print "ap_angPrior"
	to=time.time()
	ap_angprior_obj = priors.angPrior(nside_exp, prior_type="antenna_pattern", frequency=150, exp=3.0, network=network)
	print "\t", time.time()-to

	print "angPrior.plot"
	to=time.time()
	ap_angprior_obj.plot(ap_angfigname, inj=(theta_inj, phi_inj))
	print "\t", time.time()-to


#=================================================
# POSTERIORS
#=================================================
import posteriors

if opts.posterior:

	print "posterior"
	to=time.time()
	posterior_obj = posteriors.Posterior()
	print "\t", time.time()-to

	print "set_network"
	to=time.time()
	posterior_obj.set_network(network)
	print "\t", time.time()-to

	print "set_hPrior"
	to=time.time()
	posterior_obj.set_hPrior(hprior_obj)
	print "\t", time.time()-to

	print "set_angPrior"
	to=time.time()
	posterior_obj.set_angPrior(angprior_obj)
	print "\t", time.time()-to

	print "set_seglen"
	to=time.time()
	posterior_obj.set_seglen(seglen)
	print "\t", time.time()-to

	print "set_data"
	to=time.time()
	posterior_obj.set_data(data)
	print "\t", time.time()-to

	print "posterior.__init__()"
	to=time.time()
	posterior_obj = posteriors.Posterior(network=network, hPrior=hprior_obj, angPrior=angprior_obj, seglen=seglen, data=data)
	print "\t", time.time()-to

	print "posterior.set_theta_phi()"
	to=time.time()
	posterior_obj.set_theta_phi()
	print "\t", time.time()-to

	print "posterior.set_A"	
	to=time.time()
	posterior_obj.set_A()
	print "\t", time.time()-to

	print "posterior.set_B"
	to=time.time()
	posterior_obj.set_B()
	print "\t", time.time()-to

	print "posterior.set_P"
	to=time.time()
	posterior_obj.set_P()
	print "\t", time.time()-to

	print "posterior.set_dataB"
	to=time.time()
	posterior_obj.set_dataB()
	print "\t", time.time()-to

#	print "pickling posterior into ", posterior_pklname
#	to=time.time()
#	file_obj = open(posterior_pklname, "w")
#	pickle.dump(posterior_obj, file_obj)
#	file_obj.close()
#	print "\t", time.time()-to

	print "posterior.n_pol_eff()"
	to=time.time()
	posterior_obj.n_pol_eff(posterior_obj.theta, posterior_obj.phi)
	print "\t", time.time()-to

	print "posterior.mle_strain"
	to=time.time()
	mle_strain = posterior_obj.mle_strain(posterior_obj.theta, posterior_obj.phi, psi=0.0, n_pol_eff=None, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
	print "\t", time.time()-to

	print "posterior.log_posterior_elements"
	to=time.time()
	log_posterior_elements, n_pol_eff = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=False)
	print "\t", time.time()-to

#	print "posterior.log_posterior_elements_mp"
#	to=time.time()
#	mp_log_posterior_elements, mp_n_pol_eff = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), diagnostic=False, n_proc=n_proc, max_proc=max_proc, max_array_size=max_array_size)
#	print "\t", time.time()-to

#	if np.any(log_posterior_elements!=mp_log_posterior_elements):
#		raise StandardError, "conflict between log_posterior_elements and mp_log_posterior_elements"
#	if np.any(n_pol_eff!=mp_n_pol_eff):
#		raise StandardError, "conflict between n_pol_eff and mp_n_pol_eff"

	print "posterior.log_posterior_elements(diagnostic=True)"
	to=time.time()
	log_posterior_elements_diag, n_pol_eff_diag, (mle, cts, det) = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=True)
	print "\t", time.time()-to

	if np.any(log_posterior_elements_diag!=log_posterior_elements):
		raise StandardError, "conflict between log_posterior_elements and log_posterior_elements_diag"
	if np.any(n_pol_eff_diag!=n_pol_eff):
		raise StandardError, "conflict between n_pol_eff and n_pol_eff_diag"

#	print "posterior.log_posterior_elements_mp(diagnostic=True)"
#	to=time.time()
#	mp_log_posterior_elements, mp_n_pol_eff, (mp_mle, mp_cts, mp_det) = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), n_proc=n_proc, max_proc=max_proc, max_array_size=max_array_size, diagnostic=True)
#	print "\t", time.time()-to

#	if np.any(log_posterior_elements!=mp_log_posterior_elements):
#		raise StandardError, "conflict between log_posterior_elements and mp_log_posterior_elements"
#	if np.any(n_pol_eff!=mp_n_pol_eff):
#		raise StandardError, "conflict between n_pol_eff and mp_n_pol_eff"
#	if np.any(mle!=mp_mle):
#		raise StandardError, "conflict between mle and mp_mle"
#	if np.any(cts!=mp_cts):
#		raise StandardError, "conflict between cts and mp_cts"
#	if np.any(det!=mp_det):
#		raise StandardError, "conflict between det and mp_det"

	print "posterior.log_posterior"
	to=time.time()
	freq_true = np.ones((n_freqs,), bool)
	log_posterior = posterior_obj.log_posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)
	print "\t", time.time()-to

	print "posterior.posterior"
	to=time.time()
	posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)
	print "\t", time.time()-to

	print "posterior.log_bayes"
	to=time.time()
	log_bayes = posterior_obj.log_bayes(log_posterior)
	print "\t", time.time()-to

	print "posterior.bayes"
	to=time.time()
	bayes = posterior_obj.bayes(log_posterior)
	print "\t", time.time()-to

#	print "posterior.__call__"
#	to=time.time()
#	posterior = posterior_obj()
#	print "\t", time.time()-to

	print "posterior.plot"
	to=time.time()
	posterior_obj.plot(posterior_figname, posterior=posterior, title="posterior", unit="prob/pix", inj=(theta_inj, phi_inj), est=None)
	posterior_obj.plot(logposterior_figname, posterior=np.log10(posterior), title="log10( posterior )", unit="log10(prob/pix)", inj=(theta_inj, phi_inj), est=None)
	print "\t", time.time()-to

	print "diagnostic plots"
	### use log_poserterior_elements computed above
	for g in xrange(n_gaus):
		s = variances[g]**0.5
		d = int(np.log10(s))-1
		s = "%.3fe%d"%(s * 10**-d, d)

		### mle
		print "mle %s"%s
		to=time.time()
		_mle = np.sum(mle[:,g,:], axis=1) * df ### sum over freqs
		posterior_obj.plot(diag_figname%("mle",g), posterior=np.exp(_mle), title="mle %s"%s, inj=(theta_inj, phi_inj), est=None)
		posterior_obj.plot(logdiag_figname%("mle",g), posterior=_mle, title="log10( mle %s )"%s, inj=(theta_inj, phi_inj), est=None)
		print "\t", time.time()-to

		### cts
		print "cts %s"%s
		to=time.time()
		_cts = np.sum(cts[:,g,:], axis=1) * df
		posterior_obj.plot(diag_figname%("cts",g), posterior=np.exp(_cts), title="cts %s"%s, inj=(theta_inj, phi_inj), est=None)
                posterior_obj.plot(logdiag_figname%("cts",g), posterior=_cts, title="log10( cts %s )"%s, inj=(theta_inj, phi_inj), est=None)
                print "\t", time.time()-to

		### det
		print "det %s"%s
		to=time.time()
		_det = np.sum(det[:,g,:], axis=1) * df
		posterior_obj.plot(diag_figname%("det",g), posterior=np.exp(_det), title="det %s"%s, inj=(theta_inj, phi_inj), est=None)
                posterior_obj.plot(logdiag_figname%("det",g), posterior=_det, title="log10( det %s )"%s, inj=(theta_inj, phi_inj), est=None)
                print "\t", time.time()-to

		### mle+cts+det
		print "mle+cts+det %s"%s
		to=time.time()
		posterior_obj.plot(diag_figname%("mle*cts*det",g), posterior=np.exp(_mle+_cts+_det), title="mle*cts*det %s"%s, inj=(theta_inj, phi_inj), est=None)

		posterior_obj.plot(logdiag_figname%("mle*cts*det",g), posterior=_mle+_cts+_det, title="log10( mle*cts*det )", inj=(theta_inj, phi_inj), est=None)
                print "\t", time.time()-to

	print "writing posterior to file"
	hp.write_map(posterior_filename, posterior)

	print """WRITE TESTS FOR
	model_select


	actually check the sanity of these results, not just the fact that they don't throw errors
		compare F*mle_strain with data in each detector
		compare mle, cts, det terms accross n_pix, n_freq, n_gaus
		compare mle + cts + det with log_posterior_elements
		look at posterior (plot)

	test all *_mp algorithms against the single CPU equivalents
	test diagnostic elements vs log_posterior_elements
	"""
