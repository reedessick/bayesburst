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
parser.add_option("", "--model-selection", default=False, action="store_true")

parser.add_option("", "--num-proc", default=2, type="int")
parser.add_option("", "--max-proc", default=2, type="int")
parser.add_option("", "--max-array-size", default=100, type="int")

parser.add_option("", "--pkl", default=False, action="store_true")
parser.add_option("", "--check", default=False, action="store_true")
parser.add_option("", "--skip-mp", default=False, action="store_true")
parser.add_option("", "--skip-plots", default=False, action="store_true")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

opts, args = parser.parse_args()

if opts.tag:
	opts.tag = "_"+opts.tag

if opts.model_selection:
	opts.posterior = True

if opts.posterior:
	opts.hPrior = opts.angPrior = True

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

num_proc = opts.num_proc
max_proc = opts.max_proc
max_array_size = opts.max_array_size

eps = 1e-05 ### precision for "floating point errors"? Not sure where the errors are coming into AB_A and AB_invA relative to A, invA
            ### important for comparing output from different methods

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

n_freqs = 100
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

freq_truth = np.ones_like(freqs, bool)

### set up stuff for model selection
n_bins = 75

min_n_bins = 25
max_n_bins = 100
dn_bins = 5

log_bayes_thr = 0

### plotting options
log_dynamic_range = 100

### define injection data
import injections

to=0.0
phio=0.0
fo=200
tau=0.010
#hrss=2e-22 #network SNR ~50
hrss=4e-23 #network SNR ~10

h = injections.sinegaussian_f(freqs, to, phio, fo, tau, hrss, alpha=np.pi/2)

theta_inj =   np.pi/4
phi_inj   = 3*np.pi/2

data_inj = injections.inject(network, h, theta_inj, phi_inj, psi=0.0)
snrs_inj = network.snrs(data_inj) ### compute individual SNRs for detectors
snr_net_inj = np.sum(snrs_inj**2)**0.5 ### network SNR

data = data_inj
#data = np.random.random((n_freqs, n_ifo))

#=================================================
### filenames
tag = "_%d-%d-%d%s"%(n_freqs, n_gaus, nside_exp, opts.tag)

hfigname="%s/hprior%d%s.png"%(opts.output_dir, n_gaus_per_decade, tag)

posterior_figname = "%s/posterior%s.png"%(opts.output_dir, tag)
logposterior_figname="%s/log-posterior%s.png"%(opts.output_dir, tag)

fb_posterior_figname="%s/posterior-fixed_bandwidth%s.png"%(opts.output_dir, tag)
fb_logposterior_figname="%s/log-posterior-fixed_bandwidth%s.png"%(opts.output_dir, tag)

vb_posterior_figname="%s/posterior-variable_bandwidth%s.png"%(opts.output_dir, tag)
vb_logposterior_figname="%s/log-posterior-variable_bandwidth%s.png"%(opts.output_dir, tag)

lbc_posterior_figname="%s/posterior-log_bayes_cut%s.png"%(opts.output_dir, tag)
lbc_logposterior_figname="%s/log-posterior-log_bayes_cut%s.png"%(opts.output_dir, tag)

posterior_pklname = "%s/posterior%s.pkl"%(opts.output_dir, tag)
posterior_filename = "%s/posterior%s.fits"%(opts.output_dir, tag)

fb_posterior_filename = "%s/posterior-fixed_bandwidth%s.fits"%(opts.output_dir, tag)
vb_posterior_filename = "%s/posterior-variable_bandwidth%s.fits"%(opts.output_dir, tag)
lbc_posterior_filename = "%s/posterior-log_bayes_cut%s.fits"%(opts.output_dir, tag)

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

	if not opts.skip_plots:
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

	if not opts.skip_plots:
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

	if not opts.skip_plots:
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

	#=========================================
	# setting basic data stored within the object from which we compute stats
	#=========================================
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

	#=========================================
	# setting computed data stored within the object from which we compute stats
	#=========================================
	print "posterior.set_theta_phi()"
	to=time.time()
	posterior_obj.set_theta_phi()
	print "\t", time.time()-to

	print "posterior.set_A"	
	to=time.time()
	posterior_obj.set_A()
	A = posterior_obj.A
	invA = posterior_obj.invA
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.set_A_mp"
		to=time.time()
		posterior_obj.set_A_mp(num_proc=num_proc, max_proc=max_proc)
		A_mp = posterior_obj.A
		invA_mp = posterior_obj.invA
		print "\t", time.time()-to

		if opts.check:
			if np.any(A!=A_mp):
				raise StandardError, "A!=A_mp"
			else:
				print "\tA==A_mp"
			if np.any(invA!=invA_mp):
				raise StandardError, "invA!=invA_mp"
			else:
				print "\tinvA==invA_mp"

	print "posterior.set_B"
	to=time.time()
	posterior_obj.set_B()
	B = posterior_obj.B
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.set_B_mp"
		to=time.time()
		posterior_obj.set_B_mp(num_proc=num_proc, max_proc=max_proc)
		B_mp = posterior_obj.B
		print "\t", time.time()-to

		if opts.check:
			if np.any(B!=B_mp):
				raise StandardError, "B!=B_mp"
			else:
				print "\tB==B_mp"

	print "posterior.set_AB"
	to=time.time()
	posterior_obj.set_AB()
	AB_A = posterior_obj.A
	AB_invA = posterior_obj.invA
	AB_B = posterior_obj.B
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.set_AB_mp"
		to=time.time()
		posterior_obj.set_AB_mp(num_proc=num_proc, max_proc=max_proc)
		AB_A_mp = posterior_obj.A
		AB_invA_mp = posterior_obj.invA
		AB_B_mp = posterior_obj.B
		print "\t", time.time()-to

		if opts.check:
			if np.any(AB_A!=AB_A_mp):
				raise StandardError, "AB_A!=AB_A_mp"
			else:
				print "\tAB_A==AB_A_mp"
			if np.any(AB_invA!=AB_invA_mp):
				raise StandardError, "AB_invA!=AB_invA_mp"
			else:
				print "\tAB_invA==AB_invA_mp"
			if np.any(AB_B!=AB_B_mp):
				raise StandardError, "AB_B!=AB_B_mp"
			else:
				print "\tAB_B==AB_B_mp"

	if opts.check:
		if np.any(np.abs(AB_A-A) > eps*np.abs(AB_A+A)):
			raise StandardError, "AB_A!=A"
		elif np.any(AB_A!=A):
			print "AB_A~A"
		else:
			print "\tAB_A==A"
		if np.any(np.abs(AB_invA-invA) > eps*np.abs(AB_invA+invA)):
			raise StandardError, "AB_invA!=invA"
		elif np.any(AB_invA!=invA):
			print "AB_invA~invA"
		else:
			print "\tAB_invA==invA"
		if np.any(AB_B!=B):
			raise StandardError, "AB_B!=B"
		else:
			print "\tAB_B==B"

	print "posterior.set_P"
	to=time.time()
	posterior_obj.set_P()
	P, invP = posterior_obj.P, posterior_obj.invP
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.set_P_mp"
		to=time.time()
		posterior_obj.set_P_mp(num_proc=num_proc, max_proc=max_proc)
		P_mp, invP_mp = posterior_obj.P, posterior_obj.invP
		print "\t", time.time()-to

		if opts.check:
			if np.any(P!=P_mp):
				raise StandardError, "P!=P_mp"
			else:
				print "P==P_mp"
			if np.any(invP!=invP_mp):
				raise StandardError, "invP!=invP_mp"
			else:
				print "invP==invP_mp"

	print "posterior.set_dataB"
	to=time.time()
	posterior_obj.set_dataB()
	dataB = posterior_obj.dataB
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.set_dataB_mp"
		to=time.time()
		posterior_obj.set_dataB_mp(num_proc=num_proc, max_proc=max_proc)
		dataB_mp = posterior_obj.dataB
		print "\t", time.time()-to

		if opts.check:
			if np.any(dataB!=dataB_mp): 
				raise StandardError, "dataB!=dataB_mp"
			else:
				print "\tdataB==dataB_mp"

	#================================================
	# pickling to save
	#================================================
	if opts.pkl:
		print "pickling posterior into ", posterior_pklname
		to=time.time()
		file_obj = open(posterior_pklname, "w")
		pickle.dump(posterior_obj, file_obj)
		file_obj.close()
		print "\t", time.time()-to

	#================================================
	# analysis routines that do no store data within the object
	#================================================
	print "posterior.n_pol_eff()"
	to=time.time()
	posterior_obj.n_pol_eff(posterior_obj.theta, posterior_obj.phi)
	print "\t", time.time()-to

	print "posterior.mle_strain"
	to=time.time()
	mle_strain = posterior_obj.mle_strain(posterior_obj.theta, posterior_obj.phi, psi=0.0, n_pol_eff=None, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.mle_strain_mp"
		to=time.time()
		mle_strain_mp = posterior_obj.mle_strain_mp(posterior_obj.theta, posterior_obj.phi, 0.0, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, n_pol_eff=None, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
		print "\t", time.time()-to

		if opts.check:
			if np.any(mle_strain!=mle_strain_mp):
				raise StandardError, "mle_strain!=mle_strain_mp"
			else:
				print "\tmle_strain==mle_strain_mp"

	print "posterior.log_posterior_elements"
	to=time.time()
	log_posterior_elements, n_pol_eff = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=False)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.log_posterior_elements_mp"
		to=time.time()
		mp_log_posterior_elements, mp_n_pol_eff = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), diagnostic=False, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to

		if opts.check:
			if np.any(np.abs(log_posterior_elements-mp_log_posterior_elements) > eps*np.abs(log_posterior_elements+mp_log_posterior_elements)):
				raise StandardError, "conflict between log_posterior_elements and mp_log_posterior_elements"
			else:
				print "log_posterior_elements==mp_log_posterior_elements"
			if np.any(n_pol_eff!=mp_n_pol_eff):
				raise StandardError, "conflict between n_pol_eff and mp_n_pol_eff"
			else:
				print "n_pol_eff==mp_n_pol_eff"

	print "posterior.log_posterior_elements(diagnostic=True)"
	to=time.time()
	log_posterior_elements_diag, n_pol_eff_diag, (mle, cts, det) = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=True)
	print "\t", time.time()-to

	if opts.check:
		if np.any(log_posterior_elements_diag!=log_posterior_elements):
			raise StandardError, "conflict between log_posterior_elements and log_posterior_elements_diag"
		else:
			print "\tlog_posterior_elements_diag==log_posterior_elements"
		if np.any(n_pol_eff_diag!=n_pol_eff):
			raise StandardError, "conflict between n_pol_eff and n_pol_eff_diag"
		else:
			print "\tn_pol_eff_diag==n_pol_eff"

	if not opts.skip_mp:
		print "posterior.log_posterior_elements_mp(diagnostic=True)"
		to=time.time()
		mp_log_posterior_elements, mp_n_pol_eff, (mp_mle, mp_cts, mp_det) = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, diagnostic=True)
		print "\t", time.time()-to

		if opts.check:
			if np.any(log_posterior_elements!=mp_log_posterior_elements):
				raise StandardError, "conflict between log_posterior_elements and mp_log_posterior_elements"
			if np.any(n_pol_eff!=mp_n_pol_eff):
				raise StandardError, "conflict between n_pol_eff and mp_n_pol_eff"
			if np.any(mle!=mp_mle):
				raise StandardError, "conflict between mle and mp_mle"
			if np.any(cts!=mp_cts):
				raise StandardError, "conflict between cts and mp_cts"
			if np.any(det!=mp_det):
				raise StandardError, "conflict between det and mp_det"

	if opts.check:
		if np.any(log_posterior_elements!=mle+cts+det):
			raise StandardError, "log_posterior_elements!=mle+cts+det"
		else:
			print "log_posterior_elements==mle+cts+det"


	print "posterior.log_posterior"
	to=time.time()
	log_posterior = posterior_obj.log_posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.log_posterior_mp"
		to=time.time()
		log_posterior_mp = posterior_obj.log_posterior_mp(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True, num_proc=num_proc, max_proc=max_proc)
		print "\t", time.time()-to

		if opts.check:
			if np.any(log_posterior!=log_posterior_mp):
				raise StandardError, "log_posterior!=log_posterior_mp"

	print "posterior.posterior"
	to=time.time()
	posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.posterior_mp"
		to=time.time()
		posterior_mp = posterior_obj.posterior_mp(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True, num_proc=num_proc, max_proc=max_proc)
		print "\t", time.time()-to

		if opts.check:
			if np.any(posterior!=posterior_mp):
				raise StandardError, "posterior!=posterior_mp"

			if np.any(posterior!=np.exp(log_posterior)):
				raise StandardError, "posterior!=np.exp(log_posterior)"

	print "posterior.log_bayes"
	to=time.time()
	log_bayes = posterior_obj.log_bayes(log_posterior)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.log_bayes_mp"
		to=time.time()
		log_bayes_mp = posterior_obj.log_bayes_mp(log_posterior, num_proc=num_proc, max_proc=max_proc)
		print "\t", time.time()-to

		if opts.check:
			if np.any(np.abs(log_bayes-log_bayes_mp) > eps*np.abs(log_bayes+log_bayes_mp)):
				raise StandardError, "log_bayes!=log_bayes_mp"
			elif np.any(log_bayes!=log_bayes_mp):
				print "log_bayes~log_bayes_mp"
			else:
				print "log_bayes==log_bayes_mp"

	print "posterior.bayes"
	to=time.time()
	bayes = posterior_obj.bayes(log_posterior)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.bayes_mp"
		to=time.time()
		bayes_mp = posterior_obj.bayes_mp(log_posterior, num_proc=num_proc, max_proc=max_proc)
		print "\t", time.time()-to

		if opts.check:
			if np.any(bayes!=bayes_mp):
				raise StandardError, "bayes!=bayes_mp"

			if np.any(bayes!=np.exp(log_bayes)):
				raise StandardError, "bayes!=np.exp(log_bayes)"

	print "posterior.__call__"
	to=time.time()
	posterior_call = posterior_obj()
	print "\t", time.time()-to

	if opts.check:
		if np.any(posterior_call!=posterior):
			raise StandardError, "posterior_call!=posterior"
		else:
			print "\tposterior_call==posterior"

        print "writing posterior to file"
        hp.write_map(posterior_filename, posterior)

	#=========================================
	# plotting
	#=========================================
	if not opts.skip_plots:
		print "posterior.plot"
		to=time.time()
		posterior_obj.plot(posterior_figname, posterior=posterior, title="posterior\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(log_bayes, snr_net_inj), unit="prob/pix", inj=(theta_inj, phi_inj), est=None)
		posterior_obj.plot(logposterior_figname, posterior=np.log10(posterior), title="log10( posterior )\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(log_bayes, snr_net_inj), unit="log10(prob/pix)", inj=(theta_inj, phi_inj), est=None, min=np.max(np.log10(posterior))-log_dynamic_range)
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

	#=========================================
	# check sanity of results
	#=========================================
	print """WRITE TESTS FOR
        the sanity of these results, not just the fact that they don't throw errors
                compare F*mle_strain with data in each detector
	
	write and test a histogram of the posterior [p(h|theta,phi,data)] for the reconstructed signal weighted by the skymap [p(theta,phi|data)] at that point. Do this at:
		each detector (broadcast through antenna patterns into data stream)
		geocenter
	look at coverage plots for:
		sky localization
		frequency moments
		h_rss
		else?

	GOOD VISUALIZATION TOOLS MAY BE USEFUL HERE!
        """


#=================================================
if opts.model_selection:

	import model_selection

	print "model_selection.fixed_bandwidth"
	to=time.time()
	fb_model, fb_lb = model_selection.fixed_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, n_bins=n_bins)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "model_selection.fixed_bandwidth_mp"
		to=time.time()
		fb_model_mp, fb_lb_mp = model_selection.fixed_bandwidth_mp(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posteror_elements, n_pol_eff, freq_truth, n_bins=n_bins, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to

		if opts.check:
			if np.any(fb_model!=fb_model_mp):
				raise StandardError, "model!=model_mp"
			else:
				print "\tmodel==model_mp"
			if fb_lb!=fb_lb_mp:
				raise StandardError, "lb!=lb_mp"
			else:
				print "\tlb==lb_mp"
	
	print "fb_posterior"
	to=time.time()
	fb_posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, fb_model, normalize=True)
	print "\t", time.time()-to

        if not opts.skip_plots:
                print "posterior.plot(fixed_bandwidth)"
                to=time.time()
                posterior_obj.plot(fb_posterior_figname, posterior=fb_posterior, title="posterior\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(fb_lb, snr_net_inj), unit="prob/pix", inj=(theta_inj, phi_inj), est=None)
                posterior_obj.plot(fb_logposterior_figname, posterior=np.log10(fb_posterior), title="log10( posterior )\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(fb_lb, snr_net_inj), unit="log10(prob/pix)", inj=(theta_inj, phi_inj), est=None, min=np.max(np.log10(fb_posterior))-log_dynamic_range)
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(fb_posterior_filename, fb_posterior)

	print "model_selection.variable_bandwidth"
	to=time.time()
	vb_model, vb_lb = model_selection.variable_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "model_selection.variable_bandwidth_mp"
		to=time.time()
		vb_model_mp, vb_lb_mp = model_selection.variable_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to

		if opts.check:
			if np.any(vb_model!=vb_model_mp):
				raise StandardError, "vb_model!=vb_model_mp"
			else:
				print "\tvb_model==vb_model_mp"
			if vb_lb!=vb_lb_mp:
				raise StandardError, "vb_lb!=vb_lb_mp"
			else:
				print "\tvb_lb==vb_lb_mp"
	print "vb_posterior"	
	to=time.time()
        vb_posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, vb_model, normalize=True)
	print "\t", time.time()-to

        if not opts.skip_plots:
                print "posterior.plot(variable_bandwidth)"
                to=time.time()
                posterior_obj.plot(vb_posterior_figname, posterior=vb_posterior, title="posterior\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(vb_lb,snr_net_inj), unit="prob/pix", inj=(theta_inj, phi_inj), est=None)
                posterior_obj.plot(vb_logposterior_figname, posterior=np.log10(vb_posterior), title="log10( posterior )\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(vb_lb,snr_net_inj), unit="log10(prob/pix)", inj=(theta_inj, phi_inj), est=None, min=np.max(np.log10(vb_posterior))-log_dynamic_range)
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(vb_posterior_filename, vb_posterior)

	print "model_selection.log_bayes_cut"
	to=time.time()
	lbc_model, lbc_lb = model_selection.log_bayes_cut(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "model_selection.log_bayes_cut_mp"
		lbc_model_mp, lbc_lb_mp = model_selection.log_bayes_cut_mp(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, max_proc=max_proc)
		print "\t", time.time()-to

		if opts.check:
			if np.any(lbc_model!=lbc_model_mp):
				raise StandardError, "lbc_model!=lbc_model_mp"
			else:
				print "\tlbc_model==lbc_model_mp"
			if lbc_lb!=lbc_lb_mp:
				raise StandardError, "lbc_model!=lbc_model_mp"
			else:
				print "\tlbc_model==lbc_model_mp"

	print "lbc_posterior"
	to=time.time()
        lbc_posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, lbc_model, normalize=True)
	print "\t", time.time()-to

        if not opts.skip_plots:
                print "posterior.plot(log_bayes_cut)"
                to=time.time()
                posterior_obj.plot(lbc_posterior_figname, posterior=lbc_posterior, title="posterior\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(lbc_lb,snr_net_inj), unit="prob/pix", inj=(theta_inj, phi_inj), est=None)
                posterior_obj.plot(lbc_logposterior_figname, posterior=np.log10(lbc_posterior), title="log10( posterior )\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(lbc_lb,snr_net_inj), unit="log10(prob/pix)", inj=(theta_inj, phi_inj), est=None, min=np.max(np.log10(lbc_posterior))-log_dynamic_range)
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(lbc_posterior_filename, lbc_posterior)

	print """WRITE TESTS FOR
	remaining model_selection (to be written?)
	"""


