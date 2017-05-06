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

parser.add_option("", "--network", default="HL", type="string", help="which network to run")

parser.add_option("", "--hPrior", default=False, action="store_true")
parser.add_option("", "--malmquist-hPrior", default=False, action="store_true")
parser.add_option("", "--hPrior_pareto", default=False, action="store_true")

parser.add_option("", "--angPrior", default=False, action="store_true")
parser.add_option("", "--ap_angPrior", default=False, action="store_true")

parser.add_option("", "--posterior", default=False, action="store_true")
parser.add_option("", "--dpf", default=False, action="store_true")

parser.add_option("", "--model-selection", default=False, action="store_true")

parser.add_option("", "--num-proc", default=2, type="int")
parser.add_option("", "--max-proc", default=2, type="int")
parser.add_option("", "--max-array-size", default=100, type="int")

parser.add_option("", "--pkl", default=False, action="store_true")
parser.add_option("", "--check", default=False, action="store_true")
parser.add_option("", "--skip-mp", default=False, action="store_true")
parser.add_option("", "--skip-plots", default=False, action="store_true")
parser.add_option("", "--skip-diagnostic", default=False, action="store_true")
parser.add_option("", "--skip-diagnostic-plots", default=False, action="store_true")

parser.add_option("", "--zero-data", default=False, action="store_true")
parser.add_option("", "--zero-noise", default=False, action="store_true")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

opts, args = parser.parse_args()

if opts.tag:
	opts.tag = "_"+opts.tag

if opts.model_selection:
	opts.posterior = True

if opts.dpf:
	opts.posterior = True

if opts.posterior:
	opts.hPrior = opts.angPrior = True

opts.skip_diagnostic_plots = opts.skip_diagnostic + opts.skip_diagnostic_plots

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

num_proc = opts.num_proc
max_proc = opts.max_proc
max_array_size = opts.max_array_size

eps = 1e-03 ### precision for "floating point errors"? Not sure where the errors are coming into AB_A and AB_invA relative to A, invA
            ### important for comparing output from different methods
eps_bayes = 1e-03 ### different parameter for log_bayes...

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

n_gaus_per_decade = 2 ### approximate scaling found empirically to make my decomposition work well
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
nside_exp = 4
nside = 2**nside_exp
n_pix = hp.nside2npix(nside)
prior_type="uniform"

### set up stuff for ap_angprior
if opts.network == "HL":
        network = utils.Network([detector_cache.LHO, detector_cache.LLO], freqs=freqs, Np=n_pol)
elif opts.network == "HV":
        network = utils.Network([detector_cache.LHO, detector_cache.Virgo], freqs=freqs, Np=n_pol)
elif opts.network == "LV":
        network = utils.Network([detector_cache.Virgo, detector_cache.LLO], freqs=freqs, Np=n_pol)
elif opts.network == "HLV":
        network = utils.Network([detector_cache.LHO, detector_cache.LLO, detector_cache.Virgo], freqs=freqs, Np=n_pol)
else:
        raise ValueError, "--network=%s not understood"%opts.network

n_ifo = len(network.detectors)

### set up stuff for posterior

freq_truth = np.ones_like(freqs, bool)

### set up stuff for model selection
n_bins = 23

min_n_bins = 15
max_n_bins = 25
dn_bins = 1

log_bayes_thr = 0
generous_log_bayes_thr = -1

### plotting options
log_dynamic_range = 100

### define injection data
import injections

to=0.0
phio=0.0
fo=200
tau=0.010
#tau=0.100
q=2**0.5*np.pi*fo*tau ### the sine-gaussian's q, for reference
#hrss=2e-22 #network SNR ~50 (screaming)
#hrss=1e-22 #network SNR ~25 (cacophonous)
hrss=6e-23 #network SNR ~15 (loud)
#hrss=5e-23 #network SNR ~ 12.5 (audible)
#hrss=4e-23 #network SNR ~10 (quiet)
#hrss=2e-23 #network SNR ~5 (silent)

print "injecting data"
if opts.zero_data:
	data_inj = np.zeros((n_freqs, n_ifo), complex)
	snr_net_inj = 0.0
	injang=None
else:
	print "generating injection"
	theta_inj =   np.pi/4
	phi_inj   = 3*np.pi/2

	h = injections.sinegaussian_f(freqs, to=to, phio=phio, fo=fo, tau=tau, hrss=hrss, alpha=np.pi/2)

	data_inj = injections.inject(network, h, theta_inj, phi_inj, psi=0.0)
	snrs_inj = network.snrs(data_inj) ### compute individual SNRs for detectors
	snr_net_inj = np.sum(snrs_inj**2)**0.5 ### network SNR

	injang=(theta_inj, phi_inj)

if opts.zero_noise:
	noise = np.zeros((n_freqs, n_ifo), complex)
else:
	print "drawing noise"
	noise = network.draw_noise()

data = data_inj + noise

#=================================================
### filenames
tag = "_%d-%d-%d%s"%(n_freqs, n_gaus, nside_exp, opts.tag)

hfigname="%s/hprior%d%s.png"%(opts.output_dir, n_gaus_per_decade, tag)
malmquist_hfigname="%s/malmquist_hprior%d%s.png"%(opts.output_dir, n_gaus_per_decade, tag)

posterior_figname = "%s/posterior%s.png"%(opts.output_dir, tag)
logposterior_figname="%s/log-posterior%s.png"%(opts.output_dir, tag)

fb_posterior_figname="%s/posterior-fixed_bandwidth%s.png"%(opts.output_dir, tag)
fb_logposterior_figname="%s/log-posterior-fixed_bandwidth%s.png"%(opts.output_dir, tag)

vb_posterior_figname="%s/posterior-variable_bandwidth%s.png"%(opts.output_dir, tag)
vb_logposterior_figname="%s/log-posterior-variable_bandwidth%s.png"%(opts.output_dir, tag)

stacked_vb_posterior_figname="%s/posterior-stacked_variable_bandwidth%s.png"%(opts.output_dir, tag)
stacked_vb_logposterior_figname="%s/log-posterior-stacked_variable_bandwidth%s.png"%(opts.output_dir, tag)

lbc_posterior_figname="%s/posterior-log_bayes_cut%s.png"%(opts.output_dir, tag)
lbc_logposterior_figname="%s/log-posterior-log_bayes_cut%s.png"%(opts.output_dir, tag)

ma_posterior_figname="%s/posterior-model_average%s.png"%(opts.output_dir, tag)
ma_logposterior_figname="%s/log-posterior-model_average%s.png"%(opts.output_dir, tag)

wf_posterior_figname="%s/posterior-waterfill%s.png"%(opts.output_dir, tag)
wf_logposterior_figname="%s/log-posterior-waterfill%s.png"%(opts.output_dir, tag)

posterior_pklname = "%s/posterior%s.pkl"%(opts.output_dir, tag)
posterior_filename = "%s/posterior%s.fits"%(opts.output_dir, tag)

fb_posterior_filename = "%s/posterior-fixed_bandwidth%s.fits"%(opts.output_dir, tag)
vb_posterior_filename = "%s/posterior-variable_bandwidth%s.fits"%(opts.output_dir, tag)
stacked_vb_posterior_filename = "%s/posterior-stacked_variable_bandwidth%s.fits"%(opts.output_dir, tag)
lbc_posterior_filename = "%s/posterior-log_bayes_cut%s.fits"%(opts.output_dir, tag)
ma_posterior_filename = "%s/posterior-model_average%s.fits"%(opts.output_dir, tag)
wf_posterior_filename = "%s/posterior-waterfill%s.fits"%(opts.output_dir, tag)

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
	if opts.check:
		invcovariance = hprior_obj.invcovariance
		detinvcovariance = hprior_obj.detinvcovariance

	print "hPrior.set_covariance(byhand)"
	to=time.time()
	hprior_obj.set_covariance(pareto_covariance, n_freqs=n_freqs, n_pol=n_pol, n_gaus=n_gaus, byhand=True)
	print "\t", time.time()-to
	if opts.check:
		invcovariance_byhand = hprior_obj.invcovariance
		detinvcovariance_byhand = hprior_obj.detinvcovariance

		if np.any(np.abs(invcovariance-invcovariance_byhand) > eps*np.abs(invcovariance+invcovariance_byhand)):
			raise StandardError, "invcovariance != invcovariance_byhand"
		elif np.any(invcovariance!=invcovariance_byhand):
			print "\tinvcovariance-invcovariance_byhand < %s*(invcovariance+invcovariance_byhand)"%str(eps)
		else:
			print "\tinvcovariance==invcovariance_byhand"

		if np.any(np.abs(detinvcovariance-detinvcovariance_byhand) > eps*np.abs(detinvcovariance+detinvcovariance_byhand)):
			raise StandardError, "detinvcovariance != detinvcovariance_byhand"
		elif np.any(detinvcovariance!=detinvcovariance_byhand):
			print "\tdetinvcovariance-detinvcovariance_byhand <= %s*(detinvcovariance+detinvcovariance_byhand)"%str(eps)
		else:
			print "\tdetinvcovariance==detinvcovariance_byhand"

	if opts.hPrior_pareto:
		print "hPrior_pareto"
		to=time.time()
		hprior_obj = priors.hPrior_pareto(a, variances, freqs=freqs, n_freqs=n_freqs, n_gaus=n_gaus, n_pol=n_pol, byhand=True)
		print "\t", time.time()-to

		print "hPrior_pareto.get_amplitudes"
		to=time.time()
		amplitudes = hprior_obj.get_amplitudes(freq_truth=freq_truth, n_pol_eff=n_pol)
		print "\t", time.time()-to

        if not opts.skip_plots:
                print "hPrior.plot"
                to=time.time()
                hprior_obj.plot(hfigname, grid=True, xmin=xmin, xmax=xmax, npts=npts)#, ymin=1e0)
                print "\t", time.time()-to

	if opts.malmquist_hPrior:
		print "malmquist_pareto"
		to=time.time()
		malmquist_means, malmquist_covariance, malmquist_amps = priors.malmquist_pareto(a, n_freqs, n_pol, variances[1:], variances[0])
		print "\t", time.time()-to

		print "hPrior(malmquist)"
		to=time.time()
		hprior_obj = priors.hPrior(freqs, malmquist_means, malmquist_covariance, amplitudes=malmquist_amps, n_gaus=n_gaus, n_pol=n_pol)
		print "\t", time.time()-to

		if not opts.skip_plots:
			print "hPrior.plot(malmquist)"
			to=time.time()
			hprior_obj.plot(malmquist_hfigname, grid=True, xmin=xmin, xmax=xmax, npts=npts)#, ymin=1e0)
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
		angprior_obj.plot(angfigname, inj=injang)
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
		ap_angprior_obj.plot(ap_angfigname, inj=injang)
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
	print "\t", time.time()-to
	if opts.check:
		A = posterior_obj.A
		invA = posterior_obj.invA

	print "posterior.set_A(byhand)"
	to=time.time()
	posterior_obj.set_A(byhand=True)
	print "\t", time.time()-to
	if opts.check:
		A_byhand = posterior_obj.A
		invA_byhand = posterior_obj.invA
		if np.any(np.abs(A-A_byhand) > eps*np.abs(A+A_byhand)):
			raise StandardError, "A_byhand != A"
		elif np.any(A!=A_byhand):
			print "\tA_byhand-A <= %s*(A+A_byhand)"%str(eps)
		else:
			print "\tA_byhand==A"
                if np.any(np.abs(invA-invA_byhand) > eps*np.abs(invA+invA_byhand)):
                        raise StandardError, "invA_byhand != invA"
                elif np.any(invA!=invA_byhand):
                        print "\tinvA_byhand-invA <= %s*(invA+invA_byhand)"%str(eps)
                else:
                        print "\tinvA_byhand==invA"

	if not opts.skip_mp:
		print "posterior.set_A_mp"
		to=time.time()
		posterior_obj.set_A_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to
		if opts.check:
			A_mp = posterior_obj.A
			invA_mp = posterior_obj.invA

		print "posterior.set_A_mp(byhand)"
		to=time.time()
		posterior_obj.set_A_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=True)
		print "\t", time.time()-to
		if opts.check:
			A_mp_byhand = posterior_obj.A
			invA_mp_byhand = posterior_obj.invA

		if opts.check:
			if np.any(A!=A_mp):
				raise StandardError, "A!=A_mp"
			else:
				print "\tA==A_mp"
			if np.any(invA!=invA_mp):
				raise StandardError, "invA!=invA_mp"
			else:
				print "\tinvA==invA_mp"

			if np.any(A_byhand!=A_mp_byhand):
				raise StandardError, "A_byhand!=A_mp_byhand"
			else:
				print "\tA_byhand==A_mp_byhand"

			if np.any(invA_byhand!=invA_mp_byhand):
				raise StandardError, "invA_byhand!=invA_mp_byhand"
			else:
				print "\tinvA_byhand==invA_mp_byhand"

	print "posterior.set_B"
	to=time.time()
	posterior_obj.set_B()
	if opts.check:
		B = posterior_obj.B
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.set_B_mp"
		to=time.time()
		posterior_obj.set_B_mp(num_proc=num_proc, max_proc=max_proc)
		if opts.check:
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
	print "\t", time.time()-to
	if opts.check:
		AB_A = posterior_obj.A
		AB_invA = posterior_obj.invA
		AB_B = posterior_obj.B

	print "posterior.set_AB(byhand)"
	to=time.time()
	posterior_obj.set_AB(byhand=True)
	print "\t", time.time()-to
	if opts.check:
		AB_A_byhand = posterior_obj.A
		AB_invA_byhand = posterior_obj.invA
		AB_B_byhand = posterior_obj.B

		if np.any(np.abs(AB_A-AB_A_byhand) > eps*np.abs(AB_A+AB_A_byhand)):
			raise StandardError, "AB_A!=AB_A_byhand"
		elif np.any(AB_A!=AB_A_byhand):
			print "\tAB_A-AB_A_byhand <= %s*(AB_A+AB_A_byhand)"%str(eps)
		else:
			print "\tAB_A==AB_A_byhand"

		if np.any(np.abs(AB_invA-AB_invA_byhand) > eps*np.abs(AB_invA+AB_invA_byhand)):
			raise StandardError, "AB_invA!=AB_invA_byhand" 
		elif np.any(AB_invA!=AB_invA_byhand):
			print "\tAB_invA-AB_invA_byhand <= %s*(AB_invA+AB_invA_byhand)"%str(eps)
		else:
			print "\tAB_invA==AB_invA_byhand"

		if np.any(np.abs(AB_B-AB_B_byhand) > eps*np.abs(AB_B+AB_B_byhand)):
			raise StandardError, "AB_B!=AB_B_byhand"
		elif np.any(AB_B!=AB_B_byhand):
			print "\tAB_B-AB_B_byhand <= %s*(AB_B+AB_B_byhand)"%str(eps)
		else:
			print "\tAB_B==AB_B_byhand"


	if not opts.skip_mp:
		print "posterior.set_AB_mp"
		to=time.time()
		posterior_obj.set_AB_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to
		if opts.check:
			AB_A_mp = posterior_obj.A
			AB_invA_mp = posterior_obj.invA
			AB_B_mp = posterior_obj.B
		
		print "posterior.set_AB_mp(byhand)"
		to=time.time()
		posterior_obj.set_AB_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=True)
		print "\t", time.time()-to
		if opts.check:
			AB_A_mp_byhand = posterior_obj.A
			AB_invA_mp_byhand = posterior_obj.invA
			AB_B_mp_byhand = posterior_obj.B

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

                        if np.any(np.abs(AB_A_mp-AB_A_mp_byhand) > eps*np.abs(AB_A_mp+AB_A_mp_byhand)):
                                raise StandardError, "AB_A_mp!=AB_A_mp_byhand"
                        elif np.any(AB_A_mp!=AB_A_mp_byhand):
                                print "\tAB_A_mp-AB_A_mp_byhand <= %s*(AB_A_mp+AB_A_mp_byhand)"%str(eps)
                        else:
                                print "\tAB_A_mp==AB_A_mp_byhand"
                                
                        if np.any(np.abs(AB_invA_mp-AB_invA_mp_byhand) > eps*np.abs(AB_invA_mp+AB_invA_mp_byhand)):
                                raise StandardError, "AB_invA_mp!=AB_invA_mp_byhand"
                        elif np.any(AB_invA_mp!=AB_invA_mp_byhand):
                                print "\tAB_invA_mp-AB_invA_mp_byhand <= %s*(AB_invA_mp+AB_invA_mp_byhand)"%str(eps)
                        else:
                                print "\tAB_invA_mp==AB_invA_mp_byhand"

                        if np.any(np.abs(AB_B_mp-AB_B_mp_byhand) > eps*np.abs(AB_B_mp+AB_B_mp_byhand)):
                                raise StandardError, "AB_B_mp!=AB_B_mp_byhand"
                        elif np.any(AB_B_mp!=AB_B_mp_byhand):
                                print "\tAB_B_mp-AB_B_mp_byhand <= %s*(AB_B_mp+AB_B_mp_byhand)"%str(eps)
                        else:
                                print "\tAB_B_mp==AB_B_mp_byhand"

	if opts.check:
		if np.any(np.abs(AB_A-A) > eps*np.abs(AB_A+A)):
			raise StandardError, "AB_A!=A"
		elif np.any(AB_A!=A):
			print "\tAB_A-A <= %s*(AB_A+A)"%str(eps)
		else:
			print "\tAB_A==A"
		if np.any(np.abs(AB_invA-invA) > eps*np.abs(AB_invA+invA)):
			raise StandardError, "AB_invA!=invA"
		elif np.any(AB_invA!=invA):
			print "\tAB_invA-invA <= %s*(AB_A+A)"%str(eps)
		else:
			print "\tAB_invA==invA"
		if np.any(AB_B!=B):
			raise StandardError, "AB_B!=B"
		else:
			print "\tAB_B==B"

	print "posterior.set_P"
	to=time.time()
	posterior_obj.set_P()
	print "\t", time.time()-to
	if opts.check:
		P = posterior_obj.P
		invP = posterior_obj.invP
		detinvP = posterior_obj.detinvP

	print "posterior.set_P(byhand)"
	to=time.time()
	posterior_obj.set_P(byhand=True)
	print "\t", time.time()-to
	if opts.check:
		P_byhand = posterior_obj.P
		invP_byhand = posterior_obj.invP
		detinvP_byhand = posterior_obj.detinvP

		if np.any(P != P_byhand):
			raise StandardError, "P!=P_byhand"
		else:
			print "\tP==P_byhand"

		if np.any(np.abs(invP-invP_byhand) > eps*np.abs(invP+invP_byhand)):
			raise StandardError, "invP!=invP_byhand"
		elif np.any(invP != invP_byhand):
			print "\tinvP-invP_byhand <= %s*(invP+invP_byhand)"%str(eps)
		else:
			print "\tinvP==invP_byhand"

		if np.any(np.abs(detinvP-detinvP_byhand) > eps*np.abs(detinvP+detinvP_byhand)):
			raise StandardError, "detinvP!=detinvP_byhand"
		elif np.any(detinvP != detinvP_byhand):
			print "\tdetinvP-detinvP_byhand <= %s*(detinvP+detinvP_byhand)"%str(eps)
		else:
			print "\tdetinvP==detinvP_byhand"

	if not opts.skip_mp:
		print "posterior.set_P_mp"
		to=time.time()
		posterior_obj.set_P_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to
		if opts.check:
			P_mp = posterior_obj.P
			invP_mp = posterior_obj.invP
			detinvP_mp = posterior_obj.detinvP

		print "posterior.set_P_mp(byhand)"
		to=time.time()
		posterior_obj.set_P_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=True)
		print "\t", time.time()-to
		if opts.check:
			P_mp_byhand = posterior_obj.P
			invP_mp_byhand = posterior_obj.invP
			detinvP_mp_byhand = posterior_obj.detinvP

	                if np.any(P_mp != P_mp_byhand):
        	                raise StandardError, "P_mp!=P_mp_byhand"
	                else:
        	                print "\tP_mp==P_mp_byhand"

	                if np.any(np.abs(invP_mp-invP_mp_byhand) > eps*np.abs(invP_mp+invP_mp_byhand)):
        	                raise StandardError, "invP_mp!=invP_mp_byhand"
                	elif np.any(invP_mp != invP_mp_byhand):
                        	print "\tinvP_mp-invP_mp_byhand <= %s*(invP_mp+invP_mp_byhand)"%str(eps)
	                else:
        	                print "\tinvP_mp==invP_mp_byhand"
                
	                if np.any(np.abs(detinvP_mp-detinvP_mp_byhand) > eps*np.abs(detinvP_mp+detinvP_mp_byhand)):
        	                raise StandardError, "detinvP_mp!=detinvP_mp_byhand"
                	elif np.any(detinvP_mp != detinvP_mp_byhand):
                        	print "\tdetinvP_mp-detinvP_mp_byhand <= %s*(detinvP_mp+detinvP_mp_byhand)"%str(eps)
	                else:
        	                print "\tdetinvP_mp==detinvP_mp_byhand"

		if opts.check:
			if np.any(P!=P_mp):
				raise StandardError, "P!=P_mp"
			else:
				print "\tP==P_mp"
			if np.any(invP!=invP_mp):
				raise StandardError, "invP!=invP_mp"
			else:
				print "\tinvP==invP_mp"
			if np.any(detinvP!=detinvP_mp):
				raise StandardError, "invP!=invP_mp"
			else:
				print "\tdetinvP==detinvP_mp"

	print "posterior.set_dataB"
	to=time.time()
	posterior_obj.set_dataB()
	print "\t", time.time()-to
	if opts.check:
		dataB = posterior_obj.dataB
		dataB_conj = posterior_obj.dataB_conj

		if opts.check:
			if np.any(dataB!=np.conjugate(dataB_conj)):
				raise StandardError, "dataB!=conj(dataB_conj)"
			else:
				print "\tdataB==conj(dataB_conj)"

	if not opts.skip_mp:
		print "posterior.set_dataB_mp"
		to=time.time()
		posterior_obj.set_dataB_mp(num_proc=num_proc, max_proc=max_proc)
		if opts.check:
			dataB_mp = posterior_obj.dataB
			dataB_conj_mp = posterior_obj.dataB_conj
		print "\t", time.time()-to

		if opts.check:
			if np.any(dataB!=dataB_mp): 
				raise StandardError, "dataB!=dataB_mp"
			else:
				print "\tdataB==dataB_mp"
			if np.any(dataB_conj!=dataB_conj_mp):
				raise StandardError, "dataB_conj!=dataB_conj_mp"
			else:
				print "\tdataB_conj==dataB_conj_mp"

	#=========================================
	# dpf manipulations and validation
	#=========================================
	if opts.dpf:

		print "posterior.to_dpf"
		to = time.time()
		posterior_obj.to_dpf(byhand=False)
		print "\t", time.time()-to

		if opts.check:
			Adpf = posterior_obj.A
			invAdpf = posterior_obj.invA

			Bdpf = posterior_obj.B
			dataBdpf = posterior_obj.dataB
			dataBdpf_conj = posterior_obj.dataB_conj
		
			if np.any(dataBdpf!=np.conjugate(dataBdpf_conj)):
				raise StandardError, "dataBdpf!=conj(dataBdpf_conj)"
			else:
				print "\tdataBdpf==conj(dataBdpf_conj)"

			Pdpf = posterior_obj.P
			invPdpf = posterior_obj.invP
			detinvPdpf = posterior_obj.detinvP

			detAdpf = np.linalg.det(Adpf)
			detA = np.linalg.det(A)
			if np.any(abs(detAdpf - detA) > eps*abs(detAdpf+detA)):
				raise standardError, "detAdpf != detA"
			elif np.any(np.linalg.det(Adpf) != np.linalg.det(A)):
				print "\tdetAdpf - detA <= %s*(detAdpf + detA)"%str(eps)
			else:
				print "\tdetAdpf == detA"

			TrA = 0.0
			TrAdpf = 0.0
			for x in xrange(n_pol):
				TrA += A[:,:,x,x]
				TrAdpf += Adpf[:,:,x,x]
			if np.any(np.abs(TrA-TrAdpf) > eps*np.abs(TrA+TrAdpf)):
				raise StandardError, "Tr|A| != Tr|Adpf|"
			elif np.any(TrA!=TrAdpf):
				print "\tTrA-TrAdpf <= %s*(TrA+TrAdpf)"%str(eps)
			else:
				print "\tTr|A| == Tr|Adpf|"

	                TrinvA = 0.0
        	        TrinvAdpf = 0.0
                	for x in xrange(n_pol):
                        	TrinvA += invA[:,:,x,x]
	                        TrinvAdpf += invAdpf[:,:,x,x]
			if np.any(np.abs(TrinvA-TrinvAdpf) > eps*np.abs(TrinvA+TrinvAdpf)):
				raise StandardError, "Tr|invA| != Tr|invAdpf|"
        	        if np.any(TrinvA!=TrinvAdpf):
                	        print "\tTrinvA-TrinvAdpf <= %s*(TrinvA+TrinvAdpf)"%str(eps)
			else:	               
        	                print "\tTr|invA| == Tr|invAdpf|"
	
			for g in xrange(n_gaus):
				_detP = np.linalg.det(P[:,:,:,:,g])
				_detPdpf = np.linalg.det(Pdpf[:,:,:,:,g])
				if np.any(np.abs(_detP-_detPdpf) > eps*np.abs(_detP+_detPdpf)):
					raise StandardError, "det|P| != det|Pdpf|"
			else:
				print "\tdetP-detPdpf <= %s*(detP+detPdpf)"%str(eps)

			for g in xrange(n_gaus):
				_detinvP = np.linalg.det(invP[:,:,:,:,g])
				_detinvPdpf = np.linalg.det(invPdpf[:,:,:,:,g])
				if np.any(np.abs(_detinvP-_detinvPdpf) > eps*np.abs(_detinvP+_detinvPdpf)):
					raise StandardError, "det|invP| != det|invPdpf|"
			else:
				print "\tdetinvP - detinvPdpf <= %s*(detinvP+detinvPdpf)"%str(eps)

			for g in xrange(n_gaus):
				TrP = 0.0
				TrPdpf = 0.0
				for x in xrange(n_pol):
					TrP += P[:,:,x,x,g]
					TrPdpf += Pdpf[:,:,x,x,g]
				if np.any(np.abs(TrP-TrPdpf) > eps*np.abs(TrP+TrPdpf)):
					raise StandardError, "TrP!=TrPdpf"
			else:
				print "\tTrP-TrPdfp <= %s*(TrP+TrPdpf)"%str(eps)

	                for g in xrange(n_gaus):
        	                TrinvP = 0.0
                	        TrinvPdpf = 0.0
                        	for x in xrange(n_pol):
                                	TrinvP += invP[:,:,x,x,g]
	                                TrinvPdpf += invPdpf[:,:,x,x,g]
        	                if np.any(np.abs(TrinvP-TrinvPdpf) > eps*np.abs(TrinvP+TrinvPdpf)):
                	                raise StandardError, "TrinvP!=TrinvPdpf"
	                else:
        	                print "\tTrinvP-TrinvPdpf <= %s*(TrinvP+TrinvPdpf)"%str(eps)

			BAB = np.zeros((n_pix, n_freqs, n_ifo, n_ifo), complex)
			BABdpf = np.zeros_like(BAB, complex)
			for x in xrange(n_ifo):
				for y in xrange(n_ifo):
					for z in xrange(n_pol):
						BAB[:,:,x,y] += np.conjugate(B)[:,:,z,x] * np.sum( A[:,:,z,:] * B[:,:,:,y], axis=-1)
						BABdpf[:,:,x,y] += np.conjugate(Bdpf)[:,:,z,x] * np.sum( Adpf[:,:,z,:] * Bdpf[:,:,:,y], axis=-1)
			if np.any(np.abs(BAB-BABdpf) > eps*np.abs(BAB+BABdpf)):
				raise StandardError, "BAB != BABdpf"
			elif np.any(BAB!=BABdpf):
				print "\tBAB-BABdpf <= %s*(BAB+BABdpf)"%str(eps)
			else:
				print "\tBAB == BABdpf"

			dBABd = np.zeros((n_pix, n_freqs),complex)
			dBABddpf = np.zeros_like(dBABd, complex)
			for x in xrange(n_pol):
				dBABd += dataB_conj[:,:,x] * np.sum(A[:,:,x,:] * dataB, axis=-1)
				dBABddpf += dataBdpf_conj[:,:,x] * np.sum(Adpf[:,:,x,:] * dataBdpf, axis=-1)
			if np.any(np.abs(dBABd-dBABddpf) > eps*np.abs(dBABd+dBABddpf)):
				raise StandardError, "dBABd != dBABddpf"
			elif np.any(dBABd != dBABddpf):
				print "\tdBABd-dBABddpf <= %s*(dBABd+dBABddpf)"%str(eps)
			else:
				print "\tdBABd == dBABddpf"

			for g in xrange(n_ifo):
				BinvPB = np.zeros_like(invP[:,:,:,:,g], complex)
				BinvPBdpf = np.zeros_like(BinvPB, complex)
				for x in xrange(n_ifo):
	                                for y in xrange(n_ifo):
						for z in xrange(n_pol):
			                                BinvPB[:,:,x,y] += np.conjugate(B)[:,:,z,x] * np.sum(invP[:,:,z,:,g] * B[:,:,:,y], axis=-1)
        	                                	BinvPBdpf[:,:,x,y] += np.conjugate(Bdpf)[:,:,z,x] * np.sum(invPdpf[:,:,z,:,g] * Bdpf[:,:,:,y], axis=-1)
				if np.any(np.abs(BinvPB-BinvPBdpf) > eps*np.abs(BinvPB+BinvPBdpf)):
        	                        raise StandardError, "BinvPB != BinvPBdpf"
	                        elif np.any(BinvPB != BinvPBdpf):
					print "\tBinvPB - BinvPBdpf <= %s*(BinvPB+BinvPBdpf)"%str(eps)
                	        else:
                        	        print "\tBinvPB == BinvPBdpf"

				dBinvPBd = np.zeros((n_pix, n_freqs), complex)
				dBinvPBddpf = np.zeros_like(dBinvPBd, complex)
				for x in xrange(n_pol):
		                        dBinvPBd += dataB_conj[:,:,x] * np.sum(invP[:,:,x,:,g] * dataB, axis=-1)
        		                dBinvPBddpf += dataBdpf_conj[:,:,x] * np.sum(invPdpf[:,:,x,:,g] * dataBdpf, axis=-1)
				if np.any(np.abs(dBinvPBd-dBinvPBddpf) > eps*np.abs(dBinvPBd+dBinvPBddpf)):
                        	        raise StandardError, "dBinvPBd != dBinvPBddpf"
                	        elif np.any(dBinvPBd != dBinvPBddpf):
					print "\tdBinvPBd - dBinvPBddpf <= %s*(dBinvPBd+dBinvPBddpf)"%str(eps)
	                        else:
        	                        print "\tdBinvPBd == dBinvPBddpf"

		###
		print "posterior.from_dpf"
		to = time.time()
		posterior_obj.from_dpf()
		print "\t", time.time()-to

		if opts.check:
			Afrom_dpf = posterior_obj.A
			invAfrom_dpf = posterior_obj.invA

			Pfrom_dpf = posterior_obj.P
			invPfrom_dpf = posterior_obj.invP
			detinvPfrom_dpf = posterior_obj.detinvP
			
			Bfrom_dpf = posterior_obj.B
			dataBfrom_dpf = posterior_obj.dataB

			if np.any(np.abs(A-Afrom_dpf) > eps*np.abs(A+Afrom_dpf)):
				raise StandardError, "A != Afrom_dpf"
			elif np.any(A!=Afrom_dpf):
				print "\tA-Afrom_dpf <= %s*(A+Afrom_dpf)"%str(eps)
			else:
				print "\tA==Afrom_dpf"
	
                        if np.any(np.abs(invA-invAfrom_dpf) > eps*np.abs(invA+invAfrom_dpf)):
                                raise StandardError, "A != Afrom_dpf"
                        elif np.any(invA!=invAfrom_dpf):
                                print "\tinvA-invAfrom_dpf <= %s*(invA+invAfrom_dpf)"%str(eps)
			else:
				print "\tinvA==invAfrom_dpf"

                        if np.any(np.abs(P-Pfrom_dpf) > eps*np.abs(P+Pfrom_dpf)):
                                raise StandardError, "P != Pfrom_dpf"
                        elif np.any(P!=Pfrom_dpf):
                                print "\tP-Pfrom_dpf <= %s*(P+Pfrom_dpf)"%str(eps)
			else:
				print "\tP==Pfrom_dpf"
		
                        if np.any(np.abs(invP-invPfrom_dpf) > eps*np.abs(invP+invPfrom_dpf)):
                                raise StandardError, "invP != invPfrom_dpf"
                        elif np.any(invP!=invPfrom_dpf):
                                print "\tinvP-invPfrom_dpf <= %s*(invP+invPfrom_dpf)"%str(eps)
			else:
				print "\tinvP==invPfrom_dpf"

                        if np.any(np.abs(detinvP-detinvPfrom_dpf) > eps*np.abs(detinvP+detinvPfrom_dpf)):
                                raise StandardError, "invP != invPfrom_dpf"
                        elif np.any(detinvP!=detinvPfrom_dpf):
                                print "\tdetinvP-detinvPfrom_dpf <= %s*(detinvP+detinvPfrom_dpf)"%str(eps)
			else:
				print "\tdetinvP==detinvPfrom_dpf"

			if np.any(np.abs(B-Bfrom_dpf) > eps*np.abs(B+Bfrom_dpf)):
				raise StandardError, "B != Bfrom_dpf"
			elif np.any(B!=Bfrom_dpf):
				print "\tB-Bfrom_dpf <= %s*(B+Bfrom_dpf)"
			else:
				print "\tB == Bfrom_dpf"

			if np.any(np.abs(dataB-dataBfrom_dpf) > eps*np.abs(dataB+dataBfrom_dpf)):
				raise StandardError, "dataB != dataBfrom_dpf"
			elif np.any(dataB != dataBfrom_dpf):
				print "\tdataB-dataBfrom_dpf <= %s*(dataB+dataBfrom_dpf)"
			else:
				print "\tdataB == dataBfrom_dpf"

		###
		print "posterior_obj.to_dpf(byhand)"
		to = time.time()
		posterior_obj.to_dpf(byhand=True)
		print "\t", time.time()-to

		if opts.check:
			Adpf_bh = posterior_obj.A
                        invAdpf_bh = posterior_obj.invA

                        Bdpf_bh = posterior_obj.B
                        dataBdpf_bh = posterior_obj.dataB
                        dataBdpf_bh_conj = posterior_obj.dataB_conj

                        if np.any(dataBdpf_bh!=np.conjugate(dataBdpf_bh_conj)):
                                raise StandardError, "dataBdpf_bh!=conj(dataBdpf_bh_conj)"
                        else:
                                print "\tdataBdpf_bh==conj(dataBdpf_bh_conj)"

                        Pdpf_bh = posterior_obj.P
                        invPdpf_bh = posterior_obj.invP
                        detinvPdpf_bh = posterior_obj.detinvP

                        detAdpf_bh = np.linalg.det(Adpf)
                        detA_bh = np.linalg.det(A)
                        if np.any(abs(detAdpf - detAdpf_bh) > eps*abs(detAdpf+detAdpf_bh)):
                                raise standardError, "detAdpf != detAdpf_bh"
                        elif np.any(np.linalg.det(Adpf) != np.linalg.det(Adpf_bh)):
                                print "\tdetAdpf - detAdpf_bh <= %s*(detAdpf + detAdpf_bh)"%str(eps)
                        else:
                                print "\tdetAdpf == detA"

                        TrAdpf_bh = 0.0
                        TrAdpf = 0.0
                        for x in xrange(n_pol):
                                TrAdpf_bh += Adpf_bh[:,:,x,x]
                                TrAdpf += Adpf[:,:,x,x]
                        if np.any(np.abs(TrAdpf_bh-TrAdpf) > eps*np.abs(TrAdpf_bh+TrAdpf)):
                                raise StandardError, "Tr|Adpf_bh| != Tr|Adpf|"
                        elif np.any(TrAdpf_bh!=TrAdpf):
                                print "\tTrAdpf_bh-TrAdpf <= %s*(TrAdpf_bh+TrAdpf)"%str(eps)
                        else:
                                print "\tTr|Adpf_bh| == Tr|Adpf|"

                        TrinvAdpf_bh = 0.0
                        TrinvAdpf = 0.0
                        for x in xrange(n_pol):
                                TrinvAdpf_bh += invAdpf_bh[:,:,x,x]
                                TrinvAdpf += invAdpf[:,:,x,x]
                        if np.any(np.abs(TrinvAdpf_bh-TrinvAdpf) > eps*np.abs(TrinvAdpf_bh+TrinvAdpf)):
                                raise StandardError, "Tr|invAdpf_bh| != Tr|invAdpf|"
                        if np.any(TrinvAdpf_bh!=TrinvAdpf):
                                print "\tTrinvAdpf_bh-TrinvAdpf <= %s*(TrinvAdpf_bh+TrinvAdpf)"%str(eps)
                        else:
                                print "\tTr|invAdpf_bh| == Tr|invAdpf|"

                        for g in xrange(n_gaus):
                                _detPdpf_bh = np.linalg.det(Pdpf_bh[:,:,:,:,g])
                                _detPdpf = np.linalg.det(Pdpf[:,:,:,:,g])
                                if np.any(np.abs(_detPdpf_bh-_detPdpf) > eps*np.abs(_detPdpf_bh+_detPdpf)):
                                        raise StandardError, "det|Pdpf_bh| != det|Pdpf|"
                        else:
                                print "\tdetPdpf_bh-detPdpf <= %s*(detPdpf_bh+detPdpf)"%str(eps)

                        for g in xrange(n_gaus):
                                _detinvPdpf_bh = np.linalg.det(invPdpf_bh[:,:,:,:,g])
                                _detinvPdpf = np.linalg.det(invPdpf[:,:,:,:,g])
                                if np.any(np.abs(_detinvPdpf_bh-_detinvPdpf) > eps*np.abs(_detinvPdpf_bh+_detinvPdpf)):
                                        raise StandardError, "det|invPdpf_bh| != det|invPdpf|"
                        else:
                                print "\tdetinvPdpf_bh - detinvPdpf <= %s*(detinvPdpf_bh+detinvPdpf)"%str(eps)

                        for g in xrange(n_gaus):
                                TrPdpf_bh = 0.0
                                TrPdpf = 0.0
                                for x in xrange(n_pol):
                                        TrPdpf_bh += Pdpf_bh[:,:,x,x,g]
                                        TrPdpf += Pdpf[:,:,x,x,g]
                                if np.any(np.abs(TrPdpf_bh-TrPdpf) > eps*np.abs(TrPdpf_bh+TrPdpf)):
                                        raise StandardError, "TrPdpf_bh!=TrPdpf"
                        else:
                                print "\tTrPdpf_bh-TrPdfp <= %s*(TrPdpf_bh+TrPdpf)"%str(eps)

                        for g in xrange(n_gaus):
                                TrinvPdpf_bh = 0.0
                                TrinvPdpf = 0.0
                                for x in xrange(n_pol):
                                        TrinvPdpf_bh += invPdpf_bh[:,:,x,x,g]
                                        TrinvPdpf += invPdpf[:,:,x,x,g]
                                if np.any(np.abs(TrinvPdpf_bh-TrinvPdpf) > eps*np.abs(TrinvPdpf_bh+TrinvPdpf)):
                                        raise StandardError, "TrinvPdpf_bh!=TrinvPdpf"
                        else:
                                print "\tTrinvPdpf_bh-TrinvPdpf <= %s*(TrinvPdpf_bh+TrinvPdpf)"%str(eps)

                        BABdpf_bh = np.zeros((n_pix, n_freqs, n_ifo, n_ifo), complex)
                        BABdpf = np.zeros_like(BABdpf_bh, complex)
                        for x in xrange(n_ifo):
                                for y in xrange(n_ifo):
                                        for z in xrange(n_pol):
                                                BABdpf_bh[:,:,x,y] += np.conjugate(Bdpf_bh)[:,:,z,x] * np.sum( Adpf_bh[:,:,z,:] * Bdpf_bh[:,:,:,y], axis=-1)
                                                BABdpf[:,:,x,y] += np.conjugate(Bdpf)[:,:,z,x] * np.sum( Adpf[:,:,z,:] * Bdpf[:,:,:,y], axis=-1)
                        if np.any(np.abs(BABdpf_bh-BABdpf) > eps*np.abs(BABdpf_bh+BABdpf)):
                                raise StandardError, "BABdpf_bh != BABdpf"
                        elif np.any(BABdpf_bh!=BABdpf):
                                print "\tBABdpf_bh-BABdpf <= %s*(BABdpf_bh+BABdpf)"%str(eps)
                        else:
                                print "\tBABdpf_bh == BABdpf"

                        dBABddpf_bh = np.zeros((n_pix, n_freqs),complex)
                        dBABddpf = np.zeros_like(dBABddpf_bh, complex)
                        for x in xrange(n_pol):
                                dBABddpf_bh += dataBdpf_bh_conj[:,:,x] * np.sum(Adpf_bh[:,:,x,:] * dataBdpf_bh, axis=-1)
                                dBABddpf += dataBdpf_conj[:,:,x] * np.sum(Adpf[:,:,x,:] * dataBdpf, axis=-1)
                        if np.any(np.abs(dBABddpf_bh-dBABddpf) > eps*np.abs(dBABddpf_bh+dBABddpf)):
                                raise StandardError, "dBABddpf_bh != dBABddpf"
                        elif np.any(dBABddpf_bh != dBABddpf):
                                print "\tdBABddpf_bh-dBABddpf <= %s*(dBABddpf_bh+dBABddpf)"%str(eps)
                        else:
                                print "\tdBABddpf_bh == dBABddpf"

                        for g in xrange(n_ifo):
                                BinvPBdpf_bh = np.zeros_like(invPdpf_bh[:,:,:,:,g], complex)
                                BinvPBdpf = np.zeros_like(BinvPBdpf_bh, complex)
                                for x in xrange(n_ifo):
                                        for y in xrange(n_ifo):
                                                for z in xrange(n_pol):
                                                        BinvPBdpf_bh[:,:,x,y] += np.conjugate(Bdpf_bh)[:,:,z,x] * np.sum(invPdpf_bh[:,:,z,:,g] * Bdpf_bh[:,:,:,y], axis=-1)
                                                        BinvPBdpf[:,:,x,y] += np.conjugate(Bdpf)[:,:,z,x] * np.sum(invPdpf[:,:,z,:,g] * Bdpf[:,:,:,y], axis=-1)
                                if np.any(np.abs(BinvPBdpf_bh-BinvPBdpf) > eps*np.abs(BinvPBdpf_bh+BinvPBdpf)):
                                        raise StandardError, "BinvPBdpf_bh != BinvPBdpf"
                                elif np.any(BinvPBdpf_bh != BinvPBdpf):
                                        print "\tBinvPBdpf_bh - BinvPBdpf <= %s*(BinvPBdpf_bh+BinvPBdpf)"%str(eps)
                                else:
                                        print "\tBinvPBdpf_bh == BinvPBdpf"

                                dBinvPBddpf_bh = np.zeros((n_pix, n_freqs), complex)
                                dBinvPBddpf = np.zeros_like(dBinvPBddpf_bh, complex)
                                for x in xrange(n_pol):
                                        dBinvPBddpf_bh += dataBdpf_bh_conj[:,:,x] * np.sum(invPdpf_bh[:,:,x,:,g] * dataBdpf_bh, axis=-1)
                                        dBinvPBddpf += dataBdpf_conj[:,:,x] * np.sum(invPdpf[:,:,x,:,g] * dataBdpf, axis=-1)
                                if np.any(np.abs(dBinvPBddpf_bh-dBinvPBddpf) > eps*np.abs(dBinvPBddpf_bh+dBinvPBddpf)):
                                        raise StandardError, "dBinvPBddpf_bh != dBinvPBddpf"
                                elif np.any(dBinvPBddpf_bh != dBinvPBddpf):
                                        print "\tdBinvPBddpf_bh - dBinvPBddpf <= %s*(dBinvPBddpf_bh+dBinvPBddpf)"%str(eps)
                                else:
                                        print "\tdBinvPBddpf_bh == dBinvPBddpf"

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
#	print "posterior.n_pol_eff()"
#	to=time.time()
#	posterior_obj.n_pol_eff(posterior_obj.theta, posterior_obj.phi)
#	print "\t", time.time()-to

	print "posterior.mle_strain"
	to=time.time()
#	mle_strain = posterior_obj.mle_strain(posterior_obj.theta, posterior_obj.phi, psi=0.0, n_pol_eff=None, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
	mle_strain = posterior_obj.mle_strain(posterior_obj.theta, posterior_obj.phi, psi=0.0, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
	if not opts.check:
		del mle_strain
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.mle_strain_mp"
		to=time.time()
#		mle_strain_mp = posterior_obj.mle_strain_mp(posterior_obj.theta, posterior_obj.phi, 0.0, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, n_pol_eff=None, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
		mle_strain_mp = posterior_obj.mle_strain_mp(posterior_obj.theta, posterior_obj.phi, 0.0, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, invA_dataB=(posterior_obj.invA, posterior_obj.dataB))
		if not opts.check:
			del mle_strain_mp
		print "\t", time.time()-to

		if opts.check:
			if np.any(mle_strain!=mle_strain_mp):
				raise StandardError, "mle_strain!=mle_strain_mp"
			else:
				print "\tmle_strain==mle_strain_mp"

	print "posterior.log_posterior_elements"
	to=time.time()
	log_posterior_elements, n_pol_eff = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.detinvP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=False)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.log_posterior_elements_mp"
		to=time.time()
		mp_log_posterior_elements, mp_n_pol_eff = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.detinvP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), diagnostic=False, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		if not opts.check:
			del mp_log_posterior_elements, mp_n_pol_eff
		print "\t", time.time()-to

		if opts.check:
			if np.any(log_posterior_elements!=mp_log_posterior_elements) :
				raise StandardError, "conflict between log_posterior_elements and mp_log_posterior_elements"
			else:
				print "\tlog_posterior_elements==mp_log_posterior_elements"
			if np.any(n_pol_eff!=mp_n_pol_eff):
				raise StandardError, "conflict between n_pol_eff and mp_n_pol_eff"
			else:
				print "\tn_pol_eff==mp_n_pol_eff"

	if not opts.skip_diagnostic:
		print "posterior.log_posterior_elements(diagnostic=True)"
		to=time.time()
		log_posterior_elements_diag, n_pol_eff_diag, (mle, cts, det) = posterior_obj.log_posterior_elements(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.detinvP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), connection=None, diagnostic=True)
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
			mp_log_posterior_elements, mp_n_pol_eff, (mp_mle, mp_cts, mp_det) = posterior_obj.log_posterior_elements_mp(posterior_obj.theta, posterior_obj.phi, psi=0.0, invP_dataB=(posterior_obj.invP, posterior_obj.detinvP, posterior_obj.dataB, posterior_obj.dataB_conj), A_invA=(posterior_obj.A, posterior_obj.invA), num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size, diagnostic=True)
			if not opts.check:
				del mp_log_posterior_elements, mp_n_pol_eff
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
			if np.any(np.abs(log_posterior_elements-(mle+cts+det)) > eps*np.abs(log_posterior_elements+mle+cts+det)):
				raise StandardError, "log_posterior_elements!=mle+cts+det"
			elif np.any(log_posterior_elements!=mle+cts+det):
				print "\tlog_posterior_elements - (mle+cts+det) <= %s*(log_posterior_elements + mle+cts+det)"%str(eps)
			else:
				print "\tlog_posterior_elements==mle+cts+det"


	n_pol_eff = n_pol_eff[0] ### we only support integer n_pol_eff right now...

	print "posterior.log_posterior"
	to=time.time()
	log_posterior_unnorm = posterior_obj.log_posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=False)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.log_posterior_mp"
		to=time.time()
		log_posterior_unnorm_mp = posterior_obj.log_posterior_mp(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=False, num_proc=num_proc, max_proc=max_proc)
		if not opts.check:
			del log_posterior_mp
		print "\t", time.time()-to

		if opts.check:
			if np.any(log_posterior_unnorm!=log_posterior_unnorm_mp):
				raise StandardError, "log_posterior_unnorm!=log_posterior_unnorm_mp"
			else:
				print "\tlog_posterior_unnorm==log_posterior_unnorm_mp"

	print "posterior.log_posterior(normalize=True)"
	to=time.time()
	log_posterior = posterior_obj.log_posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)
        print "\t", time.time()-to

        if not opts.skip_mp:
                print "posterior.log_posterior_mp(normalize=True)"
                to=time.time()
                log_posterior_mp = posterior_obj.log_posterior_mp(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True, num_proc=num_proc, max_proc=max_proc)
                if not opts.check:
                        del log_posterior_mp
                print "\t", time.time()-to

                if opts.check:
                        if np.any(log_posterior!=log_posterior_mp):
                                raise StandardError, "log_posterior!=log_posterior_mp"
                        else:
                                print "\tlog_posterior==log_posterior_mp"

	print "posterior.posterior"
	to=time.time()
	posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.posterior_mp"
		to=time.time()
		posterior_mp = posterior_obj.posterior_mp(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True, num_proc=num_proc, max_proc=max_proc)
		if not opts.check:
			del posterior_mp
		print "\t", time.time()-to

		if opts.check:
			if np.any(posterior!=posterior_mp):
				raise StandardError, "posterior!=posterior_mp"
			else:
				print "\tposterior==posterior_mp"
			if np.any(posterior!=np.exp(log_posterior)):
				raise StandardError, "posterior!=np.exp(log_posterior)"
			else:
				print "\tposterior==np.exp(log_posterior)"

	print "posterior.log_bayes"
	to=time.time()
	log_bayes = posterior_obj.log_bayes(log_posterior_unnorm)
	print "\t", time.time()-to

	print "\tn_bins=%d, logBayes=%.3f"%(np.sum(freq_truth), log_bayes)

	if not opts.skip_mp:
		print "posterior.log_bayes_mp"
		to=time.time()
		log_bayes_mp = posterior_obj.log_bayes_mp(log_posterior_unnorm, num_proc=num_proc, max_proc=max_proc)
		if not opts.check:
			del log_bayes_mp
		print "\t", time.time()-to

		if opts.check:
			if np.any(np.abs(log_bayes-log_bayes_mp) > eps_bayes*np.abs(log_bayes+log_bayes_mp)):
				raise StandardError, "log_bayes!=log_bayes_mp"
			elif np.any(log_bayes!=log_bayes_mp):
				print "\tlog_bayes-log_bayes_mp <= %s*(log_bayes+log_bayes_mp)"%str(eps_bayes)
			else:
				print "\tlog_bayes==log_bayes_mp"

	print "posterior.bayes"
	to=time.time()
	bayes = posterior_obj.bayes(log_posterior_unnorm)
	print "\t", time.time()-to

	if not opts.skip_mp:
		print "posterior.bayes_mp"
		to=time.time()
		bayes_mp = posterior_obj.bayes_mp(log_posterior_unnorm, num_proc=num_proc, max_proc=max_proc)
		if not opts.check:
			del bayes_mp
		print "\t", time.time()-to

		if opts.check:
			if np.any(np.abs(log_bayes-log_bayes_mp) > eps_bayes*np.abs(log_bayes+log_bayes_mp)):
				raise StandardError, "bayes!=bayes_mp"
			elif np.any(bayes!=bayes_mp):
				print "\tbayes-bayes_mp <= %s*(bayes+bayes_mp)"%str(eps_bayes)
			else:
				print "\tbayes==bayes_mp"
			if np.any(bayes!=np.exp(log_bayes)):
				raise StandardError, "bayes!=np.exp(log_bayes)"
			else:
				print "\tbayes==np.exp(log_bayes)"

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
		posterior_obj.plot(posterior_figname, posterior=posterior, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(freq_truth), log_bayes, snr_net_inj), unit="prob/pix", inj=injang, est=None)
		posterior_obj.plot(logposterior_figname, posterior=np.log10(posterior), title="log10( posterior )\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(freq_truth), log_bayes, snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None) #, min=np.max(np.min(np.log10(posterior)), np.max(np.log10(posterior))-log_dynamic_range))
		print "\t", time.time()-to

		if not opts.skip_diagnostic_plots:
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
				posterior_obj.plot(diag_figname%("mle",g), posterior=np.exp(_mle), title="mle %s"%s, inj=injang, est=None)
				posterior_obj.plot(logdiag_figname%("mle",g), posterior=_mle, title="log10( mle %s )"%s, inj=injang, est=None)
				print "\t", time.time()-to

				### cts
				print "cts %s"%s
				to=time.time()
				_cts = np.sum(cts[:,g,:], axis=1) * df
				posterior_obj.plot(diag_figname%("cts",g), posterior=np.exp(_cts), title="cts %s"%s, inj=injang, est=None)
                		posterior_obj.plot(logdiag_figname%("cts",g), posterior=_cts, title="log10( cts %s )"%s, inj=injang, est=None)
	                	print "\t", time.time()-to

				### det
				print "det %s"%s
				to=time.time()
				_det = np.sum(det[:,g,:], axis=1) * df
				posterior_obj.plot(diag_figname%("det",g), posterior=np.exp(_det), title="det %s"%s, inj=injang, est=None)
        		        posterior_obj.plot(logdiag_figname%("det",g), posterior=_det, title="log10( det %s )"%s, inj=injang, est=None)
	        	        print "\t", time.time()-to

				### mle+cts+det
				print "mle+cts+det %s"%s
				to=time.time()
				posterior_obj.plot(diag_figname%("mle*cts*det",g), posterior=np.exp(_mle+_cts+_det), title="mle*cts*det %s"%s, inj=injang, est=None)
				posterior_obj.plot(logdiag_figname%("mle*cts*det",g), posterior=_mle+_cts+_det, title="log10( mle*cts*det )", inj=injang, est=None)
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
	
        print "model_selection.log_bayes_cut"
        to=time.time()
        lbc_model, lbc_lb = model_selection.log_bayes_cut(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, joint_log_bayes=True)
        print "\t", time.time()-to

        print "\tn_bins=%d, logBayes=%.3f"%(np.sum(lbc_model), lbc_lb)

        if not opts.skip_mp:
                print "model_selection.log_bayes_cut_mp"
		to=time.time()
                lbc_model_mp, lbc_lb_mp = model_selection.log_bayes_cut_mp(log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, num_proc=num_proc, max_proc=max_proc, joint_log_bayes=True)
                print "\t", time.time()-to

        	print "\tn_bins=%d, logBayes=%.3f"%(np.sum(lbc_model_mp), lbc_lb_mp)
		
		if not opts.check:
			del lbc_model_mp, lbc_lb_mp

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
                posterior_obj.plot(lbc_posterior_figname, posterior=lbc_posterior, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(lbc_model), lbc_lb,snr_net_inj), unit="prob/pix", inj=injang, est=None)
                posterior_obj.plot(lbc_logposterior_figname, posterior=np.log10(lbc_posterior), title="log10( posterior )\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(lbc_model),lbc_lb,snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None)#, min=np.max(np.min(np.log10(lbc_posterior)), np.max(np.log10(lbc_posterior))-log_dynamic_range))
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(lbc_posterior_filename, lbc_posterior)

	print "model_selection.fixed_bandwidth"
	to=time.time()
	fb_model, fb_lb = model_selection.fixed_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, n_bins=n_bins)
	print "\t", time.time()-to

	print "\tn_bins=%d, logBayes=%.3f"%(np.sum(fb_model), fb_lb)

	if not opts.skip_mp:
		print "model_selection.fixed_bandwidth_mp"
		to=time.time()
		fb_model_mp, fb_lb_mp = model_selection.fixed_bandwidth_mp(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, n_bins=n_bins, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to

		print "\tn_bins=%d, logBayes=%.3f"%(np.sum(fb_model_mp), fb_lb_mp)

		if not opts.check:
			del fb_model_mp, fb_lb_mp

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
                posterior_obj.plot(fb_posterior_figname, posterior=fb_posterior, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(fb_model),fb_lb, snr_net_inj), unit="prob/pix", inj=injang, est=None)
                posterior_obj.plot(fb_logposterior_figname, posterior=np.log10(fb_posterior), title="log10( posterior )\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(fb_model),fb_lb, snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None)#, min=np.max(np.min(np.log10(fb_posterior)), np.max(np.log10(fb_posterior))-log_dynamic_range))
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(fb_posterior_filename, fb_posterior)

	print "model_selection.variable_bandwidth"
	to=time.time()
	vb_model, vb_lb = model_selection.variable_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins)
	print "\t", time.time()-to

	print "\tn_bins=%d, logBayes=%.3f"%(np.sum(vb_model), vb_lb)

	if not opts.skip_mp:
		print "model_selection.variable_bandwidth_mp"
		to=time.time()
		vb_model_mp, vb_lb_mp = model_selection.variable_bandwidth_mp(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		print "\t", time.time()-to

		print "\tn_bins=%d, logBayes=%.3f"%(np.sum(vb_model_mp), vb_lb_mp)

		if not opts.check:
			del vb_model_mp, vb_lb_mp
	
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
                posterior_obj.plot(vb_posterior_figname, posterior=vb_posterior, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(vb_model),vb_lb,snr_net_inj), unit="prob/pix", inj=injang, est=None)
                posterior_obj.plot(vb_logposterior_figname, posterior=np.log10(vb_posterior), title="log10( posterior )\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(vb_model),vb_lb,snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None)#, min=np.max(np.min(np.log10(vb_posterior)),np.max(np.log10(vb_posterior))-log_dynamic_range))
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(vb_posterior_filename, vb_posterior)
	
	print "model_selection.variable_bandwidth(model_selection.log_bayes_cut)"
	to=time.time()
	generous_lbc_model = model_selection.log_bayes_cut(generous_log_bayes_thr, posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, joint_log_bayes=False)
	stacked_vb_model, stacked_vb_lb = model_selection.variable_bandwidth(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, generous_lbc_model, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins)
	print "\t", time.time()-to

	print "\tn_bins=%d->%d, logBayes=%.3f"%(np.sum(generous_lbc_model), np.sum(stacked_vb_model), stacked_vb_lb)

	print "stacked_vb_posterior"
	to=time.time()
	stacked_vb_posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, stacked_vb_model, normalize=True)
	print "\t", time.time()-to

        if not opts.skip_plots:
                print "posterior.plot(variable_bandwidth(log_bayes_cut))"
                to=time.time()
                posterior_obj.plot(stacked_vb_posterior_figname, posterior=stacked_vb_posterior, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(stacked_vb_model),stacked_vb_lb,snr_net_inj), unit="prob/pix", inj=injang, est=None)
                posterior_obj.plot(stacked_vb_logposterior_figname, posterior=np.log10(stacked_vb_posterior), title="log10( posterior )\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(stacked_vb_model),stacked_vb_lb,snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None)#, min=np.max(np.min(np.log10(vb_posterior)),np.max(np.log10(vb_posterior))-log_dynamic_range))
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(stacked_vb_posterior_filename, stacked_vb_posterior)

	print "setting up fixed_bandwidth models for model_selection.model_average"
	to=time.time()
	binNos = np.arange(n_freqs)[freq_truth]
	n_models = np.sum(freq_truth)-n_bins+1
	models = np.zeros((n_models, n_freqs),bool)
	for modelNo in xrange(n_models):
		models[modelNo][binNos[modelNo:modelNo+n_bins]] = True
	print "\t", time.time()-to

	print "model_selection.model_average"
	to=time.time()
	ma_log_posterior = model_selection.model_average(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, models)
	ma_posterior = np.exp( ma_log_posterior )
	print "\t", time.time()-to

	if not opts.skip_plots:
                print "posterior.plot(fixed_bandwidth model_average)"
                to=time.time()
                posterior_obj.plot(ma_posterior_figname, posterior=ma_posterior, title="posterior\n$\\rho_{net}$=%.3f"%(snr_net_inj), unit="prob/pix", inj=injang, est=None)
                posterior_obj.plot(ma_logposterior_figname, posterior=np.log10(ma_posterior), title="log10( posterior )\n$\\rho_{net}$=%.3f"%(snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None)#, min=np.max(np.min(np.log10(vb_posterior)),np.max(np.log10(vb_posterior))-log_dynamic_range))
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(ma_posterior_filename, ma_posterior)

        print "model_selection.waterfill"
        to=time.time()
        wf_model, wf_lb = model_selection.waterfill(posterior_obj, posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, freq_truth, connection=None, max_array_size=max_array_size)
        print "\t", time.time()-to

        print "\tn_bins=%d, logBayes=%.3f"%(np.sum(wf_model), wf_lb)

        print "wf_posterior"
        to=time.time()
        wf_posterior = posterior_obj.posterior(posterior_obj.theta, posterior_obj.phi, log_posterior_elements, n_pol_eff, wf_model, normalize=True)
        print "\t", time.time()-to

        if not opts.skip_plots:
                print "posterior.plot(fixed_bandwidth model_average)"
                to=time.time()
                posterior_obj.plot(wf_posterior_figname, posterior=wf_posterior, title="posterior\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(wf_model),wf_lb,snr_net_inj), unit="prob/pix", inj=injang, est=None)
                posterior_obj.plot(wf_logposterior_figname, posterior=np.log10(wf_posterior), title="log10( posterior )\nNo bins=%d\nlogBayes=%.3f\n$\\rho_{net}$=%.3f"%(np.sum(wf_model),wf_lb,snr_net_inj), unit="log10(prob/pix)", inj=injang, est=None)
                print "\t", time.time()-to

        print "writing posterior to file"
        hp.write_map(wf_posterior_filename, ma_posterior)

	print """WRITE TESTS FOR
	remaining model_selection (to be written?)

	plot diagnostic values using the models -> see how these break down
	"""



