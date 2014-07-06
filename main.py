usage = "python main.py [--options] detector1 [detector2 ...]"

import numpy as np
import healpy as hp
import utils
import priors
import posteriors
import injections as inj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import time
import detector_cache as det_cache

###############################################################################
########################### specify parameters #############################

from optparse import OptionParser

#=================================================

parser = OptionParser(usage=usage)

parser.add_option("-v", "--verbose", default=False, action="store_true")
parser.add_option("", "--time", default=False, action="store_true", help="print statements about how long the code takes to run")

parser.add_option("-n", "--nside-exp", default=6, type="int", help="HEALPix NSIDE parameter for pixelization is 2**opts.nside_exp")
parser.add_option("-p", "--num-pol", default=2, type="int", help="the number of polarizations we attempt to reconstruct. should be either 1 or 2")
parser.add_option("", "--fmin", default=40, type="float", help="the lowest frequency to analyze. should be specified in Hz")
parser.add_option("", "--fmax", default=1000, type="float", help="the highest frequency to analyze. should be specified in Hz")

parser.add_option("", "--seg-len", default=0.1, type="float", help="The time duration of the data segment, in seconds")
parser.add_option("", "--samp-freq", default=4096., type="float", help="The sampling frequency, in Hz")

parser.add_option("-N", "--num-proc", default=1, type="int", help="the number of processors used to parallelize posterior computation")
parser.add_option("-T", "--num-runs", default=1, type="int", help="the number of injections to simulate")

parser.add_option("", "--thresh", default=1.e4, type="float", help="The threshld ratio between min and max eigenvalues of A for a given pixel.  If the ratio exceeds the threshold for a pixel, the number of effective polarizations is reduced by 1")

opts, args = parser.parse_args()

#=================================================

num_pol   = opts.num_pol
nside_exp = opts.nside_exp
num_proc  = opts.num_proc
n_runs    = opts.num_runs
threshold = opts.thresh
flow = opts.fmin
fhigh = opts.fmax
seg_len = opts.seg_len #in seconds
samp_freq = opts.samp_freq  #in HZ


df = 1./seg_len
Nbins = samp_freq*seg_len
freqs = np.arange(flow, fhigh, df)

### load detectors
if opts.verbose: 
	print "loading list of detectors"
if not args:
	raise ValueError, "supply at least one detector name as an argument"
detectors = {}
for arg in args:
	if arg == "H1":
		detectors["H1"] = det_cache.LHO
	elif arg == "L1":
		detectors["L1"] = det_cache.LLO
	elif arg == "V1":
		detectors["V1"] = det_cache.Virgo
	else:
		raise ValueError, "detector=%s not understood"%arg

###############################################################################
####################### initialize network and priors ########################

nfreqs = len(freqs)
ndet = len(detectors)
print 'ndet = ', ndet

if opts.verbose:
	print "building prior objects"
	if opts.time: to = time.time()

hprior_amplitudes, hprior_means, hprior_covariance, num_gaus = priors.hpri_neg4(len_freqs=nfreqs, num_pol=num_pol, Nbins=Nbins)
hprior = priors.hPrior(freqs=freqs, means=hprior_means, covariance=hprior_covariance, amplitudes=hprior_amplitudes, num_gaus=num_gaus, num_pol=num_pol)
angprior = priors.angPrior(nside_exp=nside_exp)

if opts.verbose and opts.time:
	print "\t", time.time()-to

if opts.verbose: 
	print "building network"
	if opts.time: to=time.time()

network = utils.Network(detectors=[detectors[i] for i in detectors], freqs=freqs, Np = num_pol)

if opts.verbose:
	print network
	if opts.time: print "\t", time.time()-to

fig = plt.figure()
for h in np.logspace(start=-24,stop=-20.,num=500):
	plt.plot(h, hprior.prior_weight_f(h=[h,0], f=freqs[0], Nbins=Nbins), 'ro')
	plt.plot(h, h**(-4.), 'bo')
plt.xlabel("Strain")
plt.xscale('log')
plt.yscale('log')
plt.grid()
fig.savefig("hprior_nd%d"%ndet)
plt.close(fig)

################################################################################
###################### init signal for each run ############################

if opts.verbose:
	print "generating %d random source locations"%n_runs

signal_h, angles, snrs = inj.skypos_uni_vol(detectors=[detectors[i] for i in detectors], freqs=freqs, to=0, phio=np.pi/2., fo=200, tau=0.01, hrss=22e-22, alpha=np.pi/2., snrcut=30., n_runs=n_runs, num_pol=num_pol)

for i_ang in xrange(n_runs):

	if opts.verbose:
		print "injection %d / %d"%(i_ang+1,n_runs)

	if opts.verbose:
		print "plotting waveform"
		if opts.time: to=time.time()
	
	fig = plt.figure()
	plt.plot(freqs, np.real(signal_h[:,0]), 'r', label="Real h0")
	plt.plot(freqs, np.imag(signal_h[:,0]), 'ro', label="Imag h0")
	plt.plot(freqs, np.real(signal_h[:,1]), 'b', label="Real h1")
	plt.plot(freqs, np.imag(signal_h[:,1]), 'bo', label="Imag h1")
	plt.xlabel("Frequency [Hz]")
	plt.ylabel("Amplitude (for each polarization)")
	plt.legend(loc='best')
	plt.grid()
	fig.savefig("inj_hstrain_nd%d_%d"%(ndet, i_ang))
	plt.close(fig)
	
	if opts.verbose and opts.time:
		print "\t",time.time()-to
	
	theta = angles[i_ang,0]
	phi = angles[i_ang,1]
	snr = snrs[i_ang]

	if opts.verbose:
		print "\tcomputing antenna patterns and projecting waveform onto detectors"
		if opts.time: to=time.time()
		
	F = np.zeros((nfreqs, num_pol, ndet), 'complex')
	signal = np.zeros((nfreqs, ndet), 'complex')
	for i_det, det in enumerate(network.detector_names_list()):
		F[:,:,i_det] = network.F_det(det_name=det, theta=theta, phi=phi, psi=0.)
		signal[:,i_det] = inj.project(F[:,:,i_det], signal_h)

	if opts.verbose and opts.time:
		print "\t\t", time.time()-to

	if opts.verbose:
		print "\tgenerating noise"
		if opts.time: to=time.time()
	
	noise = np.zeros((nfreqs, ndet), 'complex')
	for i_det, det in enumerate(network.detector_names_list()):
		for i_f in xrange(nfreqs):
			noise[i_f,i_det] = 0.#np.random.normal(loc=0, scale=np.sqrt( network.detectors[det].psd.interpolate(freqs[i_f])*Nbins ) ) * np.exp(np.random.uniform(0.,2*np.pi)*1.j)
		
	if opts.verbose and opts.time:
		print "\t\t", time.time()-to

	data = signal + noise

	for i_det, det in enumerate(network.detector_names_list()):
		if opts.verbose:
			print "\tplotting data in detector:", det
			if opts.time: to=time.time()
			
		fig = plt.figure()
		plt.plot(freqs, np.real(data[:,i_det]), 'r', label='Real')
		plt.plot(freqs, np.imag(data[:,i_det]), 'b', label='Imag')
		plt.xlabel("Frequency [Hz]")
		plt.ylabel("Amplitude")
		plt.legend(loc='best')
		plt.grid()
		fig.savefig("inj_data_%s_nd%d_%d"%(det, ndet, i_ang))
		plt.close(fig)
		
		if opts.verbose and opts.time:
			print "\t\t", time.time()-to

	###############################################################################
	############################# Find posterior  ################################

	if opts.verbose:
		print "\tcomputing posterior"
		if opts.time: to=time.time()
		
	posterior = posteriors.Posterior(freqs=freqs, network=network, hprior=hprior, angprior=angprior, data=data, nside_exp=nside_exp, Nbins=Nbins)
	post_built = posterior.build_posterior(num_proc=num_proc, threshold=threshold)
	
	if opts.verbose and opts.time:
		print "\t\t", time.time()-to

	############################################################################
	##########   Save Signal Stuff ############################################

	if opts.verbose:
		print "\tsaving signal stuff"
		if opts.time: to=time.time()
		
	h_ML_theta_phi = posterior.h_ML(theta=theta, phi=phi)
	save_strain0 = zip(signal_h[:,0], h_ML_theta_phi[:,0])
	save_strain1 = zip(signal_h[:,1], h_ML_theta_phi[:,1])
	np.savetxt('hstrain_0_theta_phi_nd%d_%d'%(ndet, i_ang), save_strain0)
	np.savetxt('hstrain_1_theta_phi_nd%d_%d'%(ndet, i_ang), save_strain1)

	for i_det, det in enumerate(network.detector_names_list()):
		save_data = zip(data[:,i_det], inj.project(F[:,:,i_det], h_ML_theta_phi))
		np.savetxt('data_theta_phi_%s_nd%d_%d'%(det, ndet, i_ang), save_data)

	if opts.verbose and opts.time:
		print "\t\t", time.time()-to

	############################################################################
	####################   Deal with Posterior results ##########################

	if opts.verbose:
		print "\tsaving posterior"
		if opts.time: to=time.time()
		
	np.savetxt('posterior_nd%d_%d'%(ndet, i_ang), post_built)

	posterior.plot_posterior(posterior=post_built, figname='skymap_nd%d_%d'%(ndet, i_ang),title="theta %f, phi %f, snr %f"%(theta,phi,snr), unit='posterior density', inj_theta=theta, inj_phi=phi)

	fig = plt.figure()
	plt.hist(post_built[:,1])
	plt.xlabel("Posterior")
	fig.savefig("post_dens_hist_nd%d_%d"%(ndet, i_ang))
	plt.close(fig)

	if opts.verbose and opts.time:
		print "\t\t", time.time()-to
