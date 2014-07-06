import numpy as np
import healpy as hp
import utils
import priors
import posteriors
import injections as inj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

###############################################################################
########################### specify parameters #############################

num_pol = 2

nside_exp = 5

num_proc = 6

n_runs = 1

threshold = 1.e4

flow = 150.
fhigh = 250.

seg_len = .1 #in seconds
samp_freq = 4096  #in HZ
df = 1./seg_len
Nbins = samp_freq*seg_len

freqs = np.arange(flow, fhigh, df)

detectors = {}
detectors["H1"] = utils.LHO
detectors["L1"] = utils.LLO
#detectors["V1"] = utils.Virgo

###############################################################################
####################### initialize network and priors ########################

nfreqs = len(freqs)
ndet = len(detectors)
print 'ndet = ', ndet

hprior_amplitudes, hprior_means, hprior_covariance, num_gaus = priors.hpri_neg4(len_freqs=nfreqs, num_pol=num_pol, Nbins=Nbins)

network = utils.Network(detectors=[detectors[i] for i in detectors], freqs=freqs, Np = num_pol)
hprior = priors.hPrior(freqs=freqs, means=hprior_means, covariance=hprior_covariance, amplitudes=hprior_amplitudes, num_gaus=num_gaus, num_pol=num_pol)
angprior = priors.angPrior(nside_exp=nside_exp)

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

signal_h, angles, snrs = inj.skypos_uni_vol(detectors=[detectors[i] for i in detectors], freqs=freqs, to=0, phio=np.pi/2., fo=200, tau=0.01, hrss=22e-22, alpha=np.pi/2., snrcut=30., n_runs=n_runs, num_pol=num_pol)

for i_ang in xrange(n_runs):
	
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
	
	theta = angles[i_ang,0]
	phi = angles[i_ang,1]
	snr = snrs[i_ang]

	F = np.zeros((nfreqs, num_pol, ndet), 'complex')
	signal = np.zeros((nfreqs, ndet), 'complex')
	for i_det, det in enumerate(network.detector_names_list()):
		F[:,:,i_det] = network.F_det(det_name=det, theta=theta, phi=phi, psi=0.)
		signal[:,i_det] = inj.project(F[:,:,i_det], signal_h)
				
	noise = np.zeros((nfreqs, ndet), 'complex')
	for i_det, det in enumerate(network.detector_names_list()):
		for i_f in xrange(nfreqs):
			noise[i_f,i_det] = 0.#np.random.normal(loc=0, scale=np.sqrt( network.detectors[det].psd.interpolate(freqs[i_f])/df ) ) * np.exp(np.random.uniform(0.,2*np.pi)*1.j)
		
	data = signal + noise
	for i_det, det in enumerate(network.detector_names_list()):
		fig = plt.figure()
		plt.plot(freqs, np.real(data[:,i_det]), 'r', label='Real')
		plt.plot(freqs, np.imag(data[:,i_det]), 'b', label='Imag')
		plt.xlabel("Frequency [Hz]")
		plt.ylabel("Amplitude")
		plt.legend(loc='best')
		plt.grid()
		fig.savefig("inj_data_%s_nd%d_%d"%(det, ndet, i_ang))
		plt.close(fig)

	###############################################################################
	############################# Find posterior  ################################

	posterior = posteriors.Posterior(freqs=freqs, network=network, hprior=hprior, angprior=angprior, data=data, nside_exp=nside_exp, Nbins=Nbins)
	post_built = posterior.build_posterior(num_proc=num_proc, threshold=threshold)

	############################################################################
	##########   Save Signal Stuff ############################################

	h_ML_theta_phi = posterior.h_ML(theta=theta, phi=phi)
	save_strain0 = zip(signal_h[:,0], h_ML_theta_phi[:,0])
	save_strain1 = zip(signal_h[:,1], h_ML_theta_phi[:,1])
	np.savetxt('hstrain_0_theta_phi_nd%d_%d'%(ndet, i_ang), save_strain0)
	np.savetxt('hstrain_1_theta_phi_nd%d_%d'%(ndet, i_ang), save_strain1)

	for i_det, det in enumerate(network.detector_names_list()):
		save_data = zip(data[:,i_det], inj.project(F[:,:,i_det], h_ML_theta_phi))
		np.savetxt('data_theta_phi_%s_nd%d_%d'%(det, ndet, i_ang), save_data)

	############################################################################
	####################   Deal with Posterior results ##########################

	np.savetxt('posterior_nd%d_%d'%(ndet, i_ang), post_built)

	posterior.plot_posterior(posterior=post_built, figname='skymap_nd%d_%d'%(ndet, i_ang),title="theta %f, phi %f, snr %f"%(theta,phi,snr), unit='posterior density', inj_theta=theta, inj_phi=phi)

	fig = plt.figure()
	plt.hist(post_built[:,1])
	plt.xlabel("Posterior")
	fig.savefig("post_dens_hist_nd%d_%d"%(ndet, i_ang))
	plt.close(fig)


