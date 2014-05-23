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

n_runs = 5

freqs = np.arange(40. , 1000., 0.25)

detectors = {}
detectors["H1"] = utils.LHO
detectors["L1"] = utils.LLO
#detectors["V1"] = utils.Virgo

###############################################################################
####################### initialize network and priors ########################

nfreqs = len(freqs)
ndet = len(detectors)
print 'ndet = ', ndet

hprior_amplitudes, hprior_means, hprior_covariance, num_gaus = priors.hpri_neg4(len_freqs=nfreqs, num_pol=num_pol)

network = utils.Network(detectors=[detectors[i] for i in detectors], freqs=freqs, Np = num_pol)
hprior = priors.hPrior(freqs=freqs, means=hprior_means, covariance=hprior_covariance, amplitudes=hprior_amplitudes, num_gaus=num_gaus, num_pol=num_pol)
angprior = priors.angPrior(nside_exp=nside_exp)

################################################################################
###################### init signal for each run ############################

cos_theta = np.random.uniform(low=-1., high=1., size=n_runs)
phi = np.random.uniform(low=0., high=2.*np.pi, size=n_runs)
angles = np.zeros((n_runs, 2.))
angles[:,0] = np.arccos(cos_theta)
angles[:,1] = phi

for i_ang in xrange(n_runs):
	
	signal_h = inj.sinegaussian_f(f=freqs, to=0., phio=np.pi/2., fo=200., tau=0.01, hrss=5.e-22, alpha=np.pi/2)

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

	F = np.zeros((nfreqs, num_pol, ndet), 'complex')
	signal = np.zeros((nfreqs, ndet), 'complex')
	for i_det, det in enumerate(network.detector_names_list()):
		F[:,:,i_det] = network.F_det(det_name=det, theta=theta, phi=phi, psi=0.)
		signal[:,i_det] = inj.project(F[:,:,i_det], signal_h)
				
	noise = np.zeros((nfreqs, ndet), 'complex')
	for i_det, det in enumerate(network.detector_names_list()):
		for i_f in xrange(nfreqs):
			noise[i_f,i_det] = np.random.normal(loc=0, scale=np.sqrt( network.detectors[det].psd.interpolate(freqs[i_f]) ) ) * np.exp(np.random.uniform(0.,2*np.pi)*1.j)
		
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

	posterior = posteriors.Posterior(freqs=freqs, network=network, hprior=hprior, angprior=angprior, data=data, nside_exp=nside_exp)
	post_built = posterior.build_posterior(num_proc=num_proc)

	for i in xrange( np.shape(post_built)[0] ):
		if post_built[i,1] == 0:
			post_built[i,1] = 1.e-323

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

	posterior.plot_posterior(posterior=post_built, figname='skymap_nd%d_%d'%(ndet, i_ang),title="theta %f, phi %f"%(theta,phi), unit='log posterior density', inj_theta=theta, inj_phi=phi)

	fig = plt.figure()
	plt.hist(np.log10(post_built[:,1]))
	plt.xlabel("Log Posterior")
	fig.savefig("post_dens_hist_nd%d_%d"%(ndet, i_ang))
	plt.close(fig)


