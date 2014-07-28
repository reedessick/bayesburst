usage="""a bare-bones injection generation module meant for testing purposes"""
### written by R. Essick (ressick@mit.edu)

#=================================================

import numpy as np
import utils

#=================================================
# waveforms
#=================================================
###
def gaussian_t(t, to, tau, hrss, alpha=np.pi/2):
	"""
	basic gaussian in time domain
	"""
	ho = np.array( [np.cos(alpha), np.sin(alpha)] ) * hrss*tau**-0.5*(2/np.pi)**0.25
	return np.outer(np.exp(-((t-to)/tau)**2 ), ho)

###
def gaussian_f(f, to, tau, hrss, alpha=np.pi/2):
	"""
	basic gaussian in freq domain
	"""
	ho = np.array( [np.cos(alpha), np.sin(alpha)]) * hrss*(tau)**0.5*(2*np.pi)**0.25
	return np.outer(np.exp(-(f*np.pi*tau)**2)*np.exp(+2j*np.pi*f*to), ho)

###
def sinegaussian_t(t, to, phio, fo, tau, hrss, alpha=np.pi/2):
	"""
	basic sinegaussian in the time domain
	"""
	Q = (2)**0.5*np.pi*tau*fo
	phs = 2*np.pi*fo*(t-to)+phio
	ho = hrss * np.array( [np.cos(alpha)*(1+np.cos(2*phio)*np.exp(-Q**2))**-0.5, np.sin(alpha)*(1-np.cos(2*phio)*np.exp(-Q**2))**-0.5] ) * (4*np.pi**0.5*fo/Q)**0.5
	
	return np.outer(np.exp( -(t-to)**2/(2*tau**2) ) * np.array( [np.cos(phs), np.sin(phs)] ), ho)

###
def sinegaussian_f(f, to, phio, fo, tau, hrss, alpha=np.pi/2):
	"""
	basic sinegaussian in the freq domain
	"""
	Q = (2)**0.5*np.pi*tau*fo
	f_p2 = (f+fo)**2
	f_m2 = (f-fo)**2
	pi_tau2 = (np.pi*tau)**2
	e_p = np.exp(-f_p2*pi_tau2 + 1j*phio)
	e_m = np.exp(-f_m2*pi_tau2 - 1j*phio)

	ho = hrss * np.array( [np.cos(alpha)*(1+np.cos(2*phio)*np.exp(-Q**2))**-0.5, -1j*np.sin(alpha)*(1-np.cos(2*phio)*np.exp(-Q**2))**-0.5] ) * (4*np.pi**0.5*fo/Q)**0.5 * (np.pi**0.5 *tau / 2 )
	
	#return np.outer(np.exp(+2j*np.pi*fo*to) * np.array( [e_m + e_p, e_m - e_p] ), ho)
	sine_gaus = np.zeros((len(f),2), 'complex')
	sine_gaus[:,0] = np.exp(+2j*np.pi*fo*to)*(e_m + e_p)[:]*ho[0]
	sine_gaus[:,1] = np.exp(+2j*np.pi*fo*to)*(e_m - e_p)[:]*ho[1]
	
	return sine_gaus

#=================================================
# injection/projection methods
#=================================================
###
def project(F, h):
	"""
	project the wave-frame signal h onto the detectors using antenna patterns F
		 F[time/freq][pol], h[time/freq][pol] 
	"""
	return sum(np.transpose(F*h)) # sum over pol

###
def inject(network, h, theta, phi, psi=0.0):
	"""
	generate signal vectors in each detector within network
	this works exclusively in the freq domain
	"""
	freq = network.freq
	n_freq = len(freq)
	detectors = netowrk.detectors_list()
	n_det = len(detectors)
	signal = np.zeros((n_freq, n_det), complex=True)
	for det_ind, detector in enumerate(detectors):
		F = np.asarray( detector.antenna_patterns(theta, phi, psi, freqs=freq) )
		signal[:, det_ind] = project( F, h )
	return signal
		
		
#=====================================================

def skypos_uni_vol(detectors, freqs, to, phio, fo, tau, hrss, alpha, snrcut, n_runs=1, num_pol=2):
	"""
	generate a sky position that is uniform in volume, subject to an SNR cutoff of snrcut
	"""
	angles = np.zeros((n_runs, 2.))
	snrs = np.zeros(n_runs)
	
	df_test = 0.1
	freqs_test = np.arange(10., 2048., df_test)
	test_network = utils.Network(detectors=detectors, freqs=freqs_test, Np = num_pol)
	h_test = sinegaussian_f(f=freqs_test, to=to, phio=phio, fo=fo, tau=tau, hrss=hrss, alpha=alpha)  #2-D array (frequencies x polarizations)
	h_test_conj = np.conj(h_test)
	
	h = sinegaussian_f(f=freqs, to=to, phio=phio, fo=fo, tau=tau, hrss=hrss, alpha=alpha)  #2-D array (frequencies x polarizations)
	
	for i in xrange(n_runs):
		snr = 0.
		count = 0
		while snr <= snrcut:
			snr = 0.
			
			theta = np.arccos(np.random.uniform(low=-1., high=1.))
			phi = np.random.uniform(low=0., high=2.*np.pi)			
			
			A = test_network.A(theta=theta, phi=phi, psi=0., no_psd=False)  #3-D array (frequencies x polarizations x polarizations)
			
			for m in xrange(num_pol):
				for n in xrange(num_pol):
					snr += (4.*df_test)*np.sum(h_test_conj[:,m]*A[:,m,n]*h_test[:,n])
			snr = np.sqrt(snr)
			
			count+=1
			if count >= 1000:
				raise ValueError, "Couldn't find sky location that gives SNR %f with an hrss %e in 1000 iterations"%(snrcut, hrss)
		
		print "Found SNR of event %u to be %f after %u iterations"%(i, np.real(snr), count)
		angles[i,0] = theta
		angles[i,1] = phi
		snrs[i] = np.real(snr)
		
		hrss_calc = 0.
		for m in xrange(num_pol):
			hrss_calc += (2.*df_test)*np.sum(h_test_conj[:,m]*h_test[:,m])
		print "given hrss = ", hrss
		print "calculated hrss = ", np.real(np.sqrt(hrss_calc))
		
	return h, angles, snrs
	
	
