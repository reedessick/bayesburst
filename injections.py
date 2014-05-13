usage="""a bare-bones injection generation module meant for testing purposes"""
### written by R. Essick (ressick@mit.edu)

#=================================================

import numpy as np
import utils

#=================================================
def sinegaussian_t(t, to, phio, fo, tau, hrss, alpha=np.pi/2):
	"""
	basic sinegaussian in the time domain
	"""
	Q = (2)**0.5*np.pi*tau*fo
	phs = 2*np.pi*fo*(t-to)+phio
	ho = hrss * np.array( [np.cos(alpha)*(1+np.cos(2*phio)*np.exp(-Q**2))**-0.5, np.sin(alpha)*(1-np.cos(2*phio)*np.exp(-Q**2))**-0.5] ) * (4*np.pi**0.5*fo/Q)**0.5
	h = np.outer(np.exp( -(t-to)**2/(2*tau**2) ) * np.array( [np.cos(phs), np.sin(phs)] ), ho)

	return h

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

	h = np.outer(np.exp(-2j*np.pi*fo*to) * np.array( [e_m + e_p, e_m - e_p] ), ho)

	return h

###
def project(F, h):
	"""
	project the wave-frame signal h onto the detectors using antenna patterns F
		 F[freq][pol], h[freq][pol] 
	"""
	hT = np.transpoxe(h)
	n_freq, n_pol = np.shape(F)
	return sum(np.transpose(F*h)) # sum over pol

###
def inject(network, h, theta, phi, psi=0.0):
	"""
	generate signal vectors in each detector within network
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
		
