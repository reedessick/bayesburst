usage="""a bare-bones injection generation module meant for testing purposes"""
### written by R. Essick (ressick@mit.edu)

#=================================================

import numpy as np
import utils

#=================================================
# waveforms
#=================================================
###
def gaussian_t(t, to=0, tau=1, alpha=np.pi/2, hrss=1):
	"""
	basic gaussian in time domain
	"""
	ho = np.array( [np.cos(alpha), np.sin(alpha)] ) * hrss*tau**-0.5*(2/np.pi)**0.25
	return np.outer(np.exp(-((t-to)/tau)**2 ), ho)

###
def gaussian_f(f, to=0, tau=1, alpha=np.pi/2, hrss=1):
	"""
	basic gaussian in freq domain
	"""
	ho = np.array( [np.cos(alpha), np.sin(alpha)]) * hrss*(tau)**0.5*(2*np.pi)**0.25
	return np.outer(np.exp(-(f*np.pi*tau)**2)*np.exp(+2j*np.pi*f*to), ho)

###
def sinegaussian_t(t, to=0, phio=0, fo=1, tau=1, alpha=np.pi/2, hrss=1):
	"""
	basic sinegaussian in the time domain
	"""
	Q = (2)**0.5*np.pi*tau*fo
	phs = 2*np.pi*fo*(t-to)+phio
	ho = hrss * np.array( [np.cos(alpha)*(1+np.cos(2*phio)*np.exp(-Q**2))**-0.5, np.sin(alpha)*(1-np.cos(2*phio)*np.exp(-Q**2))**-0.5] ) * (4*np.pi**0.5*fo/Q)**0.5
	
	return np.outer(np.exp( -(t-to)**2/(2*tau**2) ) * np.array( [np.cos(phs), np.sin(phs)] ), ho)

###
def sinegaussian_f(f, to=0, phio=0, fo=1, tau=1, alpha=np.pi/2, hrss=1):
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

		np.shape(F) = (n_pol, n_freqs)
		np.shape(h) = (n_freqs, n_pol)
	"""
	return np.sum(np.transpose(F)*h, axis=1) # sum over pol

###
def inject(network, h, theta, phi, psi=0.0):
	"""
	generate signal vectors in each detector within network
	this works exclusively in the freq domain

	We require:
		np.shape(h) = (n_freqs, n_pol)

	which is compatible with all the functions defined in this module
	"""
	freqs = network.freqs
	n_freq = len(freqs)
	detectors = network.detectors_list()
	n_det = len(detectors)

	signal = np.zeros((n_freq, n_det), complex)

	for det_ind, detector in enumerate(detectors):
		F = np.asarray( detector.antenna_patterns(theta, phi, psi, freqs=freqs) )
		signal[:, det_ind] = project( F, h )

	return signal
		
		
#=================================================
# distributions
#=================================================
def pareto_hrss(network, a, waveform_func, waveform_args, min_hrss=1e-24, min_snr=5, num_inj=1, max_trials=100, verbose=False):
	"""
	draws random sky positions with fixed intrinsic parameters, except hrss is drawn from a pareto distribution with the specified lower bound

	generates waveforms using waveform_func and waveform_args
		waveform_args is passed vi waveform_func(..., **waveform_args)

        sky positions are randomly drawn (uniformly over the sky)
        
        injections are only kept if the proposed position and hrss value produce network_snr > min_snr
        procedure is repeated until num_inj events are found.

	return theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj
	"""
	freqs = network.freqs
	n_ifo = len(network.detectors)

	### place holders for injection parameters
	theta_inj = np.empty((num_inj,),float)
	phi_inj = np.empty((num_inj,),float)
	psi_inj = np.empty((num_inj,),float)
	hrss_inj = np.empty((num_inj,),float)
	snrs_inj = np.zeros((num_inj,n_ifo),float)
	
	inj_id = 0
	trial = 0
	while trial < max_trials:
	
		if inj_id >= num_inj: ### we have enough
			break

		### call random.rand only once each epoch
		theta, phi, psi, hrss = np.random.rand(4)

		### draw position and polarization angle
		theta = np.arccos( 2*theta-1 ) ### uniform in cos(theta)
		phi *= 2*np.pi ### uniform in phi
		psi *= 2*np.pi ### uniform in psi (not inclination!)
	
		### draw hrss
		### cdf = 1 - (hrss/min_hrss)**(1-a) if a > 1
		hrss = min_hrss * (1 - hrss)**(1.0/(1-a))

		### generate waveform with waveform_func
		h = waveform_func(freqs, hrss=hrss, **waveform_args)
		
		### compute snr
		snrs = network.snrs( inject(network, h, theta, phi, psi=psi) )

		if np.sum(snrs**2)**0.5 >= min_snr:
			if verbose: print "trial : %d\tnum_inj : %d"%(trial, inj_id)
			### fill in paramters
			theta_inj[inj_id] = theta
			phi_inj[inj_id] = phi
			psi_inj[inj_id] = psi
			hrss_inj[inj_id] = hrss
			snrs_inj[inj_id,:] = snrs

			inj_id += 1 ### increment id

		trial += 1
	else:
		raise StandardError, "could not find %d injections after %d trials"%(num_inj, trial)

	return theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj

###
def min_snr(network, waveform_func, waveform_args, hrss=1e-23, min_snr=5, num_inj=1, max_trials=100, verbose=False):
        """
	draws random sky positions with fixed injected (intrinsic) parameters

        generates waveforms using waveform_func and waveform_args
                waveform_args is passed vi waveform_func(..., **waveform_args)

        sky positions are randomly drawn (uniformly over the sky)
        
        injections are only kept if the proposed position and hrss value produce network_snr > min_snr
        procedure is repeated until num_inj events are found.

        return theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj
        """
        freqs = network.freqs
        n_ifo = len(network.detectors)

        ### place holders for injection parameters
        theta_inj = np.empty((num_inj,),float)
        phi_inj = np.empty((num_inj,),float)
        psi_inj = np.empty((num_inj,),float)
        hrss_inj = np.empty((num_inj,),float)
        snrs_inj = np.zeros((num_inj,n_ifo),float)

        inj_id = 0
        trial = 0
        while trial < max_trials:

                if inj_id >= num_inj: ### we have enough
                        break

                ### call random.rand only once each epoch
                theta, phi, psi = np.random.rand(3)

                ### draw position and polarization angle
                theta = np.arccos( 2*theta-1 ) ### uniform in cos(theta)
                phi *= 2*np.pi ### uniform in phi
                psi *= 2*np.pi ### uniform in psi (not inclination!)

                ### generate waveform with waveform_func
                h = waveform_func(freqs, hrss=hrss, **waveform_args)

                ### compute snr
                snrs = network.snrs( inject(network, h, theta, phi, psi=psi) )

                if np.sum(snrs**2)**0.5 >= min_snr:
                        if verbose: print "trial : %d\tnum_inj : %d"%(trial, inj_id)
                        ### fill in paramters
                        theta_inj[inj_id] = theta
                        phi_inj[inj_id] = phi
                        psi_inj[inj_id] = psi
                        hrss_inj[inj_id] = hrss
                        snrs_inj[inj_id,:] = snrs

                        inj_id += 1 ### increment id

                trial += 1
        else:
                raise StandardError, "could not find %d injections after %d trials"%(num_inj, trial)

        return theta_inj, phi_inj, psi_inj, hrss_inj, snrs_inj
