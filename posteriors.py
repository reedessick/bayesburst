import numpy as np
from numpy import linalg
import healpy as hp

#=================================================
#
#                Prior classes
#
#=================================================
class hPrior(object):
	"""
	An object that creates the priors.  Because this framework is an excercise
	in Gaussian integral, we will assume that the priors can be decomposed into
	a series of Gaussians.
	"""
	
	def __init__(self, freqs, means, covariance, amplitudes=1., num_pol=2):
		"""
		Priors are currently assumed to have form C_N(f) * exp( - ( h_k(f) - mean_k(f) )^* Z_kj(f) ( h_j(f) - mean_j(f) ) )
		Thus:
			*freqs is a 1-D array
			*means is a 2-D array, (frequencies x polarizations)
			*covariance is a 3-D array, (frequencies x polarizations x polarizations)
			*amplitudes is a 1-D array, (frequencies) (*for now*)
		N.b that values must be specified over relevant frequency range
		"""
		
		len_freqs = np.shape(np.array(freqs))[0]
		if not len_freqs:
			raise ValueError, "freqs must have at least 1 entries"
		#any other checks on frequency
		self.freqs = np.array(freqs)
		
		if not num_pol:
			raise ValueError, "Must specify at least one polarization"
		self.num_pol = num_pol
		
		if type(means) == int or type(means) == float:
			self.means = np.array(len_freqs*[self.num_pol*[means]])
		elif np.shape(np.array(means))[0] != len_freqs:
			raise ValueError, "freqs and means must have the same length"
		elif np.shape(np.array(means))[1] != self.num_pol:
			raise ValueError, "must specify means for correct number of polarizations"
		else:
			self.means = np.array(means)
			
		if type(covariance) == int or type(covariance) == float:
			self.covariance = np.array([covariance*np.identity(self.num_pol) for i in self.freqs])
		elif np.shape(covariance)[0] != np.shape(self.freqs)[0]:
			raise ValueError, "covariance matrices must have correct frequency length"
		elif np.shape(covariance)[1] != np.shape(covariance)[2]:
			raise ValueError, "At a given frequency, covariance matrix must be square"
		else:
			self.covariance = covariance
		
		for i in np.arange(len(self.freqs)):
			if linalg.det(self.covariance[i]) == 0:  #check if covariance is invertible at each frequency
				raise ValueError, "Currently covariance must be invertible at each frequency"
		self.incovariance = linalg.inv(self.covariance)
		
		if type(amplitudes) == int or type(amplitudes) == float:
			self.amplitudes = np.array(len_freqs*[amplitudes])
		elif np.shape(np.array(amplitudes))[0] != len_freqs:
			raise ValueError, "freqs and amplitudes must have the same length"
		else:
			self.amplitudes = np.array(amplitudes)
		
	def f2i(self, f):
		"""
		takes frequency and returns index
		"""
		i = np.where(self.freqs == f)
		if i[0].tolist() == []:
			raise ValueError, "Specified frequency not contained in frequency array"
		return i[0][0]
	
	def i2f(self, i):
		"""
		takes index and returns frequency
		"""
		return self.freqs[i]
		
	def means_f(self, f):
		findex = self.f2i(f)
		return self.means[findex]
		
	def covariance_f(self, f):
		findex = self.f2i(f)
		return self.covariance[findex]
		
	def incovariance_f(self, f):
		findex = self.f2i(f)
		return self.incovariance[findex]
		
	def amplitude_f(self, f):
		findex = self.f2i(f)
		return self.amplitudes[findex]
		
	def prior_weight_f(self, h, f):
		"""
		returns value of prior for a strain vector (over polarizations)
		for a given freqiency
		"""
		if type(h) == int or type(h) == float:
			raise ValueError, "Please define h as either list or array over polarizations at f"
		if np.shape(np.array(h))[0] != self.num_pol:
			raise ValueError, "Must define value of h for each polarization"
		h = np.array(h)
		
		pol_indices = np.arange(self.num_pol)
		
		displacement = h - self.means_f(f)
		displacement_conj = np.conj(displacement)
		matrix = self.incovariance_f(f)
		
		exponent = 0.
		for i in pol_indices:
			for j in pol_indices:
				exponent += -displacement_conj[i] * matrix[i][j] * displacement[j]
		value = self.amplitude_f(f)*np.exp(exponent)
		return value
	
	#add function that calculates full prior over all freqs 



class angPrior(object):
	#this might be easily implemented as option-based
	#at first make the default option to be uniform
	#will want to add some sort of beam pattern option
	#eventually should add galaxy catalogues option
	#eventually should allow for user to input general form
	
	def __init__(self, freqs):
		self.freqs = np.array(freqs)
	
	def angular_prior(self, theta, phi):
		"""
		For now, just set angPrior equal to uniform over the sphere
		"""
		
		return 1. / (4. * np.pi)
	
	#have function that calculates full prior
	#have function that maps prior to sky



#=================================================
#
#                Posterior class
#
#=================================================

class Posterior(object):
	"""
	An object that calculates the sky localization posterior at a given
	set of angular coordinates 
	"""
	

	def __init__(self, freqs, network, hPrior, angPrior, data, nside_exp):
		"""
		*freqs is a 1-D array of frequencies
		*network is a Network object defined in utils.py
		*hPrior and angPrior are Prior objects, defined above
		*data is a 2-D array, (frequencies x detectors)
		*nside = 2**nside_exp
		"""
		
		if "%s"%type(network) != "<class 'utils.Network'>"  #check that a valid network has been passed
			raise ValueError, "network must be a utils.Network object"
		self.network = network
		self.ifos = network.detectors_list  #define list of ifos
		self.num_detect = len(self.ifos)	#defines number of detectors
		
		if "%s"%type(hPrior) != "<class 'posteriors.hPrior'>"  #check that a valid hPrior has been passed
			raise ValueError, "hPrior must be a posterior.hPrior object"
		if "%s"%type(angPrior) != "<class 'posteriors.angPrior'>"  #check that a valid angPrior has been passed
			raise ValueError, "angPrior must be a posteriors.angPrior object"
		self.hPrior = hPrior
		#add way to check for consistent number of polarizations
		self.num_pol = hPrior.num_pol
		self.angPrior = angPrior
		
		self.len_freqs = np.shape(np.array(freqs))[0]
		if not len_freqs:
			raise ValueError, "freqs must have at least 1 entry"
		self.freqs = np.array(freqs)
		if (self.freqs != hPrior.freqs) or (self.freqs != angPrior.freqs):  #check that same freqs are used for priors
			raise ValueError, "specified frequencies must match those in hPrior and angPrior"
		self.delta_f = (self.freqs[-1] - self.freqs[0])/(float(self.len_freqs) - 1.)

		if np.shape(np.array(data))[0] != self.len_freqs:  #check that data is specified at every frequency
			raise ValueError, "data must be specified at each of the specified frequencies"
		if np.shape(np.array(data))[1] != self.num_detect:  #check that data is defined for each detectors
			raise ValueError, "data must be defined for each of the detectors"
		self.data = np.array(data)
		
		#do checks on nside_exp
		self.nside = 2**opts.nside_exp
				
	def h_ML(self, theta, phi, psi=0.):
		"""
		Calculates h_ML (maximum likelihood estimator) at a given set of angular coordinates.
		Returns a 2-D array (frequency x polarization).
		Theta, phi, and psi are angular coordinates (in radians).
		"""
		
		if (theta < 0.) or (theta > np.pi):  #check value of theta
			raise ValueError, "theta must be between 0 and pi"
		if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
			raise ValueError, "phi must be between 0 and 2*pi"
		if (psi < 0.) or (psi > 2.*np.pi):  #check value of psi
			raise ValueError, "psi must be between 0 and 2*pi"
		
		h_ML = np.zeros((self.len_freqs, self.num_pol)) #initialize h_ML as *zero* 2-D array (frequency x polarization)
		B = self.network.B(theta=theta, phi=phi, psi=psi, freqs=self.freqs) #should be a 3-D array (frequency x polarization x detector)
		for f in np.arange(len(self.freqs)):
			for j in np.arange(self.num_pol):
				for k in np.arange(self.num_pol):
					for beta in np.arange(self.num_detect):
						h_ML[f][j] += inA[f][j][k] * B[f][k][beta] * self.data[f][beta]
		
		return h_ML
		 	
	def posterior_weight(self, theta, phi, psi=0.):
		"""
		Calculates posterior weight (i.e. unnormalized posterior value) at a set of angular coordinates
		Theta, phi, and psi are angular coordinates (in radians).
		"""
		
		if (theta < 0.) or (theta > np.pi):  #check value of theta
			raise ValueError, "theta must be between 0 and pi"
		if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
			raise ValueError, "phi must be between 0 and 2*pi"
		if (psi < 0.) or (psi > 2.*np.pi):  #check value of psi
			raise ValueError, "psi must be between 0 and 2*pi"
				
		#A
		A = self.network.A(theta=theta, phi=phi, psi=psi) #should be a 3-D array (frequency x polarization x polarization)
		for i in np.arange(self.len_freqs):
			if linalg.det(A[i]) == 0:  #check that A is invertible at each frequency
				raise ValueError, "Currently A must be invertible at each frequency"
		inA = linalg.inv(A)
		
		#h_ML
		h_ML = self.h_ML(theta=theta, phi=phi, psi=psi)		
		h_ML_conj = np.conj(h_ML)
	
		#I
		Z = self.hPrior.incovariance #should be 3-D array (frequency * polarization * polarization)
		if np.shape(A) != np.shape(Z) #check that dimensions of A and Z are equal
			raise ValueError, "Dimensions of A and Z must be equal"
		I = A + Z
		for i in np.arange(self.len_freqs):
			if linalg.det(I[i]) == 0:  #check that I is invertible at each frequency
				raise ValueError, "Currently I must be invertible at each frequency"
		inI = linalg.inv(I)
		
		#means
		means = self.hPrior.means #should be 2-D array (frequency * polarization)
		means_conj = np.conj(means)
		
		#posterior_weight
		posterior_weight = 1.
		for f in np.arange(len(self.freqs)):
			exponent = 0.
			for j in np.arange(self.num_pol):
				for k in np.arange(self.num_pol):
					for m in np.arange(self.num_pol):
						for n in np.arange(self.num_pol):
							exponent += h_ML_conj[f][j]*A[f][j][k]*h_ML[f][k] - ( h_ML_conj[f][j] - means[f][j] )*( Z[f][j][k] - Z[f][j][m]*inI[f][m][n]*Z[f][n][k] )*( h_ML[f][k] + means[f][k] )
			
			exponent *= self.delta_f  #now have value of exponent at f
			posterior_weight *= self.hPrior.amplitudes[f]*np.exp(exponent) * np.sqrt( pow(2.*np.pi, self.num_pol) * linalg.det(inI[f]) )
		
		posterior_weight *= self.angPrior.angular_prior(theta, phi)  #multiply by angular prior to give unnormalized posterior value
		return posterior_weight

	def build_posterior(self)
		"""
		"""
		npix = hp.nside2npix(self.nside)  #number of pixels
		pixarray = np.zeros((npix,3))  #initializes a 2-D array (pixel x (ipix, theta, phi))
		for ipix in np.arange(npix):
			theta, phi = hp.pix2ang(self.nside, ipix)
			pixarray[ipix,:] = np.array([ipix, theta, phi])
		
		posterior = np.zeros((npix,2)) # initalizes 2-D array (pixel x (ipix, posterior weight))
		for ipix in pixarray[:,0]
			posterior[ipix,0] = pixarray[ipix,0]  #assigns pixel indices
			posterior[ipix,1] = self.posterior_weight(theta=pixarray[ipix,1], phi=pixarray[ipix,2]) #assigns posterior weights
		
		posterior[:,1] /= sum(posterior[:,1]) #normalizes posterior
		
		return posterior
		
	def plot_posterior(self, posterior, figname, title=None, unit=None, est_theta=None, est_phi=None, inj_theta=None, inj_phi=None)
		"""
		"""
		
		#checks on posterior, title, unit
		pixarea = hp.nside2pixarea(self.nside)  #area of each pixel
		
		fig = plt.figure()
		hp.mollview(posterior[:,1]/pixarea, title=title, unit=unit, flip="geo", min=0.0)
		ax = fig.gca()
		hp.graticule()
		
		#checks on est and inj angles
		est_marker = ax.projplot((est_theta, est_phi), "wo", markeredgecolor="w", markerfacecolor="none")[0]
		est_marker.set_markersize(10)
		est_marker.set_markeredgewidth(2)
		inj_marker = ax.projplot((inj_theta, inj_phi), "wx")[0]
		inj_marker.set_markersize(10)
		inj_marker.set_markeredgewidth(2)
		
		#checks on figname
		fig.savefig(figname)
		plt.close(fig)
		


##########NOTES##########
#*make sure A,B given in consistent array form
