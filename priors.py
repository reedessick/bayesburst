import numpy as np
from numpy import linalg
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

#=================================================
#
#                Stored Priors
#
#=================================================


def hpri_neg4(len_freqs, num_pol):
	"""
	Build the hprior of p(h) \propto h^(-4)
	"""
	num_gaus = 16
	variances = np.logspace(start=-56,stop=-41,num=num_gaus)
	amp_powers = np.logspace(start=112,stop=82,num=num_gaus)
	
	amplitudes = 2.21807*amp_powers*np.ones((len_freqs, num_gaus))  #2-D array (frequencies x Gaussians)
	means = np.zeros((len_freqs, num_pol, num_gaus))  #3-D array (frequencies x polarizations x Gaussians)
	covariance = np.zeros((len_freqs, num_pol, num_pol, num_gaus))  #4-D array (frequencies x polarizations x polarizations x Gaussians)
	for n in xrange(num_gaus):
		for i in xrange(num_pol):
			for j in xrange(num_pol):
				if i == j:
					covariance[:,i,j,n] = variances[n]  #non-zero on polarization diagonals
	return amplitudes, means, covariance, num_gaus


#=================================================
#
#               Prior Classes
#
#=================================================

class hPrior(object):
	"""
	An object that creates the strain prior.  Because this framework is an excercise
	in Gaussian integral, we will assume that the priors can be decomposed into
	a series of Gaussians.
	"""
	
	#maybe add function that calculates full prior over all freqs
	
	###
	def __init__(self, freqs, means, covariance, amplitudes=1., num_gaus=1, num_pol=2):
		"""
		Priors are assumed to have form \sum_over N{
		C_N(f) * exp( - ( h_k(f) - mean_k(f)_N )^* Z_kj(f)_N ( h_j(f) - mean_j(f)_N ) ) }
		Thus:
			*freqs is a 1-D array
			*means is a 3-D array, (frequencies x polarizations x Gaussians)
			*covariance is a 4-D array, (frequencies x polarizations x polarizations x Gaussians)
			*amplitudes is a 2-D array, (frequencies x Gaussians)
		N.b that values must be specified over relevant frequency range
		"""
		
		#Initialize frequencies
		self.len_freqs = len(np.array(freqs))
		if not self.len_freqs:
			raise ValueError, "freqs must have at least 1 entry"
		self.freqs = np.array(freqs)
		
		#Initialize polarizations
		if not num_pol:
			raise ValueError, "Must specify at least one polarization"
		self.num_pol = num_pol
		
		#Initialize amplitudes
		self.num_gaus = num_gaus
		if type(amplitudes) == int or type(amplitudes) == float:
			self.amplitudes = amplitudes*np.ones((self.len_freqs, self.num_gaus))  #2-D array (frequencies x Gaussians)
		else:
			if np.shape(np.array(amplitudes))[0] != self.len_freqs:  #make sure amplitude defined at every frequency
				raise ValueError, "Freqs and amplitudes must have the same length"
			if np.shape(np.array(amplitudes))[1] != self.num_gaus:  #make sure amplitude defined for every Gaussian
				raise ValueError, "Must define amplitude for every Gaussian"
			self.amplitudes = np.array(amplitudes)  #2-D array (frequencies x Gaussians)
		
		#Initialize prior means
		if type(means) == int or type(means) == float:
			self.means = means*np.ones((self.len_freqs, self.num_pol, self.num_gaus))  #3-D array (frequencies x polarizations x # Gaussians)
		else:
			if np.shape(np.array(means))[0] != self.len_freqs:  #make sure mean vector specified at each frequency
				raise ValueError, "Freqs and means must have the same length"
			if np.shape(np.array(means))[1] != self.num_pol: #make sure mean specified for each polarization
				raise ValueError, "Must specify means for correct number of polarizations"
			if np.shape(np.array(means))[2] != self.num_gaus:  #make sure mean defined for every Gaussian
				raise ValueError, "Must define mean for every Gaussian"
			self.means = np.array(means)  #3-D array (frequencies x polarizations x # Gaussians)
		
		#Initialize prior covariance	
		if type(covariance) == int or type(covariance) == float:
			self.covariance = np.zeros((self.len_freqs, self.num_pol, self.num_pol, self.num_gaus))  #4-D array (frequencies x polarizations x polarizations x Gaussians)
			for i in xrange(self.num_pol):
				for j in xrange(self.num_pol):
					if i == j:
						self.covariance[:,i,j,:] = covariance  #covariance on polarization diagonals		
		else:
			if np.shape(covariance)[0] != self.len_freqs:  #make sure covariance matrix defined at every frequency
				raise ValueError, "Covariance matrices must have correct frequency length"
			if (np.shape(covariance)[1] != self.num_pol) or (np.shape(covariance)[2] != self.num_pol):  #make sure covariance matrices are square in correct # of polarizations
				raise ValueError, "At a given frequency, covariance matrix must be square in correct # of polarizations"
			if np.shape(covariance)[3] != self.num_gaus:  #make sure covariance defined for every Gaussian
				raise ValueError, "Must define covariance for every Gaussian"
			self.covariance = np.array(covariance)  #4-D array (frequencies x polarizations x polarizations x Gaussians)
			
		self.incovariance = np.zeros((self.len_freqs, self.num_pol, self.num_pol, self.num_gaus))  #4-D array (frequencies x polarizations x polarizations x Gaussians)	
		for n in xrange(self.num_gaus):
			self.incovariance[:,:,:,n] = linalg.inv(self.covariance[:,:,:,n])
	
	###	
	def f2i(self, f):
		"""
		takes frequency and returns index
		"""
		i = np.where(self.freqs == f)
		if i[0].tolist() == []:
			raise ValueError, "Specified frequency not contained in frequency array"
		return i[0][0]
	
	###
	def i2f(self, i):
		"""
		takes index and returns frequency
		"""
		return self.freqs[i]
		
	###	
	def prior_weight_f(self, h, f):
		"""
		returns value of prior weight (i.e. unnormalized hPrior value) for a strain vector (defined
		for each polarization) for a given freqiency
		"""
		
		#Initialize strain at f
		if type(h) == int or type(h) == float:  #make sure h not given as single number
			raise ValueError, "Please define h as either list or array over polarizations at f"
		if np.shape(np.array(h))[0] != self.num_pol:  #make sure h defined at every polarization
			raise ValueError, "Must define value of h for each polarization"
		h = np.array(h)  #1-D array (polarizations)
		
		#Find index of specified frequency
		f_in = f2i(f)
		
		weight = 0.
		for n in xrange(self.num_gaus):
			#Calculate matrix multiplication components at f
			displacement = h - self.means[f_in,:,n]
			displacement_conj = np.conj(displacement)
			matrix = self.incovariance[f_in,:,:,n]
		
			#Calculate prior weight through matrix multiplication
			exponent_n = 0.  #exponential term of Gaussian
			for i in xrange(self.num_pol):
				for j in xrange(self.num_pol):
					exponent_n += -displacement_conj[i] * matrix[i,j] * displacement[j]
			weight += self.amplitudes[f_in,n]*np.exp(exponent_n)
			
		return weight
	 

class angPrior(object):
	"""
	An object that creates the sky position prior in angular coordinates.
	"""
	
	#right now the default option is uniform over the sky
	#will want to add some sort of beam pattern option
	#eventually should add galaxy catalogues option
	
	###
	def __init__(self, nside_exp, prior_opt='uniform'):
		"""
		*Initializes an angPrior of type prior_opt
		*nside = 2**nside_exp, nside_exp must be int
		*Currently only supports uniform priors
		"""
		#Initialize nside (sky pixelization)
		if type(nside_exp) != int:
			raise ValueError, "nside_exp must be an integer"
		self.nside = 2**nside_exp
		
		#Initialize prior type
		if prior_opt != 'uniform':
			raise ValueError, "Currently only compatible with uniform angPrior option"
		self.prior_opt = prior_opt
	
	###
	def prior_weight(self, theta, phi):
		"""
		*Calculate the prior weight (i.e. the unnormalized angPrior value)
		at a given set of angular coordinates.
		*Theta and phi are angular coordinates given in radians
		"""
		
		#Perform checks on angular coordinates
		if (theta < 0.) or (theta > np.pi):  #check value of theta
			raise ValueError, "theta must be between 0 and pi"
		if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
			raise ValueError, "phi must be between 0 and 2*pi"
		
		#Calculate prior_weight depending on what prior_opt is being used
		if self.prior_opt == 'uniform':
			return 1.
		else:
			raise ValueError, "Currently only compatible with uniform angPrior option"
	
	###		
	def build_prior(self):
		"""
		Builds the normalized prior over the entire sky.
		"""
		
		#Pixelate the sky
		npix = hp.nside2npix(self.nside)  #number of pixels
		pixarray = np.zeros((npix,3))  #initializes a 2-D array (pixels x (ipix, theta, phi)'s)
		for ipix in xrange(npix):
			theta, phi = hp.pix2ang(self.nside, ipix) #maps theta and phi to ipix
			pixarray[ipix,:] = np.array([ipix, theta, phi])
		
		#Build unnormalized angPrior over whole sky
		angprior = np.zeros((npix,2)) #initalizes 2-D array (pixels x (ipix, angPrior weight)'s)
		for ipix in pixarray[:,0]:
			angprior[ipix,0] = pixarray[ipix,0]  #assigns pixel indices
			angprior[ipix,1] = self.prior_weight(theta=pixarray[ipix,1], phi=pixarray[ipix,2]) #assigns prior weights
		
		#Normalize the prior
		angprior[:,1] /= sum(angprior[:,1])
		
		return angprior
	
	###	
	def plot_angprior(self, angprior, figname, title=None, unit=None, inj_theta=None, inj_phi=None):
		"""
		*Plots a skymap of a given angPrior
		"""
		
		#Checks on angprior, title, unit
		#~~~~~~~~~~~
		
		#Calculate area of each pixel
		pixarea = hp.nside2pixarea(self.nside)  #area of each pixel
		
		#Plot angPrior probability density function as skymap
		fig = plt.figure(0)
		hp.mollview(angprior[:,1]/pixarea, fig=0, title=title, unit=unit, flip="geo", min=0.0)
		ax = fig.gca()
		hp.graticule()
		
		#Plot injected location of event, if given
		if inj_theta or inj_phi:
			if (not inj_theta) or (not inj_phi):  #check that both inj_theta and inj_phi are given
				raise ValueError, "If plotting injected event, must specify both inj_theta and inj_phi"
			if (inj_theta < 0.) or (inj_theta > np.pi):  #check value of inj_theta
				raise ValueError, "inj_theta must be between 0 and pi"
			if (inj_phi < 0.) or (inj_phi > 2.*np.pi):  #check value of inj_phi
				raise ValueError, "inj_phi must be between 0 and 2*pi"
			inj_marker = ax.projplot((inj_theta, inj_phi), "wx")[0]
			inj_marker.set_markersize(10)
			inj_marker.set_markeredgewidth(2)
		
		#Save figure
		#~~~checks on figname
		fig.savefig(figname)
		plt.close(fig)
