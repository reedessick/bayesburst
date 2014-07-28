usage="""a module storing the methods to build priors decomposed into gaussian terms used in the analytic marginalization over all possible signals"""

#=================================================

import numpy as np
np.linalg = linalg
import healpy as hp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex":True})

#=================================================
#
#               Prior Classes
#
#=================================================
#================================================
#  prior on strain
#=================================================
class hPrior(object):
	"""
	An object representing the prior on strain.  
	We analytically marginalize over all possible signals with gaussian integrals, and therefore decompose the prior into a sum of gaussians.
	This appears to work well for Pareto distributions with lower bounds.
	"""
	
	#maybe add function that calculates full prior over all freqs
	
	###
	def __init__(self, freqs, means, covariance, amplitudes=1., n_gaus=1, n_pol=2):
		"""
		Priors are assumed to have form \sum_over N{
		C_N(f) * exp( - conj( h_k(f) - mean_k(f)_N ) * Z_kj(f)_N * ( h_j(f) - mean_j(f)_N ) ) }

		We require:
			*freqs is a 1-D array
				np.shape(freqs) = (num_freqs,)
			*means is a 3-D array or a scalar
				np.shape(means) = (num_freqs, num_pol, num_gaus)
			*covariance is a 4-D array or a scalar
				np.shape(covariance) = (num_freqs, num_pol, num_pol, num_gaus)
			*amplitudes is a 1-D array or a scalar
				np.shape(amplitudes) = (num_gaus,)
			
			if any of the above are a scalar (except freqs, which must be an array), they are cast to the correct shape. 
			If not enough information is provided to determine the shape of these arrays, we default to the optional arguments
				n_gaus, n_pol
			otherwise these are ignored
		"""
		
		### check frequencies
		if not isinstance(freqs, (np.ndarray)):
			freqs = np.array(freqs)
		if len(np.shape(freqs)) != 1:
			raise ValueError, "bad shape for freqs"
		n_freqs = len(freqs)
		if not n_freqs:
			raise ValueError, "freqs must have at least 1 entry"

		### check means and covariance
		scalar_means = isinstance(means, (int, float))
		scalar_covar = isinstance(means, (int,float))

		if scalar_means and scalar_covar: ### both are scalars
			### build means
			means = means * np.ones((n_freqs, n_pol, n_gaus), float)
			### build covariance
			cov = covariance
			covariance = np.zeros((n_freqs, n_pol, n_pol, n_gaus))
                        for i in xrange(self.num_pol):  ### fill in diagonals for n_pol polarizations
				covariance[:,i,i,:] = cov

		elif scalar_covar: ### means is an array, covariance is a scalar
			### check means
			if not isinstance(means, np.ndarray):
                                means = np.array(means)
                        if len(np.shape(means)) != 3:
                                raise ValueError, "bad shape for means"
                        n_f, n_pol, n_gaus = np.shape(means)
                        if n_freqs != n_f:
                                raise ValueError, "shape mismatch between freqs and means"
                        if n_pol <= 0:
                                raise ValueError, "must have a positive definite number of polarizations"

			### build covariance
			c = covariance
			covariance = np.zeros((n_freqs, n_pol, n_pol, n_gaus),float)
			for i in xrange(n_pol):
				covariance[:,i,i,:] = cov

		elif scalar_means: ### covariance is an array, means is a scalar
			### check covariance
			if not isinstance(covariance, np.ndarray):
				covariance = np.array(covariance)
			if len(np.shape(covariance)) != 4:
				raise ValueError, "bad shape for covariance"
			n_f, n_pol, n_p, n_gaus = np.shape(covariance)
			if n_freqs != n_f:
				raise ValueError, "shape mismatch between freqs and covariance"
			if n_pol != n_p:
				raise ValueError, "inconsistent shape within covariance"
			### build means
			means = means * np.ones((n_freqs, n_pol, n_gaus), float)

		else: ### both are arrays
                        ### check means
                        if not isinstance(means, np.ndarray):
                                means = np.array(means)
                        if len(np.shape(means)) != 3:
                                raise ValueError, "bad shape for means"
                        n_f, n_pol, n_gaus = np.shape(means)
                        if n_freqs != n_f:
                                raise ValueError, "shape mismatch between freqs and means"
                        if n_pol <= 0:
                                raise ValueError, "must have a positive definite number of polarizations"
                        ### check covariance
                        if not isinstance(covariance, np.ndarray):
                                covariance = np.array(covariance)
                        if len(np.shape(covariance)) != 4:
                                raise ValueError, "bad shape for covariance"
                        n_f, n_p1, n_p2, n_g = np.shape(covariance)
                        if n_freqs != n_f:
                                raise ValueError, "shape mismatch between freqs and covariance"
                        if n_p1 != n_p2:
                                raise ValueError, "inconsistent shape within covariance"
			if (n_p1 != n_pol) or (n_g != n_gaus):
				raise ValueError, "shape mismatch between means and covariance"

                ### set up the inverse covariance 
                incovariance = np.zeros_like(covariance, dtype=float)
                for n in xrange(n_gaus):
                        incovariance[:,:,:,n] = linalg.inv(covariance[:,:,:,n])

                ### set up and store amplitudes
                if isinstance(amplitudes, (int, float)):
                        amplitudes = amplitudes * np.ones((n_gaus,), float)
                else:
			if not isinstance(amplitudes, np.ndarray):
				amplitudes = np.array(amplitudes)
			if len(np.shape(amplitudes)) != 1:
				raise ValueError, "bad shape for amplitudes"
                        if len(amplitudes) != n_gaus:  #make sure amplitude defined for every Gaussian
                                raise ValueError, "shape mismatch between amplitudes and means"

		### store basic data about decomposition
		self.n_freqs = n_freqs
		self.n_pol = n_pol
		self.n_gaus = n_gaus

		### store all arrays
		self.freqs = freqs
		self.means = means
		self.covariance = covariance
		self.incovariance = incovariance
		self.amplitudes = amplitudes

	###
	def __call__(self, h):
		"""
		evaluates the prior for the strain "h"

                We require:
			*h is a 2-D array
				np.shape(h) = (self.n_freqs, self.n_pol)
			if h is a 1-D array, we check to see if the shape matches either n_freqs or n_pol. 
				if it does, we broadcast it to the correct 2-D array
			if h is a scalar, we broadcast it to the correct 2-D array
		"""

		### make sure h has the expected shape
		if isinstance(h, (int, float): ### h is a scalar
			h = h * np.ones((self.n_freqs, self.n_pol), float)
		elif not isinstance(h, np.ndarray):
			h = np.ndarray

		h_shape = np.shape(h)
		nD = len(h_shape)
		if nD == 1: ### h is a 1-D array
			len_h = len(h)
			if len_h == self.n_pol: ### broadcast to n_freq x n_pol
				h = np.outer(np.ones((self.n_freqs,),float), h)
			elif len_h == self.n_freqs: ### broadcast to n_freq x n_pol
				h = np.outer(h, np.ones((self.n_pol,),float))
			else:
				raise ValueError, "bad shape for h"

		elif nD == 2: ### h is a 2-D array
			if (self.n_freqs, self.n_pol) != h_shape:
				raise ValueError, "bad shape for h"
		else:
			raise ValueError, "bad shape for h"

		### compute prior evaluated for this strain
		p = 0.0
		for n in xrange(self.n_gaus): ### sum over all gaussian terms
			d = h - self.means[:,:,n] ### difference from mean values
			dc = np.conj(d)
			m = self.incovariance[:,:,:,n] ### covariance matricies

			### compute exponential term			
			e = np.zeros_like(self.freqs, float)
			for i in xrange(self.n_pol): ### sum over all polarizations
				for j in xrange(self.n_pol):
					e -= dc[:,i] * m[:,i,j] * d[:,j]

			### add to total prior
			p += self.amplitudes[n] * np.exp( np.sum(e) )

		return p

	'''
	###
	def prior_weight(self, h, freqs, Nbins):
		"""
		returns value of prior weight (i.e. unnormalized hPrior value) for a strain vector (defined
		for each polarization) summed over all frequencies
		"""
		
		#Initialize strain at f
		if type(h) == int or type(h) == float:  #make sure h not given as single number
			raise ValueError, "Please define h as either list or array over polarizations at f"
		if np.shape(np.array(h))[0] != self.num_pol:  #make sure h defined at every polarization
			raise ValueError, "Must define value of h for each polarization"
		h_array = np.zeros((len(freqs),self.num_pol))  #2-D array (frequency * polarizations)
		for f in xrange(len(freqs)):
			h_array[f,:] = h
		
		weight = 0.
		for n in xrange(self.num_gaus):
			#Calculate matrix multiplication components at f
			displacement = h_array - self.means[:,:,n]
			displacement_conj = np.conj(displacement)
			matrix = self.incovariance[:,:,:,n]
		
			#Calculate prior weight through matrix multiplication
			exponent_n = 0.  #exponential term of Gaussian
			for i in xrange(self.num_pol):
				for j in xrange(self.num_pol):
					exponent_n += (- displacement_conj[:,i] * matrix[:,i,j] * displacement[:,j]) * ( 2./seg_len)
			weight += self.amplitudes[n]*np.exp(np.sum(exponent_n))
			
		return weight
	 '''

#=================================================
# prior on sky location
#=================================================
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
		for tmp_theta, tmp_phi in zip(theta,phi):
			if (tmp_theta < 0.) or (tmp_theta > np.pi):  #check value of theta
				raise ValueError, "theta must be between 0 and pi"
			if (tmp_phi < 0.) or (tmp_phi > 2.*np.pi):  #check value of phi
				raise ValueError, "phi must be between 0 and 2*pi"
		
		#Calculate prior_weight depending on what prior_opt is being used
		if self.prior_opt == 'uniform':
			return np.ones(len(theta))
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

#=================================================
#
# Known priors
#
#=================================================
def hpri_neg4(len_freqs, num_pol, Nbins):
        """
        Build the hprior of p(h) \propto hhrss^(-4)
        """
        num_gaus = 4
        start_var = -45
        variances = np.power(np.logspace(start=(num_gaus/4.)*start_var,stop=(num_gaus/4.)*start_var + (num_gaus - 1.),num=num_gaus), 4./num_gaus)
        amp_powers = variances**(-2.)

        amplitudes = (8.872256/num_gaus)*amp_powers*np.ones(num_gaus)/2.22e67  #1-D array (Gaussians)
        means = np.zeros((len_freqs, num_pol, num_gaus))  #3-D array (frequencies x polarizations x Gaussians)
        covariance = np.zeros((len_freqs, num_pol, num_pol, num_gaus))  #4-D array (frequencies x polarizations x polarizations x Gaussians)
        for n in xrange(num_gaus):
                for i in xrange(num_pol):
                        for j in xrange(num_pol):
                                if i == j:
                                        covariance[:,i,j,n] = variances[n]/Nbins  #non-zero on polarization diagonals
        return amplitudes, means, covariance, num_gaus

###
def hpri_neg4(n_freqs, n_pol, n_bins, n_gaussians=4, log10_start=-45, log10_stop=-41):
        """
        Build the hprior of p(h) \propto hhrss^(-4)
        """
        variances = np.logspace(log10_start, log10_stop, n_gaussians) ### set up the variences we 

        variances = np.power(
np.logspace(start=(num_gaus/4.)*start_var,stop=(num_gaus/4.)*start_var + (num_gaus - 1.),num=num_gaus)
, 4./num_gaus)
        amp_powers = variances**(-2.)

        amplitudes = (8.872256/num_gaus)*amp_powers*np.ones(num_gaus)/2.22e67  #1-D array (Gaussians)
        means = np.zeros((len_freqs, num_pol, num_gaus))  #3-D array (frequencies x polarizations x Gaussians)
        covariance = np.zeros((len_freqs, num_pol, num_pol, num_gaus))  #4-D array (frequencies x polarizations x polarizations x Gaussians)

        for n in xrange(num_gaus):
                for i in xrange(num_pol):
                        for j in xrange(num_pol):
                                if i == j:
                                        covariance[:,i,j,n] = variances[n]/Nbins  #non-zero on polarization diagonals

        return amplitudes, means, covariance, num_gaus

