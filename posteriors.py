import numpy as np
from numpy import linalg
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#=================================================
#
#                Prior classes
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
		
		#Initialize frequencies
		len_freqs = len(np.array(freqs))
		if not len_freqs:
			raise ValueError, "freqs must have at least 1 entries"
		self.freqs = np.array(freqs)
		
		#Initialize polarizations
		if not num_pol:
			raise ValueError, "Must specify at least one polarization"
		self.num_pol = num_pol
		
		#Initialize prior means
		if type(means) == int or type(means) == float:
			self.means = np.array(len_freqs*[self.num_pol*[means]]) #2-D array (frequencies x polarizations)
		else:
			if np.shape(np.array(means))[0] != len_freqs:  #make sure mean vector specified at each frequency
				raise ValueError, "freqs and means must have the same length"
			if np.shape(np.array(means))[1] != self.num_pol: #make sure mean specified for each polarization
				raise ValueError, "must specify means for correct number of polarizations"
			self.means = np.array(means)  #2-D array (frequencies x polarizations)
		
		#Initialize prior covariance	
		if type(covariance) == int or type(covariance) == float:
			self.covariance = np.array([covariance*np.identity(self.num_pol) for i in self.freqs]) #3-D array (frequencies x polarizations x polarizations)
		else:
			if np.shape(covariance)[0] != np.shape(self.freqs)[0]:  #make sure covariance matrix defined at every frequency
				raise ValueError, "covariance matrices must have correct frequency length"
			if np.shape(covariance)[1] != np.shape(covariance)[2]:  #make sure covariance matrices are square
				raise ValueError, "At a given frequency, covariance matrix must be square"
			self.covariance = covariance  #3-D array (frequencies x polarizations x polarizations)
		
		for i in xrange(len(self.freqs)):
			if linalg.det(self.covariance[i]) == 0:  #check if covariance is invertible at each frequency
				raise ValueError, "Currently covariance must be invertible at each frequency"
		self.incovariance = linalg.inv(self.covariance)  #3-D array (frequencies x polarizations x polarizations)
		
		#Initialize amplitudes
		if type(amplitudes) == int or type(amplitudes) == float:
			self.amplitudes = np.array(len_freqs*[amplitudes])  #1-D array (frequencies)
		else:
			if np.shape(np.array(amplitudes))[0] != len_freqs:  #make sure amplitude defined at every frequency
				raise ValueError, "freqs and amplitudes must have the same length"
			self.amplitudes = np.array(amplitudes)  #1-D array (frequencies)
	
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
	def means_f(self, f):
		"""
		returns value of means at frequency f
		"""
		findex = self.f2i(f)
		return self.means[findex]
		
	###	
	def covariance_f(self, f):
		"""
		returns covariance matrix at frequency f
		"""
		findex = self.f2i(f)
		return self.covariance[findex]
		
	###	
	def incovariance_f(self, f):
		"""
		returns inverse covariance matrix at frequency f
		"""
		findex = self.f2i(f)
		return self.incovariance[findex]
		
	###	
	def amplitude_f(self, f):
		"""
		returns value of ampiltude at frequency f
		"""
		findex = self.f2i(f)
		return self.amplitudes[findex]
		
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
		
		#Calculate matrix multiplication components at f
		displacement = h - self.means_f(f)
		displacement_conj = np.conj(displacement)
		matrix = self.incovariance_f(f)
		
		#Calculate prior weight through matrix multiplication
		exponent = 0.  #exponential term of Gaussian
		for i in xrange(self.numpol):
			for j in xrange(self.num_pol):
				exponent += -displacement_conj[i] * matrix[i][j] * displacement[j]
		weight = self.amplitude_f(f)*np.exp(exponent)
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
			raise ValueError, "Currently only compatible with uniform prior option"
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
			raise ValueError, "Currently only compatible with uniform prior option"
	
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
		
		#Checks on posterior, title, unit
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
#                Posterior class
#
#=================================================

class Posterior(object):
	"""
	An object that calculates the sky localization posterior at a given
	set of angular coordinates 
	"""
	
	#write function to estimate location of the event
	
	###
	def __init__(self, freqs, network, hprior, angprior, data, nside_exp):
		"""
		*freqs is a 1-D array of frequencies
		*network is a Network object defined in utils.py
		*hprior and angprior are Prior objects, defined above
		*data is a 2-D array, (frequencies x detectors)
		*nside = 2**nside_exp, nside_exp must be int
		"""
		
		#Initalize network
		if "%s"%type(network) != "<class 'utils.Network'>":  #check that a valid network has been passed
			raise ValueError, "network must be a utils.Network object"
		self.network = network
		self.ifos = network.detectors_list()  #list of ifos in network
		self.num_detect = len(self.ifos)	#number of ifos in network
		
		#Initialize priors (both h and ang)
		if "%s"%type(hprior) != "<class 'posteriors.hPrior'>":  #check that a valid hprior has been passed
			raise ValueError, "hprior must be a posterior.hPrior object"
		if "%s"%type(angprior) != "<class 'posteriors.angPrior'>":  #check that a valid angprior has been passed
			raise ValueError, "angprior must be a posteriors.angPrior object"
		if hprior.num_pol != network.Np:  #check that number of polarizations is consistent in both network and priors
			raise ValueError, "Number of polarizations must be consistent"
		self.num_pol = hprior.num_pol  #number of polarizations
		self.hprior = hprior
		self.angprior = angprior
		
		#Intialize list of frequencies
		self.len_freqs = np.shape(np.array(freqs))[0]  #number of frequency bins
		if not self.len_freqs:
			raise ValueError, "freqs must have at least 1 entry"
		self.freqs = np.array(freqs)  #1-D array of specified frequencies
		if (self.freqs.all() != hprior.freqs.all()) or (self.freqs.all() != network.freqs.all()):  #check that same freqs are used for prior and network
			raise ValueError, "specified frequencies must match those in hprior and angprior"
		self.delta_f = (self.freqs[-1] - self.freqs[0])/(float(self.len_freqs) - 1.)  #frequency spacing

		#Initialize data
		if np.shape(np.array(data))[0] != self.len_freqs:  #check that data is specified at every frequency
			raise ValueError, "data must be specified at each of the specified frequencies"
		if np.shape(np.array(data))[1] != self.num_detect:  #check that data is defined for each detectors
			raise ValueError, "data must be defined for each of the detectors"
		self.data = np.array(data)  #2-D array (frequencies x detectors)
		
		#Initialize nside (sky pixelization)
		if type(nside_exp) != int:
			raise ValueError, "nside_exp must be an integer"
		self.nside = 2**nside_exp

	###		
	def h_ML(self, theta, phi, psi=0., Apass=None):
		"""
		*Calculates h_ML (maximum likelihood estimator of strain) at a given set of angular coordinates.
		*Returns a 2-D array (frequencies x polarizations).
		*Theta, phi, and psi are angular coordinates (in radians).
		*A is a 3-D Network matrix (freqs x pols x pols), can be passed into this function to avoid recomputation
		"""
		
		#ALSO PASS inA !!!!!###
		
		#Perform checks on the angular coordinates
		if (theta < 0.) or (theta > np.pi):  #check value of theta
			raise ValueError, "theta must be between 0 and pi"
		if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
			raise ValueError, "phi must be between 0 and 2*pi"
		if (psi < 0.) or (psi > 2.*np.pi):  #check value of psi
			raise ValueError, "psi must be between 0 and 2*pi"
			
		#Initialize A
		if Apass != None:
			if np.shape(Apass)[0] != self.len_freqs:  #check that Apass defined at each freq
				raise ValueError, "Apass must be defined at each frequency"
			if (np.shape(Apass)[1] != self.num_pol) or (np.shape(Apass)[2] != self.num_pol):  #check that Apass has correct polarization shape
				raise ValueError, "Apass matrices must have correct polarization shape"
			A = Apass
			for i in np.arange(self.len_freqs):
				if linalg.det(A[i]) == 0:  #check that A is invertible at each frequency
					raise ValueError, "Currently A must be invertible at each frequency"
			inA = linalg.inv(A)
		else:
			A = self.network.A(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x polarizations)
			for i in np.arange(self.len_freqs):
				if linalg.det(A[i]) == 0:  #check that A is invertible at each frequency
					raise ValueError, "Currently A must be invertible at each frequency"
			inA = linalg.inv(A)
			
		#Initialize B
		B = self.network.B(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x detectors)
		
		#Calculate h_ML
		h_ML = np.zeros((self.len_freqs, self.num_pol), 'complex')  #initialize h_ML as 2-D array (frequencies x polarizations)
		for fi in xrange(self.len_freqs):
			for j in xrange(self.num_pol):
				for k in xrange(self.num_pol):
					for beta in xrange(self.num_detect):
						h_ML[fi][j] += inA[fi][j][k] * B[fi][k][beta] * self.data[fi][beta]
		
		return h_ML
	
	### 	
	def log_posterior_weight(self, theta, phi, psi=0.):
		"""
		*Calculates posterior weight (i.e. unnormalized posterior value) at a set of angular coordinates
		*Theta, phi, and psi are angular coordinates (in radians).
		"""
		
		#Perform checks on angular coordinates
		if (theta < 0.) or (theta > np.pi):  #check value of theta
			raise ValueError, "theta must be between 0 and pi"
		if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
			raise ValueError, "phi must be between 0 and 2*pi"
		if (psi < 0.) or (psi > 2.*np.pi):  #check value of psi
			raise ValueError, "psi must be between 0 and 2*pi"
				
		#Calculate A
		A = self.network.A(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x polarizations)
		for i in range(self.len_freqs):
			if linalg.det(A[i]) == 0:  #check that A is invertible at each frequency
				raise ValueError, "Currently A must be invertible at each frequency"
		inA = linalg.inv(A)
		
		#Calculate h_ML (maximum likelihood strain estimator)
		h_ML = self.h_ML(theta=theta, phi=phi, psi=psi, Apass=A)		
		h_ML_conj = np.conj(h_ML)
	
		#Calculate P = A + Z
		Z = self.hprior.incovariance  #3-D array (frequencies * polarizations * polarizations)
		if np.shape(A) != np.shape(Z): #check that dimensions of A and Z are equal
			raise ValueError, "Dimensions of A and Z must be equal"
		P = A + Z
		for i in xrange(self.len_freqs):
			if linalg.det(P[i]) == 0:  #check that P is invertible at each frequency
				raise ValueError, "Currently P must be invertible at each frequency"
		inP = linalg.inv(P)
		
		#Get hprior means
		means = self.hprior.means  #2-D array (frequency * polarization)
		means_conj = np.conj(means)
		
		#Calculate posterior_weight
		#posterior_weight = 1.
		log_posterior_weight = 0.
		for f in xrange(len(self.freqs)):
			term1_jk = 0.
			term2_jk = 0.
			term3_jmnk = 0.
			
			for j in xrange(self.num_pol):
				vec_j = h_ML_conj[f][j] - means_conj[f][j]
				
				for k in xrange(self.num_pol):
					vec_k = h_ML[f][k] - means[f][k]
					
					term1_jk += h_ML_conj[f][j]*A[f][j][k]*h_ML[f][k]
					term2_jk += vec_j * Z[f][j][k] * vec_k
					
					for m in xrange(self.num_pol):
						for n in xrange(self.num_pol):
							term3_jmnk +=  vec_j * Z[f][j][m] * inP[f][m][n] * Z[f][n][k] * vec_k
				
			exponent = self.delta_f*( term1_jk - term2_jk + term3_jmnk ) #exponential term for a given frequency
			log_posterior_weight += np.log(self.hprior.amplitudes[f]) + exponent + np.log( np.sqrt( pow(2.*np.pi, self.num_pol) * linalg.det(inP[f]) ) )
			
		log_posterior_weight += np.log(self.angprior.prior_weight(theta=theta, phi=phi))
		return log_posterior_weight
	
	###
	def build_posterior(self):
		"""
		Builds the normalized posterior over the entire sky.
		"""
		
		#Pixelate the sky
		npix = hp.nside2npix(self.nside)  #number of pixels
		pixarray = np.zeros((npix,3))  #initializes a 2-D array (pixels x (ipix, theta, phi)'s)
		for ipix in xrange(npix):
			theta, phi = hp.pix2ang(self.nside, ipix)  #maps theta and phi to ipix
			pixarray[ipix,:] = np.array([ipix, theta, phi])
		
		#Build unnormalized log posterior over whole sky
		posterior = np.zeros((npix,2), 'complex') # initalizes 2-D array (pixels x (ipix, posterior weight)'s)
		for ipix in pixarray[:,0]:
			posterior[ipix,0] = pixarray[ipix,0]  #assigns pixel indices
			posterior[ipix,1] = self.log_posterior_weight(theta=pixarray[ipix,1], phi=pixarray[ipix,2]) #assigns posterior weights
		
		#Find max log posterior value and subtract it from all log posterior values (to make exponentiation feasible)
		max_log_pos = max(posterior[:,1])
		posterior[:,1] -= max_log_pos
		
		#Convert unnormalized log posterior to actual unormalized posterior
		posterior[:,1] = np.exp(posterior[:,1])
		
		#Normalize posterior
		posterior[:,1] /= sum(posterior[:,1]) #normalizes posterior
		
		return posterior
	
	###	
	def plot_posterior(self, posterior, figname, title=None, unit=None, est_theta=None, est_phi=None, inj_theta=None, inj_phi=None):
		"""
		*Plots a skymap of a given posterior
		"""
		
		#Checks on posterior, title, unit
		#~~~~~~~~~~~
		
		#Calculate area of each pixel
		pixarea = hp.nside2pixarea(self.nside)  #area of each pixel
		
		#Plot posterior probability density function as skymap
		fig = plt.figure(0)
		hp.mollview(np.log10(posterior[:,1]/pixarea), fig=0, title=title, unit=unit, flip="geo", min=None, max=None)
		ax = fig.gca()
		hp.graticule()
		
		#Plot injected location of event, if given
		if (inj_theta != None) or (inj_phi != None):
			if (inj_theta == None) or (inj_phi == None):  #check that both inj_theta and inj_phi are given
				raise ValueError, "If plotting injected event, must specify both inj_theta and inj_phi"
			if (inj_theta < 0.) or (inj_theta > np.pi):  #check value of inj_theta
				raise ValueError, "inj_theta must be between 0 and pi"
			if (inj_phi < 0.) or (inj_phi > 2.*np.pi):  #check value of inj_phi
				raise ValueError, "inj_phi must be between 0 and 2*pi"
			inj_marker = ax.projplot((inj_theta, inj_phi), "wx")[0]
			inj_marker.set_markersize(10)
			inj_marker.set_markeredgewidth(2)
		
		#Plot estimated location of event, if given
		if (est_theta != None) or (est_phi != None):
			if (est_theta == None) or (est_phi == None):  #check that both est_theta and est_phi are given
				raise ValueError, "If plotting estimated event, must specify both est_theta and est_phi"
			if (est_theta < 0.) or (est_theta > np.pi):  #check value of est_theta
				raise ValueError, "est_theta must be between 0 and pi"
			if (est_phi < 0.) or (est_phi > 2.*np.pi):  #check value of est_phi
				raise ValueError, "est_phi must be between 0 and 2*pi"
			est_marker = ax.projplot((est_theta, est_phi), "wo", markeredgecolor="w", markerfacecolor="none")[0]
			est_marker.set_markersize(10)
			est_marker.set_markeredgewidth(2)

		#checks on figname
		#~~~checks on figname
		fig.savefig(figname)
		plt.close(fig)
		


##########NOTES##########
#test everything

