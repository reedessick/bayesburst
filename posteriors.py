import numpy as np
from numpy import linalg
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

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
	def __init__(self, freqs, network, hprior, angprior, data, nside_exp, Nbins):
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
		if "%s"%type(hprior) != "<class 'priors.hPrior'>":  #check that a valid hprior has been passed
			raise ValueError, "hprior must be a priors.hPrior object"
		if "%s"%type(angprior) != "<class 'priors.angPrior'>":  #check that a valid angprior has been passed
			raise ValueError, "angprior must be a priors.angPrior object"
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
		self.Nbins = Nbins

		#Initialize data
		if np.shape(np.array(data))[0] != self.len_freqs:  #check that data is specified at every frequency
			raise ValueError, "data must be specified at each of the specified frequencies"
		if np.shape(np.array(data))[1] != self.num_detect:  #check that data is defined for each detectors
			raise ValueError, "data must be defined for each of the detectors"
		self.data = np.array(data)  #2-D array (frequencies x detectors)
		
		#Initialize sky pixelization
		if type(nside_exp) != int:
			raise ValueError, "nside_exp must be an integer"
		self.nside = 2**nside_exp
		self.npix = hp.nside2npix(self.nside)  #number of pixels

	###		
	def h_ML(self, theta, phi, psi=0., Apass=None, inApass=None, Bpass=None, eff_pol=None):
		"""
		*Calculates h_ML (maximum likelihood estimator of strain) at a given set of angular coordinates.
		*Returns a 2-D array (frequencies x polarizations).
		*Theta, phi, and psi are angular coordinates (in radians).
		*A is a 3-D Network matrix (freqs x pols x pols), can be passed into this function to avoid recomputation
		"""
		
		#Perform checks on the angular coordinates
		if (theta < 0.) or (theta > np.pi):  #check value of theta
			raise ValueError, "theta must be between 0 and pi"
		if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
			raise ValueError, "phi must be between 0 and 2*pi"
		if (psi < 0.) or (psi > 2.*np.pi):  #check value of psi
			raise ValueError, "psi must be between 0 and 2*pi"
			
		#Establish effective number of polarizations
		if (eff_pol == None):
			eff_pol = self.num_pol
			
		#Initialize A
		if (Apass != None) and (inApass != None):
			if np.shape(Apass)[0] != self.len_freqs:  #check that Apass defined at each freq
				raise ValueError, "Apass must be defined at each frequency"
			if (np.shape(Apass)[1] != eff_pol) or (np.shape(Apass)[2] != eff_pol):  #check that Apass has correct effective polarization shape
				raise ValueError, "Apass matrices must have correct effective polarization shape"
			if np.shape(inApass)[0] != self.len_freqs:  #check that inApass defined at each freq
				raise ValueError, "inApass must be defined at each frequency"
			if (np.shape(inApass)[1] != eff_pol) or (np.shape(inApass)[2] != eff_pol):  #check that inApass has correct effective polarization shape
				raise ValueError, "inApass matrices must have correct effective polarization shape"
			A = Apass
			inA = inApass
		else:
			A = self.network.A(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x polarizations)
			inA = linalg.inv(A)
			
		#Initialize B
		if (Bpass != None):
			B = Bpass
		else:
			B = self.network.B(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x detectors)
		
		#Calculate h_ML
		h_ML = np.zeros((self.len_freqs, eff_pol), 'complex')  #initialize h_ML as 2-D array (frequencies x polarizations)
		for j in xrange(eff_pol):
			for k in xrange(eff_pol):
				for beta in xrange(self.num_detect):
					h_ML[:,j] += inA[:,j,k] * B[:,k,beta] * self.data[:,beta]
		
		return h_ML
	
	### 	
	def log_gaussian_data(self, thetas, phis, psi=0., connection=None, threshold=0.):
		"""
		*Calculates log of Gaussian terms for each frequency at a set of angular coordinates
		*Theta, phi, and psi are angular coordinates (in radians).
		"""
		
		if connection == None:
			raise ValueError, "Must provide mp connection"
		
		thetas = np.array(thetas)
		phis = np.array(phis)
		
		if len(thetas) != len(phis):
			raise ValueError, "Must give same number of thetas and phis"
		
		npix = len(thetas)
		pix_ind = 0
		g_array = np.zeros((npix, self.hprior.num_gaus, self.len_freqs))  #3-D array (pixels * Gaussian terms * frequencies)
		
		for theta, phi in zip(thetas,phis):
			
			#Perform checks on angular coordinates
			if (theta < 0.) or (theta > np.pi):  #check value of theta
				raise ValueError, "theta must be between 0 and pi"
			if (phi < 0.) or (phi > 2.*np.pi):  #check value of phi
				raise ValueError, "phi must be between 0 and 2*pi"
			if (psi < 0.) or (psi > 2.*np.pi):  #check value of psi
				raise ValueError, "psi must be between 0 and 2*pi"
					
			#Calculate A and B
			A = self.network.A(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x polarizations)
			inA = linalg.inv(A)
			B = self.network.B(theta=theta, phi=phi, psi=psi, no_psd=False)  #3-D array (frequencies x polarizations x detectors)
			
			#Calculate eigenvalues of A, apply effective number of polarizations
			eig_vals, eig_vecs = linalg.eigh(A)  #sorted by eigenvalue (small to large), note A is Hermitian
			
			num_pol_eff = self.num_pol
			if any(eig_vals[:,1]/eig_vals[:,0] >= threshold):
				
				num_pol_eff -= 1
									
				R = eig_vecs  #Rotation matrix to convert to DPF
				inR = linalg.inv(R)
				
				A = np.zeros((self.len_freqs, num_pol_eff, num_pol_eff))
				for i_eff in xrange(num_pol_eff):
					A[:,i_eff,i_eff] = eig_vals[:,i_eff+1]
				inA = linalg.inv(A)

				B_tmp = np.zeros((self.len_freqs, self.num_pol, self.num_detect), 'complex')
				for i in xrange(self.num_pol):
					for j in xrange(self.num_pol):
						for beta in xrange(self.num_detect):
							B_tmp[:,i,beta] += B[:,j,beta]*inR[:,j,i]
				B = B_tmp
				B = np.delete(B,0,1)
			
			#Calculate h_ML (maximum likelihood strain estimator)
			h_ML = self.h_ML(theta=theta, phi=phi, psi=psi, Apass=A, inApass=inA, Bpass=B, eff_pol=num_pol_eff)		
			h_ML_conj = np.conj(h_ML)
			
			#Calculate inP = (A + Z)^(-1)
			Z = self.hprior.incovariance  #4-D array (frequencies * polarizations * polarizations * Gaussians)
			Z = Z[:,:num_pol_eff,:num_pol_eff,:]  #Reduce down to effective number of polarizations (if necessary)
			inP = np.zeros((self.len_freqs, num_pol_eff, num_pol_eff, self.hprior.num_gaus))  #4-D array (frequencies * polarizations * polarizations * Gaussians)
			for n in xrange(self.hprior.num_gaus):
				if np.shape(A) != np.shape(Z[:,:,:,n]): #check that dimensions of A and Z are equal for every Gaussian
					raise ValueError, "Dimensions of A and Z must be equal for every Gaussian"
				inP[:,:,:,n] = linalg.inv(A + Z[:,:,:,n])
			
			#Get hprior means
			means = self.hprior.means  #3-D array (frequencies * polarizations * Gaussians)
			means_conj = np.conj(means)
			
			#Calculate log posterior_weight
			for igaus in xrange(self.hprior.num_gaus):
				
				term1_jk = np.zeros(self.len_freqs, 'complex')
				term2_jk = np.zeros(self.len_freqs, 'complex')
				term3_jmnk = np.zeros(self.len_freqs, 'complex')
				
				for j in xrange(num_pol_eff):
					vec_j = h_ML_conj[:,j] - means_conj[:,j,igaus]
					
					for k in xrange(num_pol_eff):
						vec_k = h_ML[:,k] - means[:,k,igaus]
						
						term1_jk += h_ML_conj[:,j]*A[:,j,k]*h_ML[:,k]
						term2_jk += vec_j * Z[:,j,k,igaus] * vec_k
						
						for m in xrange(num_pol_eff):
							for n in xrange(num_pol_eff):
								term3_jmnk +=  vec_j * Z[:,j,m,igaus] * inP[:,m,n,igaus] * Z[:,n,k,igaus] * vec_k
		
				log_exp_f = np.real( term1_jk - term2_jk + term3_jmnk ) * np.log10(np.e) / self.Nbins  #log exponential, 1-D array (frequencies)
				log_det_f = 0.5 * np.log10( (2.*np.pi)**num_pol_eff * linalg.det(inP[:,:,:,igaus]) )  #log determinant, 1-D array (frequencies)	
					
				g_array[pix_ind, igaus,:] = log_exp_f + log_det_f  #array containing the necessary sum of logs for each pixel, Gaussian, and frequency
				
			pix_ind += 1
		
		connection.send(g_array)
	
	###
	def build_log_gaussian_array(self, num_proc=1, threshold=0.):
		"""
		Builds data array for each pixel, Gaussian term, and frequency.
		"""
		
		#Pixelate the sky
		pixarray = np.zeros((self.npix,3))  #initializes a 2-D array (pixels x (ipix, theta, phi)'s)
		for ipix in xrange(self.npix):
			theta, phi = hp.pix2ang(self.nside, ipix)  #maps theta and phi to ipix
			pixarray[ipix,:] = np.array([ipix, theta, phi])
		
		#Launch processes that calculate posterior
		procs = []  #holders for process identification
		
		proc_num_pix = np.ceil(self.npix/num_proc)
		proc_pix_start = 0
		proc_pix_end = proc_num_pix
		
		for iproc in xrange(num_proc):
			proc_thetas = pixarray[proc_pix_start:proc_pix_end,1]
			proc_phis = pixarray[proc_pix_start:proc_pix_end,2]
			
			con1, con2 = mp.Pipe()
			args = (proc_thetas, proc_phis, 0., con2, threshold)
			
			p = mp.Process(target=self.log_gaussian_data, args=args)
			p.start()
			con2.close()  # this way the process is the only thing that can write to con2
			procs.append((p, proc_pix_start, proc_pix_end, con1))
			
			proc_pix_start += proc_num_pix
			proc_pix_end += proc_num_pix
					
		#Build unnormalized log posterior over whole sky
		g_array = np.zeros((self.npix, self.hprior.num_gaus, self.len_freqs)) #3-D array (pixels * Gaussian terms * frequencies)
		print np.shape(g_array)
		while len(procs):
			for ind, (p, _, _, _) in enumerate(procs):
				if not p.is_alive():
					p, proc_pix_start, proc_pix_end, con1 = procs.pop(ind)
					g_array[proc_pix_start:proc_pix_end,:,:] = con1.recv()
		
		return g_array
	
	###
	def calculate_posterior_weight(self, g_array, f_low, f_up):
		"""
		Calculates posterior weight
		"""
		
		#Pixelate the sky
		pixarray = np.zeros((self.npix,3))  #initializes a 2-D array (pixels x (ipix, theta, phi)'s)
		for ipix in xrange(self.npix):
			theta, phi = hp.pix2ang(self.nside, ipix)  #maps theta and phi to ipix
			pixarray[ipix,:] = np.array([ipix, theta, phi])
		thetas = pixarray[:,1]
		phis = pixarray[:,2]
		
		#Find indices of flow and fhigh
		iflow = np.where(self.freqs==f_low)[0][0]
		ifup = np.where(self.freqs==f_up)[0][0]
		num_f = iflow - ifup
		num_g = np.shape(g_array)[1]
		
		#sum Gaussians over frequency range
		g_array_summed = np.sum(g_array[:,:,iflow:(ifup+1)], axis=2)
		
		#Add Gaussian terms together
		max_log = np.amax(g_array_summed, axis=1) #find maximum Gaussian-term value for each pixel
		max_log_array = np.ones((self.npix,num_g))*np.array([max_log]*num_g).transpose()
		g_array_summed -= max_log_array  #factor out maximum value for each pixel
		amplitudes = np.array([self.hprior.amplitudes]*self.npix)
		g_array_summed = amplitudes*np.power(10., g_array_summed)  #convert from log value to actual value for each gaussian term and add amplitudes
		log_posterior_weight = max_log + np.log10( np.sum(g_array_summed, axis=1) )
		log_posterior_weight += np.log10( self.angprior.prior_weight(theta=thetas, phi=phis) )
		
		#Find max log posterior value and subtract it from all log posterior values (partial normalization)
		max_log_pos = max(log_posterior_weight)		
		log_posterior_weight -= max_log_pos

		#Convert log posterior to normal posterior
		posterior_weight = np.power(10., log_posterior_weight)

		return posterior_weight, max_log_pos
		
	
	###
	def calculate_log_bfactor(self, posterior_weight, max_log_pos, fmin, fmax):
		"""
		Calculates bayes factor for a given frequency range
		"""
		
		#Calculate and print log Bayes factor
		log_sum_term = np.log10(sum(posterior[:,1]))
		log_B = max_log_pos + log_sum_term
		return log_B
			
	###
	def model_select(self, g_array, fmin, fmax):
		"""
		Model select to select optimal frequency window to build a posterior with
		"""
		flow = fmin
		f_up = fmax
		return f_low, f_up
		
	###
	def build_posterior(self, g_array, f_low, f_up):
		"""
		Builds the posterior over the entire sky.
		"""
		
		#Get posterior weights		
		posterior = np.zeros((self.npix,2)) # initalizes 2-D array (pixels x (ipix, log posterior weight)'s)
		posterior[:,0] = np.array(range(self.npix))  #assigns pixel indices
		posterior[:,1] = self.calculate_posterior_weight(g_array=g_array, f_low=f_low, f_up=f_up)[0]
			
		#Normalize posterior
		posterior[:,1] /= sum(posterior[:,1])
		
		return posterior
	
	###	
	def plot_posterior(self, posterior, figname, title=None, unit='posterior density', est_theta=None, est_phi=None, inj_theta=None, inj_phi=None, min_val=None, max_val=None):
		"""
		*Plots a skymap of a given posterior density
		"""
		
		#Checks on posterior, title, unit
		#~~~~~~~~~~~
		
		#Calculate area of each pixel
		pixarea = hp.nside2pixarea(self.nside)  #area of each pixel
		
		#Plot posterior probability density function as skymap
		fig = plt.figure(0)
		hp.mollview(posterior[:,1], fig=0, title=title, unit=unit, flip="geo", min=min_val, max=max_val)
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
		
