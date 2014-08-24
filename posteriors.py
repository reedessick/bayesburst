usage = """module that computes bayesian posteriors via analytic marginalization """

import utils
np = utils.np
linalg = utils.linalg
hp = utils.hp

import priors
plt = priors.plt

import multiprocessing as mp
import time



print """WARNING: 
	normalizations are questionable for these posteriors.
	need to check that exponents are REAL
		-> put a tolerance on the REAL vs IMAG components and raise error if violated
		-> cast to reals when filling in array
	

	We also need to implement effective numbers of polarizations. This can be done by zeroing certain elements of the arrays (like invP, A, etc).
		if we zero those elements, we need to make copies of the arrays so we don't over-write them...
		we should delegate this to a separate method that knows how to handle A, invA, B, P, invP, etc.
			write n_pol_eff so it figures out the effective number of polarizations at each point in the sky (using network.rank(A)?)
			pass this into the other function that re-formats A, invA, B, P, invP as needed
			compute a projection matrix (in n_pol-space) for each ipix, then use it on A, invA, B, P, invP, etc as needed
"""



#=================================================
#
#                Posterior class
#
#=================================================
class Posterior(object):
	"""
	an object capable of calculating the marginalized posteriors
	"""
	
	#write function to estimate location of the event
	
	###
	def __init__(self, network=None, hPrior=None, angPrior=None, seglen=None, data=None):
		"""
		Posterior computed using prior information, network, and data

		We require:
			*network is an instance of utils.Network
			*hPrior is an instance of priors.hPrior
			*angPrior is an instance of priors.angPrior
			*data is a 2-D array
				np.shape(data) = (n_freqs, n_ifo)
			*seg_len is a float
		"""
		### set up placeholders that will be filled
		self.n_pol = None
		self.n_freqs = None
		self.n_ifo = None
		self.nside = None

		self.network = None
		self.hPrior = None
		self.angPrior = None
		self.seglen = None
		self.data = None

		### set data within this object
		if network != None:
			self.set_network(network)
		if hPrior != None:
			self.set_hPrior(hPrior)
		if angPrior != None:
			self.set_angPrior(angPrior)
		if seglen != None:
			self.set_seglen(seglen)
		if data != None:
			self.set_data(data)

		### storage of pre-computed values
		self.theta = None
		self.phi = None
		self.A = None
		self.invA = None
		self.B = None
		self.P = None
		self.invP = None

	###
	def set_network(self, network):
		""" check and set network """
                # check network
                if not isinstance(network, utils.Network):
                        raise ValueError, "network must be a utils.Network object"
                n_pol = network.Np
		if self.n_pol and (n_pol != self.n_pol):
			raise ValueError, "inconsistent n_pol"

		freqs = network.freqs
		n_freqs = len(freqs)
		if self.n_freqs and (n_freqs != self.n_freqs):
			raise ValueError, "inconsistent n_freqs"
		if self.hPrior and np.any(freqs != self.hPrior.freqs):
			raise ValueError, "inconsistent freqs with self.hPrior"

                ifos = network.detectors_list()
                n_ifo = len(ifos)
		if self.data and (n_ifo != self.n_ifo):
			raise ValueError, "please set data=None before changing network"
		
		self.network = network
		self.n_pol = n_pol
		self.n_freqs = n_freqs

		self.ifos = ifos
		self.n_ifo = n_ifo

	###
	def set_hPrior(self, hPrior):
		""" check and set hPrior """
                # check hPrior
                if not isinstance(hPrior, priors.hPrior):
                        raise ValueError, "hPrior must be a priors.hPrior object"
		n_pol = hPrior.n_pol
		if self.n_pol and (n_pol != self.n_pol):
                        raise ValueError, "inconsistent n_pol"

		freqs = hPrior.freqs
		n_freqs = len(freqs)
		if self.n_freqs and (n_freqs != self.n_freqs):
			raise ValueError, "inconsistent n_freqs"
		if self.network and np.any(freqs != self.network.freqs):
			raise ValueError, "inconsistent freqs with self.network"

		self.hPrior = hPrior
		self.n_freqs = n_freqs 
		self.n_pol = n_pol

	### 
	def set_angPrior(self, angPrior):
		""" check and set angPrior """
                # check angPrior
                if not isinstance(angPrior, priors.angPrior):
                        raise ValueError, "angPrior must be a priors.angPrior object"

		nside = angPrior.nside
		if self.nside and (nside != self.nside):
			raise ValueError, "inconsistent nside"

		self.angPrior = angPrior
		self.nside = nside
		self.npix = hp.nside2npix(nside)

	###
	def set_seglen(self, seglen):
		""" set seglen """
		self.seglen = seglen

	###
	def set_data(self, data):
		""" check and set the data """
		### make sure we have a network already
		if not self.network:
			raise StandardError, "please set_network() before setting data"

                if not isinstance(data, np.ndarray):
                        data = np.array(data)
                if len(np.shape(data)) != 2:
                        raise ValueError, "bad shape for data"

                n_freqs, n_ifo = np.shape(data)
                if n_freqs != self.n_freqs:
                        raise ValueError, "inconsistent n_freqs"
                if n_ifo != self.n_ifo:
                        raise ValueError, "inconsistent n_ifo"

		self.data = data

	###
	def set_theta_phi(self, coord_sys="E", **kwargs):
		"""
		store theta, phi in coord sys.
		delegates to self.angPrior.set_theta_phi, which delegates to utils.set_theta_phi
		"""
		if not self.nside:
			raise ValueError, "set_angPrior() first"
		self.angPrior.set_theta_phi(coord_sys=coord_sys, **kwargs)

		self.theta = self.angPrior.theta
		self.phi = self.angPrior.phi
		self.coord_sys = self.angPrior.coord_sys

	###
	def set_A(self, psi=0.0):
		"""
		compute and store A, invA
		A is computed through delegation to self.network
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
		if not self.network:
			raise ValueError, "set_network() first"

		if (self.theta != None) and (self.phi != None):
			thetas = self.theta
			phis = self.phi
		else:
			if not self.nside:
				raise ValueError, "set_angPrior() first"
			thetas, phis = hp.pix2ang(self.nside, np.arange(self.npix))

                self.A = self.network.A(thetas, phis, psi, no_psd=False)
		self.invA = linalg.inv(self.A)

	###
	def set_B(self, psi=0.0):
		"""
		compute and store B
		B is computed through delegation to self.network
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
		if not self.network:
			raise ValueError, "set_network() first"

                if (self.theta!=None) and (self.phi!=None):
                        thetas = self.theta
                        phis = self.phi
                else:
                        if not self.nside:
                                raise ValueError, "set_angPrior() first"
                        thetas, phis = hp.pix2ang(self.nside, np.arange(self.npix))

		self.B = self.network.B(thetas, phis, psi, no_psd=False)

	###
	def set_P(self, psi=0.0):
		"""
		comptue and store P, invP
		P is computed with knowledge of self.A and self.hPrior
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
		if self.A == None:
			raise ValueError, "set_A() first"
		A = self.A

		n_pix = self.npix
		n_freqs = self.n_freqs
		n_pol = self.n_pol
		n_gaus = self.hPrior.n_gaus
		P = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), complex) ### (A+Z)
		invP = np.empty_like(P)
                for g in xrange(n_gaus):
			Z = np.empty_like(A, complex)
			Z[:] = self.hPrior.incovariance[:,:,:,g]

                        P[:,:,:,:,g] = (A + Z)
			invP[:,:,:,:,g] = linalg.inv( P[:,:,:,:,g] ) ### this appears to be VERY memory intensive...
			                                             ### perhaps prohibitively so

		self.P = P
		self.invP = invP

	###
	def __check_theta_phi_psi(self, theta, phi, psi):
		""" delegates to utils.check_theta_phi_psi() """
		return utils.check_theta_phi_psi(theta, phi, psi)

	###
	def __check_A(self, A, n_pix, n_pol_eff):
		""" checks A's shape """
		if len(np.shape(A)) != 4:
			raise ValueError, "bad shape for A"
                n, n_freqs, n_pol, n_p = np.shape(A)
                if n != n_pix:
                	raise ValueError, "inconsistent shape between n_pix and A"
                if n_pol != n_p:
                	raise ValueError, "inconsistent shape within A"
                if np.any(n_pol != n_pol_eff):
                	raise ValueError, "n_pol != n_pol_eff"
                if n_freqs != self.n_freqs:
                	raise ValueError, "inconsistent n_freqs"

		return n_pix, n_freqs, n_pol, A

	###
	def __check_B(self, B, n_pix, n_pol_eff):
		""" check's B's shape """
		if len(np.shape(B)) != 4:
			raise ValueError, "bad shape for B"
		n, n_freqs, n_pol, n_ifo = np.shape(B)
		if n != n_pix:
			raise ValueError, "inconsistent shape between theta, phi, psi and B"
		if np.any(n_pol != n_pol_eff):
			raise ValueError, "n_pol != n_pol_eff"
		if n_freqs != self.n_freqs:
			raise ValueError, "inconsistent n_freqs"
		if n_ifo != self.n_ifo:
			raise ValueError, "inconsistent n_ifo"

		return n_pix, n_freqs, n_pol, n_ifo, B

	###
	def __check_P(self, P, n_pix, n_pol_eff):
		""" checks P's shape """
		if len(np.shape(P)) != 5:
			raise ValueError, "bad shape for P"
		n, n_freqs, n_pol, n_p, n_gaus = np.shape(P)
		if n != n_pix:
			raise ValueError, "inconsistent shape between n_pix and P"
		if n_pol != n_p:
			raise ValueError, "inconsistent shape within P"
		if np.any(n_pol != n_pol_eff):
			raise ValueError, "n_pol != n_pol_eff"
		if n_freqs != self.n_freqs:
			raise ValueError, "inconsistent n_freqs"
		if n_gaus != self.hPrior.n_gaus:
			raise ValueError, "inconsistent n_gaus"

		return n_pix, n_freqs, n_pol, n_gaus, P

	###
	def n_pol_eff(self, theta, phi):
		"""
		computes the effective number of polarizations given the network and theta, phis
		
		right now, this is a trivial delegation
		"""
		n_pix, theta, phi, psi = self.__check_theta_phi_psi(theta, phi, 0.0)
		return self.n_pol*np.ones((n_pix,),int)

	###		
	def mle_strain(self, theta, phi, psi=0., n_pol_eff=None, invA_B=None):
		"""
		calculates maximum likelihood estimate of strain at a given position.
		
		We require:
			theta: float or 1-D array (radians)
				np.shape(theta) = (N)
			phi: float or 1-D array (radians)
				np.shape(theta) = (N)
			psi: float or 1-D array (radians)
				np.shape(theta) = (N)
				ignored if invA_B!=None and 

			invA_B = (invA, B)
				if theta, phi, psi are scalars:
					invA : 3-D array
						np.shape(A) = (n_freqs, n_pol, n_pol)
					B : 3-D array
						np.shape(B) = (n_freqs, n_ifo, n_pol)
				if theta, phi, psi are arrays
					invA : 4-D array:
						np.shape(A) = (N, n_freqs, n_pol, n_pol)
					B : 4-D array:
						np.shape(B) = (N, n_freqs, n_ifo, n_pol)

			n_pol_eff: int
				the effective number of polarizations

		computes mle_strain using self.data
			mle_strain[j] = \sum_k \sum_beta invA[j][k] B[beta][k] self.data[beta]
		"""
		### check that we have data
		if self.data==None:
			raise ValueError, "set_data() first"
	
		### check angular coords
		n_pix, theta, phi, psi = self.__check_theta_phi_psi(theta, phi, psi)

		# effective number of polarizations
		if (n_pol_eff == None):
			n_pol_eff = self.n_pol
		else:
			n_pol_eff = np.max(n_pol_eff)

		### check invA_B
		if invA_B:
			invA, B = invA_B
			if n_pix == 1:
				if len(np.shape(invA)) == 3:
					invA = np.array([invA])
				else:
					raise ValueError, "bad shape for invA"
				if len(np.shape(B)) == 3:
					B = np.array([B])
				else:
					raise ValueError, "bad shape for B"

			n_pix, n_freqs, n_pol, invA = self.__check_A(invA, n_pix, n_pol_eff)
			n_pix, n_freqs, n_pol, n_ifo, B = self.__check_B(B, n_pix, n_pol_eff)
				
		else:
			invA = linalg.inv(self.network.A(thetas, phis, psi, no_psd=False)) ### may throw errors due to singular matrix
			B = self.network.B(thetas, phis, psi, no_psd=False)
			n_freqs = self.n_freqs
			n_ifo = self.n_ifo
		
					
		### Calculate h_ML
		mle_h = np.zeros((n_pix, n_freqs, n_pol_eff), 'complex')  #initialize h_ML as 3-D array (N, n_freqs, n_pol_eff)
		for ipix in xrange(n_pix):
			for j in xrange(n_pol_eff):
				for k in xrange(n_pol_eff):
					for beta in xrange(n_ifo):
						mle_h[ipix,:,j] += invA[ipix,:,j,k] * B[ipix,:,k,beta] * self.data[:,beta]
		
		if n_pix == 1:
			return mle_h[0]
		else:
			return mle_h

	### 	
	def log_posterior_elements(self, theta, phi, psi=0.0, invP_B=None, A_invA=None, connection=None, max_array_size=100, diagnostic=False):
		"""
                calculates natural logarithm of posterior for each pixel, gaussian term, and frequency
                
                We require:
                        theta: float or 1-D array (radians)
                                np.shape(theta) = (n_pix)
                        phi: float or 1-D array (radians)
                                np.shape(theta) = (n_pix)
                        psi: float or 1-D array (radians)
                                np.shape(theta) = (n_pix)
                                ignored if invA_B!=None and 

			COMPUTE THEM IN THE DOMINANT POLARIZATION FRAME?

                        invP_B = (invP, B)
                                if theta, phi, psi are scalars:
                                        invP : 3-D array
                                                np.shape(invP) = (n_freqs, n_pol, n_pol)
                                        B : 3-D array
                                                np.shape(B) = (n_freqs, n_ifo, n_pol)
                                if theta, phi, psi are arrays
                                        invP : 4-D array:
                                                np.shape(invP) (N, n_freqs, n_pol, n_pol)
                                        B : 4-D array:
                                                np.shape(B) = (N, n_freqs, n_ifo, n_pol)

		computes log_posterior_elements, n_pol_eff using self.data

		if connection:
			we return the output through connection rather than with a return statement
			used for parallelization
			if array is larger than max_array_size, we divide it into chunks of size <= max_array size and send those through the pipe sequentially.

		returns ans, n_pol_eff
			ans : 3-D array
				contains the log of the unnormalized posterior split into pixels, gaussian terms, and frequencies
				np.shape(ans) = (n_pix, n_gaus, n_freqs)
			n_pol_eff: 1-D array
				contains the number of effective polarizations used to compute the terms in ans

			ans is computed without directy reference to mle_strain, which means we do not have to invert A
			x[j,:] = \sum_beta B[ipix,:,beta,j] * self.data[beta,:] ### hidden index is frequency
			ans[ipix,g,:] = \sum_j \sum_k np.conjugate(x[j,:]) * invP[:,j,k] * x[k,:]

		if diagnostic:
			we also compute ans by division into separate terms
				mle =   np.conjugate(mle_strain) * A * mle_strain
				cts = - np.conjuaget(mle_strain-means) * (Z - Z*invP*Z) * (mle_strain-means)
				det =   np.log( linalg.det(invP) ) + np.log( linalg.det(Z) )
		
		we expect
			ans = mle + cts + det
		"""
		if np.any(self.hPrior.means != 0):
			raise ValueError, "we only support zero-mean gaussians for now"

                ### check that we have data
                if self.data==None:
                        raise ValueError, "set_data() first"

		### check angular coordinates
		n_pix, theta, phi, psi = self.__check_theta_phi_psi(theta, phi, psi)

		### pull out number of gaussian terms
                n_gaus = self.hPrior.n_gaus

		### pull out number of frequencies
		n_freqs = self.n_freqs

		### check invP_B
		if invP_B:
			invP, B = invP_B
			if n_pix == 1:
				if len(np.shape(invP)) == 4:
                                        invP = np.array([invP])
                                else:
                                        raise ValueError, "bad shape for invP"
				if len(np.shape(B)) == 3:
                                        invA = np.array([B])
                                else:
                                        raise ValueError, "bad shape for B"

			n_pix, n_freqs, n_pol, n_gauss, invP = self.__check_P(invP, n_pix, self.n_pol)
                	n_pix, n_freqs, n_pol, n_ifo, B = self.__check_B(B, n_pix, self.n_pol)
		else:
			A = self.network.A(theta, phi, psi, no_psd=False)
			invP = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), float) ### (A+Z)^{-1}
			for g in xrange(n_gaus):
	                        Z = np.empty_like(A, complex)
        	                Z[:] = self.hPrior.incovariance[:,:,:,g]
        	                invP[:,:,:,:,g] = linalg.inv(A + Z)
                        B = self.network.B(theta, phi, psi, no_psd=False)
                        n_freqs = self.n_freqs
			n_pol = self.n_pol
                        n_ifo = self.n_ifo

		###################################################################################
		#
		# if we want to work in DPF, we should make all transformations here
		# also, store the effective number of polarizations, etc so we can downsample as needed
		#
		###################################################################################
		n_pol_eff = self.n_pol_eff(theta, phi)
		
		### instantiate arrays
		ans = np.zeros((n_pix, n_gaus, n_freqs), float) ### array storing all data
		                                                ### (n_pix, n_gaus, n_freqs)

		### set up required structures to compute diagnostic values
		if diagnostic:
			if A_invA:
				A, invA = A_invA
				if n_pix == 1:
					if len(np.shape(A)) == 3:
						A = np.array([A])
					else:
						raise ValueError, "bad shape for A"
					if len(np.shape(invA)) == 3:
						invA = np.array([invA])
					else:
						raise ValueError, "bad shape for invA"

				n_pix, n_freqs, n_pol, A = self.__check_A(A, n_pix, self.n_pol)
				n_pix, n_freqs, n_pol, invA = self.__check_A(invA, n_pix, self.n_pol)
			else:
				A = self.network.A(theta, phi, psi, no_psd=False)
				invA = linalg.inv(A)

			h_mle = self.mle_strain(theta, phi, psi, n_pol_eff=n_pol_eff, invA_B=(invA,B))
			means = self.hPrior.means  ### required to be zeros to compute "ans", but may be non-zero here...
                        means_conj = np.conj(means)

			mle = np.zeros_like(ans, float) ### maximum likelihood estimate's contribution
			cts = np.zeros_like(ans, float) ### term from completing the square in the marginalization
			det = np.zeros_like(ans, float) ### term from determinant from marginalization
	
		### define frequency spacing (important for determinant terms)
		df = self.seglen**-1
	
		### iterate over all sky positions
		for ipix, (t,p) in enumerate(zip(theta, phi)):
			### compute ans. this formulation requires zero-mean gaussian prior decomposition
			for g in xrange(n_gaus):
				x = np.zeros((n_freqs, n_pol), complex) ### transformed data
				for alpha in xrange(n_ifo):
					for j in xrange(n_pol):
						x[:,j] += self.data[:,alpha] * B[ipix,:,j,alpha]
				x_conj = np.conjugate(x)

				### sum over polarizations and compute ans
				for j in xrange(n_pol):
					for k in xrange(n_pol):
						ans[ipix,g,:] += x_conj[:,j] * invP[ipix,:,j,k,g] * x[:,k]

				### include determinant
				Z = self.hPrior.incovariance[:,:,:,g]
#				ans[ipix,g,:] += 0.5*( np.log( linalg.det(invP[ipix,:,:,:,g]/df) ) + np.log( linalg.det(Z*df) ) ) / df
				ans[ipix,g,:] += ( np.log( linalg.det(invP[ipix,:,:,:,g]/df) ) + np.log( linalg.det(Z*df) ) ) / df

			if diagnostic: ### break things up term-by-term for diagnostic purposes
				### compute mle estimate for strain (this may not be optimal for the likelihood calculation...)
				h = h_mle[ipix]	
				h_conj = np.conjugate(h)
					
				n_eff = n_pol_eff[ipix]
				for g in xrange(n_gaus): ### iterate over gaussian terms
	
					Z = self.hPrior.incovariance[:,:,:,g]

					### iterate over polarizations
					for j in xrange(n_eff): 
						diff_conj = h_conj[:,j] - means_conj[:,j,g] ### calculate displacement from mean
	
						for k in xrange(n_eff):
							diff = h[:,k] - means[:,k,g] ### calculate displacement from mean
							
							mle[ipix,g,:] += h_conj[:,j] * A[ipix,:,j,k] * h[:,k] ### maximum likelihood estimate
							cts[ipix,g,:] -= diff_conj * Z[:,j,k] * diff ### complete the square
						
							for m in xrange(n_eff):
								for n in xrange(n_eff):
									cts[ipix,g,:] += diff_conj * Z[:,j,m] * invP[ipix,:,m,n,g] * Z[:,n,k] * diff ### complete the square

#					det[ipix,g,:] = 0.5*( np.log( linalg.det(invP[ipix,:,:,:,g]/df) ) + np.log( linalg.det(Z*df) ) ) / df ### determinant
					det[ipix,g,:] = ( np.log( linalg.det(invP[ipix,:,:,:,g]/df) ) + np.log( linalg.det(Z*df) ) ) / df ### determinant

		if connection:
			### send ans
			utils.flatten_and_send(connection, ans, max_array_size=max_array_size)
			### send n_pol_eff
			utils.flatten_and_send(connection, n_pol_eff, max_array_size=max_array_size)

			if diagnostic:
				### send mle
				utils.flatten_and_send(connection, mle, max_array_size=max_array_size)
				### send cts
				utils.flatten_and_send(connection, cts, max_array_size=max_array_size)
				### send det
				utils.flatten_and_send(connection, det, max_array_size=max_array_size)
#			if diagnostic:
#				connection.send((ans, n_pol_eff, (mle, cts, det)))
#			else:
#				connection.send((ans, n_pol_eff))
		else:
			if diagnostic:
				return ans, n_pol_eff, (mle, cts, det)
			else:
				return ans, n_pol_eff

	###
	def log_posterior_elements_mp(self, theta=None, phi=None, psi=0.0,  invP_B=None, A_invA=None, n_proc=1, max_proc=1, max_array_size=100, diagnostic=False):
		"""
		divides the sky into jobs and submits them through multiprocessing

		if theta==None or phi==None:
			pixelization is delegated to healpix and we compute the posterior for all points in the sky
			we ignore invP_B, A_invA if they are supplied because we have no reference for angPrior

		actual computation is delegated to __log_posterior()
		"""
		### get positions in the sky	
		if theta==None or phi==None:
			theta, phi = hp.pix2ang(self.nside, np.arange(self.npix))
			invP_B = A_invA = None ### we ignore these because we have no reference for positions?

		n_pix, theta, phi, psi = self.__check_theta_phi_psi(theta, phi, psi)

                if invP_B==None:
                        A = self.network.A(theta, phi, psi, no_psd=False)
                        invP = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), float) ### (A+Z)^{-1}
                        for g in xrange(n_gaus):
	                        Z = np.empty_like(A, complex)
        	                Z[:] = self.hPrior.incovariance[:,:,:,g]
                                invP[:,:,:,:,g] = linalg.inv(A + Z)
                        B = self.network.B(theta, phi, psi, no_psd=False)
#			invP_B = (invP, B)
		else:
			invP, B = invP_B

                ### instantiate arrays
		n_gaus = self.hPrior.n_gaus
		n_freqs = self.n_freqs
                ans = np.empty((n_pix, n_gaus, n_freqs), float) ### array storing all data
                                                                ### (n_pix, n_gaus, n_freqs)
		n_pol_eff = np.empty((n_pix,), int)

                if diagnostic:
                        mle = np.empty_like(ans, float) ### maximum likelihood estimate's contribution
                        cts = np.empty_like(ans, float) ### term from completing the square in the marginalization
                        det = np.empty_like(ans, float) ### term from determinant from marginalization

			if A_invA==None:
				A = self.network.A(theta, phi, psi, no_psd=False)
                                invA = linalg.inv(A)
#				A_invA = (A, invA)
			else:
				A, invA = A_invA

		### define size of jobs
		npix_per_proc = np.ceil(1.0*n_pix/n_proc)
		procs = []

		### iterate through jobs
		for i in xrange(n_proc):

			if len(procs): ### if we already have some processes launched
				### reap old processes
				if len(procs) >= max_proc:
					p, start, end, con1 = procs.pop()

				### fill in data
				shape = np.shape(ans[start:end])
				ans[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)
				n_pol_eff[start:end] = utils.recv_and_reshape(con1, np.shape(n_pol_eff[start:end]), max_array_size=max_array_size)
				if diagnostic:
					mle[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)
					cts[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)
					det[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)

#				ind = 0
#				while len(procs) >= max_proc:
#					for ind, (p, _,_,_) in enumerate(procs):
#						if not p.is_alive():
#							print "bing"
#							break
#					else:
#						continue
#					p, start, end, con1 = procs.pop(ind)

#				if diagnostic:
#					_ans, _n_pol_eff, (_mle,_cts,_det) = con1.recv()
#					ans[start:end,:,:] = _ans
#					n_pol_eff[start:end] = _n_pol_eff
#					mle[start:end,:,:] = _mle
#					cts[start:end,:,:] = _cts
#					det[start:end,:,:] = _det
#				else:
#					_ans, _n_pol_eff = con1.recv()
#					ans[start:end,:,:] = _ans
#					n_pol_eff[start:end] = _n_pol_eff

			### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
			_psi = psi[start:end]
			_invP_B = (invP[start:end], B[start:end])
			if diagnostic:
				_A_invA = (A[start:end], invA[start:end])
			else:
				_A_invA = None

			### launch process
			con1, con2 = mp.Pipe()
			p = mp.Process(target=self.log_posterior_elements, args=(_theta, _phi, _psi, _invP_B, _A_invA, con2, max_array_size, diagnostic))
			p.start()
#			con2.close()
			procs.append( (p, start, end, con1) )

		### reap remaining processes
		ind = 0
		while len(procs):
			p, start, end, con1 = procs.pop()
			### fill in data
                        shape = np.shape(ans[start:end])
                        ans[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)
                        n_pol_eff[start:end] = utils.recv_and_reshape(con1, np.shape(n_pol_eff[start:end]), max_array_size=max_array_size)
                        if diagnostic:
                        	mle[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)
                                cts[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)
                                det[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size)

			for ind, (p, _,_,_) in enumerate(procs):
				if not p.is_alive():
					print "bing"
					break
			else:
				continue

			p, start, end, con1 = procs.pop(ind)
#			if diagnostic:
#				_ans, _n_pol_eff, (_mle,_cts,_det) = con1.recv()
 #                               ans[start:end,:,:] = _ans
#				n_pol_eff[start:end] = _n_pol_eff
 #                               mle[start:end,:,:] = _mle
  #                              cts[start:end,:,:] = _cts
   #                             det[start:end,:,:] = _det
    #                    else:
#				_ans, _n_pol_eff = con1.recv()
 #                               ans[start:end,:,:] = _ans
#				n_pol_eff[start:end] = _n_pol_eff

		if diagnostic:
			return ans, n_pol_eff, (mle, cts, det)
		else:
			return ans, n_pol_eff

	###
	def log_posterior(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False):
		"""
		computes log(posterior) using
			log_posterior_elements : a large structure containing posterior values for each pixel, gaussian term, and frequency
			n_pol_eff : an array storing the effective number of polarizations for each element of log_posterior_elements
			freq_truth : a truth array labeling which frequency bins we should include

		calculation sums over all frequencies selected by freq_truth 
		sums over all gaussian terms (requires manipulation of logarithms)

		thetas, phis are used with self.angPrior to compute that contribution

		returns log(posterior) : 1-D array 
			np.shape(log_posterior) = (n_pix)
		"""
		
		### sum over frequencies
		df = self.seglen**-1 ### frequency spacing
		summed_log_posterior_elements = np.sum(log_posterior_elements[:,:,freq_truth], axis=2) * df

		### sum over gaussians
		log_posterior_weight = np.empty((len(summed_log_posterior_elements),),float)
		for ipix in xrange(len(summed_log_posterior_elements)):
			log_posterior_weight[ipix] = utils.sum_logs(summed_log_posterior_elements[ipix]+np.log(self.hPrior.amplitudes)) ### add the gaussian amplitudes only after summing over frequencies

		### add contribution of angular Prior
		log_posterior_weight += np.log( self.angPrior(theta=thetas, phi=phis) )
	
		'''	
		#Find normalization factor for prior on h
		eff_num_pol = g_array[:,:,0,1]  #2-D array (pixels * Gaussians)
		log_var_terms = (eff_num_pol*num_f/2.)*np.log10([self.hPrior.covariance[0,0,0,:]]*self.npix)  #2-D array (pixels * Gaussians)
		log_var_max = np.amax(log_var_terms, axis=1)
		log_var_max_array = np.ones((self.npix,num_g))*np.array([log_var_max]*num_g).transpose()
		log_var_terms -= log_var_max_array
		hpri_sum_terms = amplitudes_unnormed * np.power(10., log_var_terms)
		log_hpri_norm = (eff_num_pol[:,0]*num_f/2.)*np.log10(np.pi*self.seg_len/2.) + log_var_max + np.log10( np.sum(hpri_sum_terms, axis=1))
		log_posterior_weight -= log_hpri_norm
		'''

		if normalize:
			log_posterior_weight -= utils.sum_logs(log_posterior_weight)

		return log_posterior_weight
		

	###
	def posterior(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False):
		""" np.exp(self.log_posterior_weight) """
		return np.exp( self.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize) )

	###
	def posterior_mp(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False):
		raise StandardError, "WRITE ME"

	###
	def log_bayes(self, log_posterior):
		""" 
		computes the log(bayes factor) using log_posterior 
		this is done through direct marginalization over all n_pix
		"""
		return utils.sum_logs(log_posterior)

	###
	def log_bayes_mp(self, log_posterior):
		raise StandardError, "WRITE ME"

	###
	def bayes(self, log_posterior):
		""" np.exp(self.log_bayes(log_posterior)) """
		return np.exp( self.log_bayes(log_posterior) )

	###
	def bayes_mp(self, log_posterior):
		raise StandardError, "WRITE ME"

	###
	def plot(self, figname, posterior=None, title=None, unit=None, inj=None, est=None):
                """
                generate a plot of the posterior and save it to figname
                if inj != None:
                        (theta,phi) = inj
                        plot marker at theta,phi

		if posterior==None:
			we calculate the posterior at all points in the sky defined by self.nside
			we use all frequency bins
                """
		if posterior==None:
			posterior = self()/hp.nside2pixarea(self.nside) ### delegate to self.__call__()

                ### generate plot
                fig_ind = 0
                fig = plt.figure(fig_ind)
                hp.mollview(posterior, title=title, unit=unit, flip="geo", fig=fig_ind)
                hp.graticule()

                ### plot point if supplied
                if inj:
                        ax = fig.gca()
                        marker = ax.projplot(inj, "wx", alpha=0.5)[0]
                        marker.set_markersize(10)
                        marker.set_markeredgewidth(2)

		if est:
			ax = fig.gca()
			marker = ax.projplot(est, "wo", alpha=0.5)[0]
			marker.set_markersize(10)
			marker.set_markeredgewidth(2)

                ### save
                fig.savefig(figname)
                plt.close()
	
	###
	def __call__(self):
		"""
		calculate the posterior at all points in the sky defined by self.nside and healpy
		we use all frequency bins
		"""
		### check for pre-computed arrays
		if self.theta==None or self.phi==None:
			self.set_theta_phi()
		theta = self.theta
		phi = self.phi

		if self.invP==None:
			self.set_P(self.npix, self.n_pol)
		if self.B==None:
			self.set_B(self.npix, self.n_pol)
		invP_B = (self.invP, self.B)

		if self.A==None:
			self.set_A(self.n_pix, self.n_pol)
		A_invA = (self.A, self.invA)

		log_posterior_elements, n_pol_eff = self.log_posterior_elements(theta, phi, psi=0.0, invP_B=invP_B, A_invA=A_invA, connection=None, diagnostic=False)

		freq_truth = np.ones((self.n_freqs,), "bool")

		return self.posterior(theta, phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=True)

	###
	def __repr__(self):
		return self.__str__()

	###
	def __str__(self):
		return """posteriors.Posterior object
	seglen = %s

	network = %s

	hPrior = %s

	angPrior = %s"""%(str(self.seglen), self.network, self.hPrior, self.angPrior)
		
