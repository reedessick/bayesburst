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
	
	need to implement timers to track algorithmic optimization

	WE CAN COMPUTE DETERMINANT TERMS AHEAD OF TIME-> DON'T NEED TO COMPUTE THIS FOR EVERY POINT IN THE SKY
		write this into set_hPrior (detZ) and set_P (detinvP)
		just reference in log_posterior_elements(_mp)	

	WRITE STATIC METHODS that are equivalent to the manipulation methods? these can then be called outside of the object...

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

	#=========================================
	# set_* functions
	#=========================================

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
		self.n_pix = hp.nside2npix(nside)

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
		self.dataB = None
		self.dataB_conj = None

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

#	###
#	def set_theta_phi_mp(self, coord_sys="E", num_proc=1, max_proc=1, max_array_size=100, **kwargs):
#		raise StandardError, "WRITE ME"

	###
	def set_A(self, psi=0.0):
		"""
		compute and store A, invA
		A is computed through delegation to self.network
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
		if not self.network:
			raise ValueError, "set_network() first"

		if (self.theta == None) or (self.phi == None):
			raise ValueError, "set_theta_phi() first"

                self.A = self.network.A(self.theta, self.phi, psi, no_psd=False)
		self.invA = linalg.inv(self.A)

	###
	def __set_A_mp(self, theta, phi, psi, connection, max_array_size=100):
		"""
		a helper method for set_A_mp
		"""
		A = self.network.A(theta, phi, psi, no_psd=False)
		invA = linalg.inv(A)
		utils.flatten_and_send(connection, A, max_array_size=max_array_size)
		utils.flatten_and_send(connection, invA, max_array_size=max_array_size)

	###
	def set_A_mp(self, psi=0.0, num_proc=1, max_proc=1, max_array_size=100):
		"""
		compute and store A, invA
		A is computed through delegation to self.network
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
                if num_proc==1:
                        self.set_A(psi=psi)
			return

                if not self.network:
                        raise ValueError, "set_network() first"
                if (self.theta == None) or (self.phi == None):
                        raise ValueError, "set_theta_phi() first"

                n_pix, theta, phi, psi = self.check_theta_phi_psi(self.theta, self.phi, psi)
                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

                self.A = np.empty((n_pix, self.n_freqs, self.n_pol, self.n_pol), float)
		self.invA = np.empty_like(self.A)
                for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

                        	        ### fill in data
                                	shape = np.shape(self.A[start:end])
	                                self.A[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
					self.invA[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
			_psi = psi[start:end]

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.__set_A_mp, args=(_theta, _phi, _psi, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )

                while len(procs):
                        p, start, end, con1 = procs.pop(0)
                        shape = np.shape(self.A[start:end])
                        self.A[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
			self.invA[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

	###
	def set_B(self, psi=0.0):
		"""
		compute and store B
		B is computed through delegation to self.network
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
		if not self.network:
			raise ValueError, "set_network() first"

                if (self.theta==None) or (self.phi==None):
			raise ValueError, "set_theta_phi() first"

		self.B = self.network.B(self.theta, self.phi, psi, no_psd=False)

	###
	def __set_B_mp(self, theta, phi, psi, connection, max_array_size=100):
                """
                a helper method for set_B_mp
                """
                B = self.network.B(theta, phi, psi, no_psd=False)
                utils.flatten_and_send(connection, B, max_array_size=max_array_size)

        ###
        def set_B_mp(self, psi=0.0, num_proc=1, max_proc=1, max_array_size=100):
                """
                compute and store B
                B is computed through delegation to self.network
                pixelization is defined by healpy and self.nside (through self.angPrior)
                """
                if num_proc==1:
                        self.set_B(psi=psi)
			return

                if not self.network:
                        raise ValueError, "set_network() first"
                if (self.theta == None) or (self.phi == None):
                        raise ValueError, "set_theta_phi() first"

                n_pix, theta, phi, psi = self.check_theta_phi_psi(self.theta, self.phi, psi)
                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

                self.B = np.empty((n_pix, self.n_freqs, self.n_pol, self.n_ifo), complex)
                for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

        	                        ### fill in data
	                                shape = np.shape(self.B[start:end])
                	                self.B[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
                        _psi = psi[start:end]

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.__set_B_mp, args=(_theta, _phi, _psi, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )

                while len(procs):
                        p, start, end, con1 = procs.pop(0)
                        shape = np.shape(self.B[start:end])
                        self.B[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

	###
	def set_AB(self, psi=0.0):
		""" computes and stores both A and B """
		if not self.network:
			raise ValueError, "set_network() first!"
		if (self.theta==None) or (self.phi==None):
			raise ValueError, "set_theta_phi() first"

		self.A, self.B = self.network.AB(self.theta, self.phi, psi, no_psd=False)
		self.invA = linalg.inv(self.A)

        ###
        def __set_AB_mp(self, theta, phi, psi, connection, max_array_size=100):
                """
                a helper method for set_AB_mp
                """
                A, B = self.network.AB(theta, phi, psi, no_psd=False)
		invA = linalg.inv(A)
		utils.flatten_and_send(connection, A, max_array_size=max_array_size)
		utils.flatten_and_send(connection, invA, max_array_size=max_array_size)
                utils.flatten_and_send(connection, B, max_array_size=max_array_size)

        ###
        def set_AB_mp(self, psi=0.0, num_proc=1, max_proc=1, max_array_size=100):
                """
                compute and store A, invA, B
                A,B are computed through delegation to self.network
                pixelization is defined by healpy and self.nside (through self.angPrior)
                """
                if num_proc==1:
                        self.set_AB(psi=psi)
			return 

                if not self.network:
                        raise ValueError, "set_network() first"
                if (self.theta == None) or (self.phi == None):
                        raise ValueError, "set_theta_phi() first"

                n_pix, theta, phi, psi = self.check_theta_phi_psi(self.theta, self.phi, psi)
                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

		self.A = np.empty((n_pix, self.n_freqs, self.n_pol, self.n_pol), float)
		self.invA = np.empty_like(self.A, float)
                self.B = np.empty((n_pix, self.n_freqs, self.n_pol, self.n_ifo), complex)
                for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

                                        ### fill in data
                                        shape = np.shape(self.A[start:end])
					self.A[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
					self.invA[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

                                        shape = np.shape(self.B[start:end])
                                        self.B[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
                        _psi = psi[start:end]

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.__set_AB_mp, args=(_theta, _phi, _psi, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )

                while len(procs):
                        p, start, end, con1 = procs.pop(0)

                        shape = np.shape(self.A[start:end])
			self.A[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
			self.invA[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
		
                        shape = np.shape(self.B[start:end])
                        self.B[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

	###
	def set_dataB(self):
		"""
		computes and stores data*B
		B is required to be present through self.B
		"""
		if self.data==None:
			raise ValueError, "set_data() first!"
		if self.B==None:
			raise ValueError, "set_B() first!"

		n_freqs = self.n_freqs
	        n_pol = self.n_pol
        	n_ifo = self.n_ifo
		n_pix = self.n_pix

		### this implementation is faster than your numpy array manipulations
		self.dataB = np.zeros((n_pix, n_freqs, n_pol), complex) ### transformed data
                for alpha in xrange(n_ifo):
                	for j in xrange(n_pol):
                        	self.dataB[:,:,j] += self.data[:,alpha] * self.B[:,:,j,alpha]

		self.dataB_conj = np.conjugate(self.dataB)


        def __set_dataB_mp(self, n_pix, B, connection, max_array_size=100):
                """
                a helper method for set_dataB_mp
                """
		n_pix, n_freqs, n_pol, n_ifo, B = self.check_B(B, n_pix, self.n_pol)
		dataB = np.zeros((n_pix, n_freqs, n_pol), complex)
		for alpha in xrange(n_ifo):
			for j in xrange(n_pol):
				dataB[:,:,j] += self.data[:,alpha] * B[:,:,j,alpha]

		utils.flatten_and_send(connection, dataB, max_array_size=max_array_size)

	###
	def set_dataB_mp(self, num_proc=1, max_proc=1, max_array_size=100):
                if num_proc==1:
                        self.set_dataB()
			return

		if self.data==None:
                        raise ValueError, "set_data() first!"
                if self.B==None:
                        raise ValueError, "set_B() first!"

		n_pix, n_freqs, n_pol, n_ifo, B = self.check_B(self.B, self.n_pix, self.n_pol)

                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

                self.dataB = np.empty((n_pix, n_freqs, n_pol), complex)
                for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

                                        ### fill in data
                                        shape = np.shape(self.dataB[start:end])
                                        self.dataB[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        ### pull out only those ranges of relevant arrays
                        _B = self.B[start:end]

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.__set_dataB_mp, args=(len(_B), _B, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )

                while len(procs):
                        p, start, end, con1 = procs.pop(0)

                        shape = np.shape(self.dataB[start:end])
                        self.dataB[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

	###
	def set_P(self):
		"""
		comptue and store P, invP
		P is computed with knowledge of self.A and self.hPrior
		pixelization is defined by healpy and self.nside (through self.angPrior)
		"""
		if self.A == None:
			raise ValueError, "set_A() first"
		n_pix, n_freqs, n_pol, A = self.check_A(self.A, self.n_pix, self.n_pol)

		n_gaus = self.hPrior.n_gaus
		self.P = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), complex) ### (A+Z)
		self.invP = np.empty_like(self.P, complex)
		self.detinvP = np.empty((n_pix, n_freqs, n_gaus), complex)
                for g in xrange(n_gaus):
			Z = np.empty_like(A, complex)
			Z[:] = self.hPrior.invcovariance[:,:,:,g]

			P = A+Z

                        self.P[:,:,:,:,g] = P
			self.invP[:,:,:,:,g] = linalg.inv( P ) ### this appears to be VERY memory intensive...
			                                             ### perhaps prohibitively so
			self.detinvP[:,:,g] = 1.0/linalg.det( P )

        ###
        def __set_P_mp(self, n_pix, A, connection, max_array_size=100):
                """
                a helper method for set_AB_mp
                """

		n_pix, n_freqs, n_pol, A = self.check_A(A, n_pix, self.n_pol)
		n_gaus = self.hPrior.n_gaus

		P = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), complex)
		invP = np.empty_like(P, complex)
		detinvP = np.empty((n_pix, n_freqs, n_gaus), complex)
		for g in xrange(n_gaus):
			Z = np.empty_like(A, complex)
			Z[:] = self.hPrior.invcovariance[:,:,:,g]

			this_P = A+Z

			P[:,:,:,:,g] = this_P
			invP[:,:,:,:,g] = linalg.inv( this_P )
			detinvP[:,:,g] = linalg.det( this_P )
		utils.flatten_and_send(connection, P, max_array_size=max_array_size)
		utils.flatten_and_send(connection, invP, max_array_size=max_array_size)
		utils.flatten_and_send(connection, detinvP, max_array_size=max_array_size)

        ###
        def set_P_mp(self, num_proc=1, max_proc=1, max_array_size=100):
                """
                compute and store A, invA, B
                A,B are computed through delegation to self.network
                pixelization is defined by healpy and self.nside (through self.angPrior)
                """
                if num_proc==1:
                        self.set_P()
			return

                if self.A == None:
			raise ValueError, "set_A() first"
		n_pix, n_freqs, n_pol, A = self.check_A(self.A, self.n_pix, self.n_pol)

		n_gaus = self.hPrior.n_gaus
                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

                self.P = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), complex) ### (A+Z)
                self.invP = np.empty_like(self.P, complex)
		self.detinvP = np.empty((n_pix, n_freqs, n_gaus), complex)
                for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

                                        ### fill in data
                                        shape = np.shape(self.P[start:end])
                                        self.P[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)
                                        self.invP[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)
					shape = np.shape(self.detinvP[start:end])
					self.detinvP[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _A = A[start:end] ### pull out only those ranges of relevant arrays

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.__set_P_mp, args=(len(_A), _A, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )

                while len(procs):
                        p, start, end, con1 = procs.pop(0)

                        shape = np.shape(self.P[start:end])
                        self.P[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)
                        self.invP[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)
			shape = np.shape(self.detinvP[start:end])
			self.detinvP[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

	#=========================================
	# check_* functions
	#=========================================

	###
	def check_theta_phi_psi(self, theta, phi, psi):
		""" delegates to utils.check_theta_phi_psi() """
		return utils.check_theta_phi_psi(theta, phi, psi)

	###
	def check_A(self, A, n_pix, n_pol_eff):
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
	def check_B(self, B, n_pix, n_pol_eff):
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
	def check_dataB(self, dataB, n_pix, n_pol_eff):
		""" check data*B's shape """
		if len(np.shape(dataB)) != 3:
			raise ValueError, "bad shape for data*B"
		n, n_freqs, n_pol = np.shape(dataB)
		if n != n_pix:
			raise ValueError, "inconsistent shape between theta, phi, psi and data*B"
		if n_freqs != self.n_freqs:
			raise ValueError, "inconsistent n_freqs"
		if np.any(n_pol != n_pol_eff):
			raise ValueError, "n_pol != n_pol_eff"

		return n_pix, n_freqs, n_pol, dataB

	###
	def check_P(self, P, n_pix, n_pol_eff):
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
	def check_detP(self, detP, n_pix):
		""" checks detP's shape """
		if len(np.shape(detP)) != 3:
			raise ValueError, "bad shape for detP"
		n, n_freqs, n_gaus = np.shape(detP)
		if n != n_pix:
			raise ValueError, "inconsistent n_pix"
		if n_freqs != self.n_freqs:
			raise ValueError, "inconsistent n_freqs"
		if n_gaus != self.hPrior.n_gaus:
			raise ValueError, "inconsistent n_gaus"

		return n_pix, n_freqs, n_gaus, detP

	###
	def check_log_posterior_elements(self, log_posterior_elements, n_pix):
		""" checks log_posterior_element's shape """
		if len(np.shape(log_posterior_elements)) != 3:
			raise ValueError, "bad shape for log_posterior_elements"
		n, n_gaus, n_freqs = np.shape(log_posterior_elements)
		if n != n_pix:
			raise ValueError, "inconsistent n_pix"
		if n_gaus != self.hPrior.n_gaus:
			raise ValueError, "inconsistent n_gaus"
		if n_freqs != self.n_freqs:
			raise ValueError, "inconsistent n_freqs"

		return n_pix, n_gaus, n_freqs, log_posterior_elements

	def check_mle_strain(self, mle_strain, n_pix):
		"""checks mle_strain's shape """
		if len(np.shape(mle_strain)) != 3:
			raise ValueError, "bad shape for mle_strain"
		n, n_freqs, n_pol = np.shape(mle_strain)
		if n != n_pix:
			raise ValueError, "inconsistent n_pix"
		if n_freqs != self.n_freqs:
			raise ValueError, "inconsistent n_freqs"
		if n_pol != self.n_pol:
			raise ValueError, "inconsistent n_pol"

		return n_pix, n_freqs, n_pol, mle_strain

	#=========================================
	# MAIN analysis functionality
	#=========================================

	###
	def n_pol_eff(self, theta, phi):
		"""
		computes the effective number of polarizations given the network and theta, phis
		
		right now, this is a trivial delegation
		"""
		n_pix, theta, phi, psi = self.check_theta_phi_psi(theta, phi, 0.0)
		return self.n_pol*np.ones((n_pix,),int)

	###		
	def mle_strain(self, theta, phi, psi=0., n_pol_eff=None, invA_dataB=None, connection=None, max_array_size=100):
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
		n_pix, theta, phi, psi = self.check_theta_phi_psi(theta, phi, psi)

		# effective number of polarizations
		if (n_pol_eff == None):
			n_pol_eff = self.n_pol
		else:
			n_pol_eff = np.max(n_pol_eff)

		### check invA_B
		if invA_dataB:
			invA, dataB = invA_dataB
			if n_pix == 1:
				if len(np.shape(invA)) == 3:
					invA = np.array([invA])
				else:
					raise ValueError, "bad shape for invA"
				if len(np.shape(B)) == 3:
					B = np.array([B])
				else:
					raise ValueError, "bad shape for B"

			n_pix, n_freqs, n_pol, invA = self.check_A(invA, n_pix, n_pol_eff)
			n_pix, n_freqs, n_pol, dataB = self.check_dataB(dataB, n_pix, n_pol_eff)
				
		else:
			n_freqs = self.n_freqs
			n_ifo = self.n_ifo
			invA = linalg.inv(self.network.A(theta, phi, psi, no_psd=False)) ### may throw errors due to singular matrix
			B = self.network.B(theta, phi, psi, no_psd=False)
			dataB = np.zeros((n_pix, n_freqs, n_pol_eff), complex)
                	for alpha in xrange(n_ifo):
                        	for j in xrange(n_pol_eff):
                                	dataB[:,:,j] += self.data[:,alpha] * B[:,:,j,alpha]
		
					
		### Calculate h_ML
		mle_h = np.zeros((n_pix, n_freqs, n_pol_eff), complex)
		for j in xrange(n_pol_eff):
			for k in xrange(n_pol_eff):
				mle_h[:,:,j] += invA[:,:,j,k] * dataB[:,:,k]

		if connection:
			utils.flatten_and_send(connection, mle_h, max_array_size=max_array_size)
		elif n_pix == 1:
			return mle_h[0]
		else:
			return mle_h

	###
	def mle_strain_mp(self, theta, phi, psi, num_proc=1, max_proc=1, max_array_size=100, n_pol_eff=None, invA_dataB=None):
		"""
		multiprocessing equivalent of mle_strain
		"""
		if num_proc==1:
                        return self.mle_strain(theta, phi, psi, n_pol_eff=n_pol_eff, invA_dataB=invA_dataB)

		### check that we have data
                if self.data==None:
                        raise ValueError, "set_data() first"

                ### check angular coords
                n_pix, theta, phi, psi = self.check_theta_phi_psi(theta, phi, psi)
		if n_pix == 1: ### parallelization will not help you here
			return self.mle_strain(theta, phi, psi, n_pol_eff=n_pol_eff, invA_dataB=invA_dataB)

                # effective number of polarizations
                if (n_pol_eff == None):
                        n_pol_eff = self.n_pol
                else:
                        n_pol_eff = np.max(n_pol_eff)

                ### instantiate arrays
		n_freqs = self.n_freqs
                mle_h = np.zeros((n_pix, n_freqs, n_pol_eff), complex)

                ### define size of jobs
                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

		if invA_dataB != None:
			invA, dataB = invA_dataB

                ### iterate through jobs
                for i in xrange(num_proc):

                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

                	                ### fill in data
                        	        shape = np.shape(mle_h[start:end])
	                                mle_h[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
                        _psi = psi[start:end]
			if invA_dataB != None:
				_invA_dataB = (invA[start:end], dataB[start:end])
			else:
				_invA_dataB = None

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.mle_strain, args=(_theta, _phi, _psi, n_pol_eff, _invA_dataB, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )

		while len(procs):
			p, start, end, con1 = procs.pop(0)

			shape = np.shape(mle_h[start:end])
                        mle_h[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=complex)

		return mle_h

	### 	
	def log_posterior_elements(self, theta, phi, psi=0.0, invP_dataB=None, A_invA=None, connection=None, max_array_size=100, diagnostic=False):
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
		n_pix, theta, phi, psi = self.check_theta_phi_psi(theta, phi, psi)

		### pull out number of gaussian terms
                n_gaus = self.hPrior.n_gaus

		### pull out number of frequencies
		n_freqs = self.n_freqs

		### check invP_B
		if invP_dataB:
			invP, detinvP, dataB, dataB_conj = invP_dataB
			if n_pix == 1:
				if len(np.shape(invP)) == 4:
                                        invP = np.array([invP])
                                else:
                                        raise ValueError, "bad shape for invP"
				if len(np.shape(detinvP)) == 2:
					detinvP = np.array([detinvP])
				else:
					raise ValueError, "bad shape for detinvP"
				if len(np.shape(dataB)) == 2:
					dataB = np.array([dataB])
				else:
					raise ValueError, "bad shape for dataB"
				if len(np.shape(dataB_conj)) == 2:
					dataB_conj = np.array([dataB_conj])
				else:
					raise ValueError, "bad shape for dataB_conj"

			n_pix, n_freqs, n_pol, n_gaus, invP = self.check_P(invP, n_pix, self.n_pol)
			n_pix, n_freqs, n_gaus, detinvP = self.check_detP(detinvP, n_pix)
			n_pix, n_freqs, n_pol, dataB = self.check_dataB(dataB, n_pix, self.n_pol)
			n_pix, n_freqs, n_pol, dataB_conj = self.check_dataB(dataB_conj, n_pix, self.n_pol)
		else:
			A = self.network.A(theta, phi, psi, no_psd=False)
			invP = np.empty((n_pix, n_freqs, n_pol, n_pol, n_gaus), complex) ### (A+Z)^{-1}
			detinvP = np.empty((n_pix, n_freqs, n_gaus), complex)
			for g in xrange(n_gaus):
	                        Z = np.empty_like(A, complex)
        	                Z[:] = self.hPrior.invcovariance[:,:,:,g]
        	                invP[:,:,:,:,g] = linalg.inv(A + Z)
				detinvP[:,:,g] = linalg.det( invP[:,:,:,:,g] )
                        B = self.network.B(theta, phi, psi, no_psd=False)
			dataB = np.zeros((n_pix, n_freqs, n_pol), complex) ### transformed data
                        for alpha in xrange(n_ifo):
                                for j in xrange(n_pol):
                                        dataB[:,:,j] += self.data[:,alpha] * B[:,:,j,alpha]
                        dataB_conj = np.conjugate(dataB)
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

				n_pix, n_freqs, n_pol, A = self.check_A(A, n_pix, self.n_pol)
				n_pix, n_freqs, n_pol, invA = self.check_A(invA, n_pix, self.n_pol)
			else:
				A = self.network.A(theta, phi, psi, no_psd=False)
				invA = linalg.inv(A)

			h_mle = self.mle_strain(theta, phi, psi, n_pol_eff=n_pol_eff, invA_dataB=(invA,dataB))
			h_mle_conj = np.conjugate(h_mle)

			means = self.hPrior.means  ### required to be zeros to compute "ans", but may be non-zero here...
                        means_conj = np.conj(means)

			mle = np.zeros_like(ans, float) ### maximum likelihood estimate's contribution
			cts = np.zeros_like(ans, float) ### term from completing the square in the marginalization
			det = np.zeros_like(ans, float) ### term from determinant from marginalization
	
		### define frequency spacing (important for determinant terms)
#		df = self.seglen**-1
#		npol_logdf = np.log(df)*n_pol

		### compute ans
		for g in xrange(n_gaus):
			for j in xrange(n_pol):
				for k in xrange(n_pol):
					ans[:,g,:] += (dataB_conj[:,:,j] * invP[:,:,j,k,g] * dataB[:,:,k]).real ### we keep only the real part
			detZ = self.hPrior.detinvcovariance[:,g]
			ans[:,g,:] += ( np.log( detinvP[:,:,g]) + np.log( detZ ) ).real * self.seglen ### we keep only the real part
#			ans[:,g,:] += ( np.log( detinvP[:,:,g]) - npol_logdf ) / df ### CURRENT NORMALIZATION SCHEME USES UN-NORMALIZED KERNALS

		### diagnostic arrays
		if diagnostic:
			### mle
			_mle = np.zeros((n_pix, n_freqs), float)
			for j in xrange(n_pol):
				for k in xrange(n_pol):
					_mle += (h_mle_conj[:,:,j] * A[:,:,j,k] * h_mle[:,:,k]).real ### take only the real part
			for g in xrange(n_gaus):
				mle[:,g,:] = _mle

			### cts		
			for g in xrange(n_gaus):
				Z = self.hPrior.invcovariance[:,:,:,g]
				for j in xrange(n_pol):
					diff_conj = h_mle_conj[:,:,j] - means_conj[:,j,g]
					for k in xrange(n_pol):
						diff = h_mle[:,:,k] - means[:,k,g]
						
						cts[:,g,:] -= (diff_conj * Z[:,j,k] * diff).real ### take only real part
						for m in xrange(n_pol):
							for n in xrange(n_pol):
								cts[:,g,:] += (diff_conj * Z[:,j,m] * invP[:,:,m,n,g] * Z[:,n,k] * diff).real ### take only real part

			### det
			for g in xrange(n_gaus):
				detZ = self.hPrior.detinvcovariance[:,g]
				det[:,g,:] = ( np.log( detinvP[:,:,g]) + np.log( detZ ) ).real * self.seglen ### tak only the real part
				                                                                    ### determinant includes detZ because we need to normalize the individual frequencies
				                                                                    ### see write-up for tentative rationalization
				                                                                    ### practically, this is needed to prevent determinant-domination for all terms
				                                                                    ###   without these controlling detZ terms, the determinant can diverge around singular points for A
				                                                                    ###   in the limit of large numbers of frequency bins.
#				det[:,g,:] = ( np.log( detinvP[:,:,g]) - npol_logdf ) / df ### CURRENT NORMALIZATION SCHEME USES UN-NORMALIZED KERNALS

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
		else:
			if diagnostic:
				return ans, n_pol_eff, (mle, cts, det)
			else:
				return ans, n_pol_eff

	###
	def log_posterior_elements_mp(self, theta=None, phi=None, psi=0.0,  invP_dataB=None, A_invA=None, num_proc=1, max_proc=1, max_array_size=100, diagnostic=False):
		"""
		divides the sky into jobs and submits them through multiprocessing

		if theta==None or phi==None:
			pixelization is delegated to healpix and we compute the posterior for all points in the sky
			we ignore invP_B, A_invA if they are supplied because we have no reference for angPrior

		actual computation is delegated to __log_posterior()
		"""
		### get positions in the sky	
		if theta==None or phi==None:
			theta, phi = hp.pix2ang(self.nside, np.arange(self.n_pix))
			invP_B = A_invA = None ### we ignore these because we have no reference for positions?

		n_pix, theta, phi, psi = self.check_theta_phi_psi(theta, phi, psi)

		if (num_proc == 1) or (n_pix == 1):
			return self.log_posterior_elements(theta, phi, psi=psi, invP_dataB=invP_dataB, A_invA=A_invA, diagnostic=diagnostic)

		if invP_dataB != None:
			invP, detinvP, dataB, dataB_conj = invP_dataB

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

			if A_invA!=None:
				A, invA = A_invA

		### define size of jobs
		npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
		procs = []

		### iterate through jobs
		for i in xrange(num_proc):

			if len(procs): ### if we already have some processes launched
				### reap old processes
				if len(procs) >= max_proc:
					p, start, end, con1 = procs.pop(0)

					### fill in data
					shape = np.shape(ans[start:end])
					ans[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
					n_pol_eff[start:end] = utils.recv_and_reshape(con1, np.shape(n_pol_eff[start:end]), max_array_size=max_array_size, dtype=int)
					if diagnostic:
						mle[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
						cts[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
						det[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

			### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
			_psi = psi[start:end]
			if invP_dataB!=None:
				_invP_dataB = (invP[start:end], detinvP[start:end], dataB[start:end], dataB_conj[start:end])
			else:
				_invP_dataB = None
			if diagnostic and A_invA!=None:
				_A_invA = (A[start:end], invA[start:end])
			else:
				_A_invA = None

			### launch process
			con1, con2 = mp.Pipe()
			p = mp.Process(target=self.log_posterior_elements, args=(_theta, _phi, _psi, _invP_dataB, _A_invA, con2, max_array_size, diagnostic))
			p.start()
			con2.close()
			procs.append( (p, start, end, con1) )

		### reap remaining processes
		while len(procs):
			p, start, end, con1 = procs.pop(0)
			### fill in data
                        shape = np.shape(ans[start:end])
                        ans[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
                        n_pol_eff[start:end] = utils.recv_and_reshape(con1, np.shape(n_pol_eff[start:end]), max_array_size=max_array_size, dtype=int)
                        if diagnostic:
                        	mle[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
                                cts[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)
                                det[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

		if diagnostic:
			return ans, n_pol_eff, (mle, cts, det)
		else:
			return ans, n_pol_eff


	#=========================================
	# manipulation functions
	#=========================================

	###
	def log_posterior(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False, connection=None, max_array_size=100):
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
		n_pix, thetas, phis, psis = self.check_theta_phi_psi(thetas, phis, 0.0)
		n_pix, n_gaus, n_freqs, log_posterior_elements = self.check_log_posterior_elements(log_posterior_elements, n_pix)

		### sum over frequencies
		df = self.seglen**-1 ### frequency spacing
		summed_log_posterior_elements = np.sum(log_posterior_elements[:,:,freq_truth], axis=2) * df ### df is included because we're approximating an integral

		### sum over gaussians
		log_posterior_weight = utils.sum_logs(summed_log_posterior_elements, coeffs=self.hPrior.amplitudes) ### amplitudes are the coefficients
		                                                                                                    ### sum over gaussians is implicitly assumed within utils.sum_logs (axis=-1)
		
		### add normalization for prior using the model embodied by "freq_truth"
		log_posterior_weight += self.hPrior.lognorm(freq_truth)

		### add contribution of angular Prior
		log_posterior_weight += np.log( self.angPrior(theta=thetas, phi=phis) )
	
		'''	
		#Find normalization factor for prior on h
		eff_num_pol = g_array[:,:,0,1]  #2-D array (pixels * Gaussians)
		log_var_terms = (eff_num_pol*num_f/2.)*np.log10([self.hPrior.covariance[0,0,0,:]]*self.n_pix)  #2-D array (pixels * Gaussians)
		log_var_max = np.amax(log_var_terms, axis=1)
		log_var_max_array = np.ones((self.n_pix,num_g))*np.array([log_var_max]*num_g).transpose()
		log_var_terms -= log_var_max_array
		hpri_sum_terms = amplitudes_unnormed * np.power(10., log_var_terms)
		log_hpri_norm = (eff_num_pol[:,0]*num_f/2.)*np.log10(np.pi*self.seg_len/2.) + log_var_max + np.log10( np.sum(hpri_sum_terms, axis=1))
		log_posterior_weight -= log_hpri_norm
		'''

		if normalize:
			log_posterior_weight -= utils.sum_logs(log_posterior_weight)

		if connection:
			utils.flatten_and_send(connection, log_posterior_weight, max_array_size=max_array_size)
		else:
			return log_posterior_weight
	
	###
	def log_posterior_mp(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False, num_proc=1, max_proc=1, max_array_size=100):
		if num_proc==1:
			return self.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize)

		n_pix, theta, phi, psi = self.check_theta_phi_psi(thetas, phis, 0.0)
		npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
		procs = []

		log_posterior_weight = np.empty((n_pix,),float)
		for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, start, end, con1 = procs.pop(0)

                        	        shape = np.shape(log_posterior_weight[start:end])
                                	log_posterior_weight[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _theta = theta[start:end] ### pull out only those ranges of relevant arrays
                        _phi = phi[start:end]
                        _lpe = log_posterior_elements[start:end]

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.log_posterior, args=(_theta, _phi, _lpe, n_pol_eff, freq_truth, False, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, start, end, con1) )
		while len(procs):
			p, start, end, con1 = procs.pop(0)
			shape = np.shape(log_posterior_weight[start:end])
                        log_posterior_weight[start:end] = utils.recv_and_reshape(con1, shape, max_array_size=max_array_size, dtype=float)

		if normalize:
			log_posterior_weight -= utils.sum_logs(log_posterior_weight)
	
		return log_posterior_weight

	###
	def posterior(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False, connection=None, max_array_size=100):
		""" np.exp(self.log_posterior_weight) """
		if connection:
			utils.flatten_and_send(connection, np.exp( self.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize) ), max_array_size=max_array_size)
		else:
			return np.exp( self.log_posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize) )

	###
	def posterior_mp(self, thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=False, num_proc=1, max_proc=1, max_array_size=100):
		if num_proc==1:
			return self.posterior(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize)

		return np.exp( self.log_posterior_mp(thetas, phis, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size) )

	###
	def log_bayes(self, log_posterior, connection=None, max_array_size=100):
		""" 
		computes the log(bayes factor) using log_posterior 
		this is done through direct marginalization over all n_pix
		"""
		if connection:
			connection.send( utils.sum_logs(log_posterior) )
		else:
			return utils.sum_logs(log_posterior)

	###
	def log_bayes_mp(self, log_posterior, num_proc=1, max_proc=1, max_array_size=100):
		if num_proc==1:
			return self.log_bayes(log_posterior)

		n_pix = len(log_posterior)
                npix_per_proc = int(np.ceil(1.0*n_pix/num_proc))
                procs = []

		_log_bayes = np.empty((num_proc),float)
                for i in xrange(num_proc):
                        if len(procs): ### if we already have some processes launched
                                ### reap old processes
                                if len(procs) >= max_proc:
                                        p, iproc, start, end, con1 = procs.pop(0)
					_log_bayes[iproc] = con1.recv()

                        ### launch new process
                        start = i*npix_per_proc ### define ranges of pixels for this process
                        end = start + npix_per_proc

                        _lp = log_posterior[start:end]

                        ### launch process
                        con1, con2 = mp.Pipe()
                        p = mp.Process(target=self.log_bayes, args=(_lp, con2, max_array_size))
                        p.start()
                        con2.close()
                        procs.append( (p, i, start, end, con1) )
                while len(procs):
                        p, iproc, start, end, con1 = procs.pop(0)
			_log_bayes[iproc] = con1.recv()

                return utils.sum_logs( _log_bayes )

	###
	def bayes(self, log_posterior):
		""" np.exp(self.log_bayes(log_posterior)) """
		return np.exp( self.log_bayes(log_posterior) )

	###
	def bayes_mp(self, log_posterior, num_proc=1, max_proc=1, max_array_size=100):
		if num_proc==1:
			return self.bayes(log_posterior)

		return np.exp( self.log_bayes_mp(log_posterior, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size) )

	###
	def plot(self, figname, posterior=None, title=None, unit=None, inj=None, est=None, graticule=False, min=None, max=None):
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
		if min!=None and max!=None:
			hp.mollview(posterior, title=title, unit=unit, flip="geo", fig=fig_ind, min=min, max=max)
		elif min!=None:
			hp.mollview(posterior, title=title, unit=unit, flip="geo", fig=fig_ind, min=min)
		elif max!=None:
			hp.mollview(posterior, title=title, unit=unit, flip="geo", fig=fig_ind, max=max)
		else:
	                hp.mollview(posterior, title=title, unit=unit, flip="geo", fig=fig_ind)
		if graticule:
	                hp.graticule()

                ### plot point if supplied
                if inj:
                        ax = fig.gca()
                        marker = ax.projplot(inj, "wx")[0]
                        marker.set_markersize(10)
                        marker.set_markeredgewidth(2)
#			marker.set_alpha(0.5)

		if est:
			ax = fig.gca()
			marker = ax.projplot(est, "wo")[0]
			marker.set_markersize(10)
			marker.set_markeredgewidth(2)
			marker.set_markeredgecolor("w")
			marker.set_markerfacecolor("none")
#			marker.set_alpha(0.5)

                ### save
                fig.savefig(figname)
                plt.close()
	
	###
	def __call__(self, normalize=True, num_proc=1, max_proc=1, max_array_size=100):
		"""
		calculate the posterior at all points in the sky defined by self.nside and healpy
		we use all frequency bins

		we may want to run everything in parallel instead of recombining the outputs at each step
			we will save time on communicating?
			although we will lose time in the end?

		add optional arguments for theta, phi and pass those along via delegation?
		"""
		### check for pre-computed arrays
		if self.theta==None or self.phi==None:
			self.set_theta_phi()
		theta = self.theta
		phi = self.phi

		if self.invP==None:
			self.set_P_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		if self.B==None:
			self.set_B_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		if self.dataB==None:
			self.set_dataB_mp(num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)
		invP_dataB = (self.invP, self.detinvP, self.dataB, self.dataB_conj)

		log_posterior_elements, n_pol_eff = self.log_posterior_elements_mp(theta, phi, psi=0.0, invP_dataB=invP_dataB, A_invA=None, diagnostic=False, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)

		freq_truth = np.ones((self.n_freqs,), bool)

		return self.posterior_mp(theta, phi, log_posterior_elements, n_pol_eff, freq_truth, normalize=normalize, num_proc=num_proc, max_proc=max_proc, max_array_size=max_array_size)

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

