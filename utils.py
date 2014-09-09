### written by R.Essick (ressick@mit.edu)

usage = """ a general utilities module for sky localization. All distances are measured in seconds """

import numpy as np
from numpy import linalg
import healpy as hp
import pickle

#=================================================
#
#            general utilities
#
#=================================================
deg2rad = np.pi/180
rad2deg = 1.0/deg2rad

#========================
# multiprocessing
#========================
def flatten_and_send(conn, array, max_array_size=100):
	"""
	flattens and sends an array through a pipe
	"""
	array = array.flatten()
	size = np.size(array)
	i=0
	while i*max_array_size < size:
		conn.send( array[i*max_array_size:(i+1)*max_array_size] )
		i += 1

###
def recv_and_reshape(conn, shape, max_array_size=100, dtype=float):
	"""
	receives a flattened array through conn and returns an array with shape defined by "shape"
	"""
	flat_array = np.empty(shape, dtype).flatten()
	size = np.size(flat_array)
	i=0
	while i*max_array_size < size:
		flat_array[i*max_array_size:(i+1)*max_array_size] = conn.recv()
		i += 1
	return flat_array.reshape(shape)

#========================
# pixelization
#========================
def set_theta_phi(nside, coord_sys="E", **kwargs):
	""" defines that pixelization using Healpix decomposition in the desired coordinate system
	E : Earth-fixed coordinates (theta, phi)
	C : celestial coordinates (Ra, Dec)
	G : galactic coordinates 

	this function "lays the grid" in the correct coordinate system and then rotates the points into theta, phi in Earth-fixed coordinates, which should be used to compute antenna patterns. However, upon evaluation of the likelihood functional and rotation back into the desired frame, the likelihood will be evaluated exactly on the pixel centers defined by the Healpix decomposition.
	"""
	if kwargs.has_key("npix"):
		npix = kwargs["npix"]
	else:
		npix = hp.nside2npix(nside)

	if coord_sys == "E":
		return hp.pix2ang(nside, np.arange(npix))

	elif coord_sys == "C":
		raise StandardError, "WRITE ME"

	elif coord_sys == "G":
		raise StandardError, "WRITE ME"

	else:
		raise ValueError, "coord_sys=%s not understood"%coord_sys

###
def check_theta_phi_psi(theta, phi, psi):
	""" checks angular variables. Casts them to the correct shape and wraps them into the correct ranges
        implemented in one place for convenience """
        ### check theta's shape
        if isinstance(theta, (int, float)):
        	theta = np.array([theta])
        elif not isinstance(theta, np.ndarray):
		theta = np.array(theta)
	if len(np.shape(theta)) != 1:
		raise ValueError, "bad shape for theta"

        ### check phi's shape
        if isinstance(phi, (int, float)):
        	phi = np.array([phi])
        elif not isinstance(phi, np.ndarray):
                phi = np.array(phi)
        if len(np.shape(phi)) != 1:
                raise ValueError, "bad shape for phi"

        ### check psi's shape:
        if isinstance(psi, (int,float)):
                psi = np.array([psi])
        elif not isinstance(psi, np.ndarray):
                psi = np.array(psi)
        if len(np.shape(psi)) != 1:
                raise ValueError, "bad shape for psi"

        ### check whether theta, phi, psi agree on their shape
        len_theta = len(theta)
        len_phi = len(phi)
        len_psi = len(psi)
        n_pix = max(len_theta, len_phi, len_psi)
        if len_theta != n_pix:
        	if len_theta == 1:
                        theta = np.outer(theta, np.ones((n_pix,),float)).flatten()
                else:
                        raise ValueError, "inconsistent size between theta, phi, psi"
        if len_phi != n_pix:
                if len_phi == 1:
                        phi = np.outer(phi, np.ones((n_pix,),float)).flatten()
                else:
                        raise ValueError, "inconsistent size between theta, phi, psi"
        if len_psi != n_pix:
                if len_psi == 1:
                        psi = np.outer(psi, np.ones((n_pix,),float)).flatten()
                else:
                	raise ValueError, "inconsistent size between theta, phi, psi"

        ### enforce bounds for theta, phi, psi
        theta = theta%np.pi
        phi = phi%(2*np.pi)
        psi = psi%(2*np.pi)

        return n_pix, theta, phi, psi


#========================
# arithmatic
#========================
def sum_logs(logs, base=np.exp(1)):
	"""sums an array of logs accurately"""
	if not isinstance(logs, np.ndarray):
		logs = np.array(logs)

	_max = np.max(logs)
	ans = np.sum(base**(logs-_max))

	return np.log(ans)*np.log(base) + _max

#========================
# I/O utilies
#========================
def load_toacache(filename):
	"""
	loads time-of-arrival information from filename
	"""
	file_obj = open(filename, "r")
	toacache = pickle.load(file_obj)
	file_obj.close()
	return toacache

#========================
# timing utilities
#========================
def time_of_flight(theta, phi, dr):
	"""
	computes the time of flight between two points (defined by dr=(dx,dy,dz)) for a plane wave propagating from (theta,phi)
	"""
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)

	dx, dy, dz = dr

	tau = dx*sin_theta*cos_phi + dy*sin_theta*sin_phi + dz*cos_theta

	return tau

#========================
# antenna pattern utilites
#========================
def antenna_patterns(theta, phi, psi, nx, ny, freqs=None, dt=0.0, dr=None):
	"""
	computes the antenna patterns for detector arms oriented along nx and ny (cartesian vectors). 
		if freqs, it computes time-shift phases in the frequency domain using dt. 
		if dr and freq, it will compute dt for itself (save time with cos(theta), etc.

	Antenna patterns are computed accoring to Eqn. B7 from Anderson, et all PhysRevD 63(04) 2003
	"""
	n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)

	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	cos_phi = np.cos(phi)
	sin_phi = np.sin(phi)
	cos_psi = np.cos(psi)
	sin_psi = np.sin(psi)

	Xx = sin_phi*cos_psi - sin_psi*cos_phi*cos_theta
	Xy = -cos_phi*cos_psi - sin_psi*sin_phi*cos_theta
	Xz = sin_psi*sin_theta

	Yx = -sin_phi*sin_psi - cos_psi*cos_phi*cos_theta
	Yy = cos_phi*sin_psi - cos_psi*sin_phi*cos_theta
	Yz = sin_theta*cos_psi

        X = (Xx, Xy, Xz)
        Y = (Yx, Yy, Yz)

	### iterate over x,y,z to compute F+ and Fx
	Fp = np.zeros((n_pix,),float)
	Fx = np.zeros((n_pix,),float)
#	Fp = 0.0 # without freqs, these are scalars
#	Fx = 0.0
	for i in xrange(3):
		nx_i = nx[i]
		ny_i = ny[i]
		Xi = X[i]
		Yi = Y[i]
		for j in xrange(3):
			Xj = X[j]
			Yj = Y[j]
			Dij = 0.5*(nx_i*nx[j] - ny_i*ny[j])
			Fp += (Xi*Xj - Yi*Yj)*Dij
			Fx += (Xi*Yj + Yi*Xj)*Dij

	### apply time-shits
	if freqs != None:
		freqs = np.array(freqs)
		if dr != None:
			dx, dy, dz = dr
			dt = dx*sin_theta*cos_phi + dy*sin_theta*sin_phi + dz*cos_theta
		phs = np.exp(-2j*np.pi*np.outer(dt,freqs))
#		phs = np.exp(-2j*np.pi*freqs*dt)

#		if isinstance(Fp, np.float): # this avoids weird indexing with outer and a single point
#			Fp *= phs
#			Fx *= phs
#		else:
#			Fp = np.outer(Fp, phs)
#			Fx = np.outer(Fx, phs)
		Fp = np.outer(Fp, np.ones(len(freqs)))*phs
		Fx = np.outer(Fx, np.ones(len(freqs)))*phs

	if n_pix == 1:
		return Fp[0], Fx[0]
	else:
		return Fp, Fx

#=================================================
#
#                psd class
#
#=================================================
class PSD(object):
	"""
	an object that holds onto power-spectral densities with associated frequency samples
	we define a scipy.interpolate.interp1d object for convenience
	"""

	###
	def __init__(self, freqs, psd, kind="linear"):
		len_freqs = len(freqs)
		if len(psd) != len_freqs:
			raise ValueError, "freqs and ps must have the same length"
		if not len_freqs:
			raise ValueError, "freqs and psd must have at least 1 entries"
		elif len_freqs == 1:
			freqs = np.array(2*list(freqs))
			psd = np.array(2*list(psd))
#		from scipy.interpolate import interp1d
		self.freqs = freqs
		self.psd = psd
#		self.interp = interp1d(freqs, psd, kind=kind, copy=False)

	###
	def check(self):
		return len(self.freqs) == len(self.psd)

	###
	def update(self, psd, freqs=None):
		if freqs and psd:
			self.freqs = freqs
			self.psd = psd
		else:
			self.psd=psd

	###
	def get_psd(self):
		return self.psd

	###
	def get_freqs(self):
		return self.freqs

	###
	def interpolate(self, freqs):
#		return self.interp(freqs)
		return np.interp(freqs, self.freqs, self.psd)

	###
	def normalization(self, fs, T):
		""" returns the normalization for psd given an FFT defined by
	fs : sampling frequency
	T : duration
	=> No. points = fs*T

		this should be interpreted as a multiplicative factor for psd to compute the expected noise in that frequency bin
		"""
		return fs*T

	###
	def __repr__(self):
		return self.__str__()

	###
	def __str__(self):
		min_psd = np.min(self.psd)
		d=int(np.log10(min_psd))-1
		return """utils.PSD object
	min{freqs}=%.5f
	max{freqs}=%.5f
	No. freqs =%d
	min{psd}=%.5fe%d  at freqs=%.5f"""%(np.min(self.freqs), np.max(self.freqs), len(self.freqs), min_psd*10**(-d), d, self.freqs[min_psd==self.psd])

#=================================================
#
#              detector class
#
#=================================================
class Detector(object):
	"""
	an object representing a gravitational wave detector. methods are meant to be convenient wrappers for more general operations. 
	"""

	###
	def __init__(self, name, dr, nx, ny, psd):
		"""
	        name = None  # detector's name (eg: H1)
	        dr = np.zeros((3,)) # r_detector - r_geocent
        	nx = np.zeros((3,)) # direction of the x-arm
	        ny = np.zeros((3,)) # direction of the y-arm
        	psd = None   # the psd for network (should be power, not amplitude)
		"""
		self.name = name
		if not isinstance(dr, np.ndarray):
			dr = np.array(dr)
		self.dr = dr
		if not isinstance(nx, np.ndarray):
			nx = np.array(nx)		
		self.nx = nx
		if not isinstance(ny, np.ndarray):
			ny = np.array(ny)
		self.ny = ny
		self.psd = psd

	###
	def __str__(self):
		return "Detector : %s"%self.name

	###
	def __repr__(self):
		return self.__str__()

	###
	def set_psd(self, psd, freqs=None):
		self.psd.update(psd, freqs=freqs)

	###
	def get_psd(self):
		return self.psd

	###	
	def dt_geocent(self, theta, phi):
		""" returns t_geocent - t_detector"""
		return _time_of_flight(theta, phi, -self.dr)

	###
	def antenna_patterns(self, theta, phi, psi, freqs=None, dt=None):
		""" returns the antenna patterns for this detector. If psi is not supplied, returns antenna patterns that diagonalize A_{ij} """
		if dt != None:
			return antenna_patterns(theta, phi, psi, self.nx, self.ny, freqs=freqs, dt=dt)
		else:
			return antenna_patterns(theta, phi, psi, self.nx, self.ny, freqs=freqs, dr=self.dr)

	###
	def __repr__(self):
		return self.__str__()

	###
	def __str__(self):
		return """utils.Detector object
	name : %s
	dr = %.5f , %.5f , %.5f 
	nx = %.5f , %.5f , %.5f
	ny = %.5f , %.5f , %.5f
	PSD : %s"""%(self.name, self.dr[0], self.dr[1], self.dr[2], self.nx[0], self.nx[1], self.nx[2], self.ny[0], self.ny[1], self.ny[2], str(self.psd))

#=================================================
#
#              network class
#
#=================================================
class Network(object):
	"""
	an object representing a network of gravitational wave detectors.
	"""

	###
	def __init__(self, detectors=[], freqs=None, Np=2):
		self.freqs = freqs
		self.Np = Np
		self.detectors = {}
		self.set_detectors(detectors)

	###
	def __len__(self):
		"""returns the number of detectors in the network"""
		return len(self.detectors)

	###
	def __str__(self):
		s = """utils.Network object
	min{freqs}=%.5f
	max{freqs}=%.5f
	No. freqs=%d
	No. polarizations = %d
	No. detectors = %d"""%(np.min(self.freqs), np.max(self.freqs), len(self.freqs), self.Np, len(self))

		for name in self.detector_names_list():
			s += "\n\n\t%s : %s"% (name, str(self.get_detectors(name)))
		return s

	###
	def __repr__(self):
		return self.__str__()
	
	###	
	def set_detectors(self, detectors):
		"""add detector(s) to the network"""
		try:
			for detector in detectors:
				self.__add_detector(detector)
		except TypeError: # only a single detector was supplied
			self.__add_detector(detectors)

	###
	def __add_detector(self, detector):
		if self.freqs == None:
			self.freqs = detector.get_psd().get_freqs()
		self.detectors[detector.name] = detector

	###
	def remove_detectors(self, detectors):
		"""remove a detector from the network"""
		try:
			for detector in detectors:	
				self.__remove_detector(detector)
		except TypeError:
			self.__remove_detector(detectors)
	
	###	
	def __remove_detector(self, detector):
		try:
			self.detectors.pop(detector.name)
		except KeyError: # detector was not present
			pass

	###
	def get_detectors(self, names):
		"""returns instances of detector objects stored in this network"""
		if isinstance(names, str):
			detector = self.detectors.get(names)
			if detector == None:
				raise KeyError, "network does not contain %s"%name
			return detector
		else:
			detectors = []
			for name in names:
				detector = self.detectors.get(name)
				if detector == None:
					raise KeyError, "network does not contain %s"%name
				detectors.append(detector)
			return detectors

	###
	def detectors_list(self):
		"""lists detectors in a consistent order"""
		ans = self.detectors.items()
		ans.sort(key=lambda l: l[0]) # sort by detector names
		return [l[1] for l in ans]
		
	def detector_names_list(self):
		"""lists detector names in a consistent order"""
		return sorted(self.detectors)

	###
	def contains_name(self, name):
		"""checks to see if name is associated with any detector in the network"""
		return self.detectors.has_key(name)

	###
	def ang_res(self, f, degrees=False):
		"""computes the minimum angular resolution achievable with this network for a signal at frequency "f" 
		approximates timing uncertainty by dt ~ 0.5/f
		approximates smallest dtheta allowable by dtheta >= dt/(d*sin(theta)) >= dt/d
		returns dtheta (radians)
		"""
		detectors = self.detector_list()
		d = 0.0
		for ind, detector1 in enumerate(detectors):
			for detector2 in detectors[ind:]:
				this_d = abs(np.sum(detector1.dr-detector2.dr)**2) ### compute baseline between detectors (in sec)
				if this_d > d: ### keep only the biggest baseline
					d = this_d
		dtheta = (0.5/f)/d
		if degrees:
			dtheta *= 180/np.pi
		return dtheta





############ we should use network.get_detector(name).antenna_pattern(theta, phi, psi, freqs=freqs) instead!

#	###
#	def F_det(self, det_name, theta, phi, psi=0.):
#		"""Calculates 2-D antenna pattern array (frequencies x polarizations) for a given detector"""
#		F_det = np.zeros((len(self.freqs), self.Np), 'complex')
#		for i_f in xrange(len(self.freqs)):
#			for i_pol in xrange(self.Np):
#				F_det[i_f,i_pol] = self.detectors[det_name].antenna_patterns(theta=theta, phi=phi, psi=psi, freqs=self.freqs)[i_pol][i_f]
#		return F_det
############

	###
	def A(self, theta, phi, psi, no_psd=False):
		"""computes the entire matrix"""
		n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)

		if no_psd:
			a=np.zeros((n_pix, self.Np,self.Np))
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) #time shifts cancel within A so we supply no freqs
				for i in xrange(self.Np):
					a[:,i,i] += np.abs(F[i])**2
					for j in xrange(i+1,self.Np):
						_ = np.conjugate(F[i])*F[j]
						a[:,i,j] += _
						a[:,j,i] += np.conjugate(_)
		else:  #if given a psd
			a=np.zeros((n_pix, len(self.freqs), self.Np, self.Np))  #initialize a 3-D array (frequencies x polarizations x polarizations)
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) #tuple of numbers (pols), time shifts cancel within A so we supply no freqs
				_psd = detector.psd.interpolate(self.freqs)  #1-D array (frequencies)
				for i in xrange(self.Np):
					a[:,:,i,i] += np.outer(np.abs(F[i])**2, _psd**-1)
					for j in xrange(i+1,self.Np):
						_ = np.outer(np.conjugate(F[i])*F[j], _psd**-1)
						a[:,:,i,j] += _
						a[:,:,j,i] += np.conjugate(_)
		if n_pix == 1:
			return a[0]
		else:
			return a

	###
	def Aij(self, i, j, theta, phi, psi, no_psd=False):  #edit this for given psds (see above)
		"""computes a single component of the matrix"""
		n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)

		if no_psd:
			aij = np.zeros((n_pix,), float)
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) # time shifts cancel within A so we supply no freqs
				aij += np.conjugate(F[i])*F[j]
		else:
			aij = np.zeros((n_pix,len(self.freqs)), float)
			for detector in self.detectors.values():
                                F = detector.antenna_patterns(theta, phi, psi, freqs=None)
				_psd = detector.get_psd().interpolate(self.freqs)
				aij += np.outer(np.conjugate(F[i])*F[j], _psd**-1)
		if n_pix == 1:
			return aij[0]
		else:
			return aij

	###
	def B(self, theta, phi, psi, no_psd=False):
		"""computes the entire matrix"""
		n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)

		sorted_detectors = self.detectors_list()
		Nd = len(sorted_detectors)
		if no_psd:
			B = np.zeros((n_pix, len(self.freqs), self.Np, Nd))
			for d_ind, detector in enumerate(sorted_detectors):
				F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs)
				for i in xrange(self.Np):
					B[:,:,i,d_ind] = np.conjugate(F[i])
		else:  #if given a psd
			B = np.zeros((n_pix, len(self.freqs), self.Np, Nd), 'complex') #initialize a 3-D array (freqs x polarizations x detectors)
			for d_ind, detector in enumerate(sorted_detectors):
				F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs)   #tuple (pols) of 1-D arrays (frequencies)
				for i in xrange(self.Np):
						B[:,:,i,d_ind] = np.conjugate(F[i]) * np.outer(np.ones((n_pix,)), detector.psd.interpolate(self.freqs)**-1)
		if n_pix == 1:
			return B[0]
		else:
			return B

	###
	def Bni(self, name, i, theta, phi, psi, no_psd=False):  #edit this for given psds (see above)
		"""computes a single component of the matrix"""
		n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)

		if not self.contains_name(name):
			raise KeyError, "detector=%s not contained in this network"%name
		detector = self.detectors[name]
		F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs)
		if no_psd:
			B = F[:,i]
		else:
			B = F[:,i]* np.outer(np.ones((n_pix,)), detector.get_psd().interpolate(self.freqs)**-1)

		if n_pix == 1:
			return B[0]
		else:
			return B


	###
	def AB(self, theta, phi, psi, no_psd=False):
		""" computes the entire matrices and avoids redundant work """
		
                n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)
		sorted_detectors = self.detectors_list()
		Nd = len(sorted_detectors)
		n_freqs = len(self.freqs)

                if no_psd:
                        a=np.zeros((n_pix, self.Np, self.Np), float)
			b=np.zeros((n_pix, n_freqs, self.Np, Nd), complex)
                        for d_ind, detector in enumerate(sorted_detectors):
                                F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs) #time shifts cancel within A so we supply no freqs
                                for i in xrange(self.Np):
                                        a[:,i,i] += np.abs(F[i])**2
					b[:,:,i,d_ind] = np.conjugate(F[i])
                                        for j in xrange(i+1,self.Np):
                                                _ = np.conjugate(F[i])*F[j]
                                                a[:,i,j] += _
                                                a[:,j,i] += np.conjugate(_)
                else:  #if given a psd
                        a=np.zeros((n_pix, n_freqs, self.Np, self.Np), float)  #initialize a 3-D array (frequencies x polarizations x polarizations)
			b=np.zeros((n_pix, n_freqs, self.Np, Nd), complex)
                        for d_ind, detector in enumerate(sorted_detectors):
                                F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs) #tuple of numbers (pols), time shifts cancel within A so we supply no freqs
                                _psd = detector.psd.interpolate(self.freqs)  #1-D array (frequencies)
                                for i in xrange(self.Np):
                                        a[:,:,i,i] += np.outer(np.abs(F[i])**2, _psd**-1)
					b[:,:,i,det_ind] =  np.conjugate(F[i]) * np.outer(np.ones((n_pix,)), _psd**-1)
                                        for j in xrange(i+1,self.Np):
                                                _ = np.outer(np.conjugate(F[i])*F[j], _psd**-1)
                                                a[:,:,i,j] += _
                                                a[:,:,j,i] += np.conjugate(_)
                if n_pix == 1:
                        return a[0], b[0]
                else:
                        return a, b


	################################################################################
	### things below here are a bit shakey or have not been written
	################################################################################

        ###
        def rank(self, A, tol=1e-10):
                """wrapper for numpy.linalg.matrix_rank that computes the rank of A. Rank is defined as the number of eigenvalues larger than tol"""
                return linalg.matrix_rank(A, tol=tol)

        ###
        def eigvals(self, A):
                """wrappter for numpy.linalg.eigvals that computes the eigenvalues of A"""
                vals = linalg.eigvals(A)
                if len(np.shape(vals)) == 1:
                        vals.sort()
                        return np.diag(vals[::-1]) # return in order of decreasing eigenvalue
                else:
                        v=[]
                        for val in vals:
                                val.sort()
                                v.append( np.diag(val) )
                        return np.array(v)

        ###
        def eig(self, A):
                """wrappter for numpy.linalg.eig that computes the eigenvalues and eigenvectors of A"""
                raise StandardError, "need to figure out how to best sort the eigenvalues and the associated eigenvectors"
                return linalg.eig(A)

	###
	def A_dpf(self, theta, phi, A=None, no_psd=False):
		"""computes A in the dominant polarization frame. If A is supplied, it converts A to the dominant polarization frame"""
		if not A:
			A = self.A(theta, phi, 0.0, no_psd=no_psd)
		return self.eigvals(A)

	###
	def Aii_dpf(self, i, theta, phi, A=None, no_psd=False):
		"""computes a single component of A in the dominant polarization frame. If A is supplied, it converts to the dominant polarizatoin frame"""
		if not A:
			A = self.A(theta, phi, 0.0, no_psd=no_psd)
		return self.eigvals(A)[i,i]

	###
	def B_dpf(self, theta, phi, A_B=None, no_psd=False):
		"""computes B in the dominant polarization frame. If A_B=(A,B) is supplied, we use it to define the dominant polarization frame transformation"""
		raise StandardError, "write me!"

	###
	def Bni(self, name, i, theta, phi, A_B=None, no_psd=False):
		"""computes a single component of B in the dominant polarization frame. If A_B=(A,B) is supplied, we use it to define the dominant polarization frame"""
		raise StandardError, "write me!"

	###
	def AB_dpf(self, theta, phi, A_B=None, no_psd=False):
		""" computes A and B in the dominant polarization frame """
		raise StandardError, "write me!"

#=================================================
#
# physical constants
#
#=================================================
c = 299792458.0 #m/s

