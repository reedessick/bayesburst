### written by R.Essick (ressick@mit.edu)

usage = """ a general utilities module for sky localization. All distances are measured in seconds """

import numpy as np
from numpy import linalg
import healpy as hp
import pickle

from pylal import Fr

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
	flat_array = np.zeros(shape, dtype).flatten()
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
def sum_logs(logs, base=np.exp(1), coeffs=None):
	"""
	sums an array of logs accurately
	if logs has multiple dimensions, sums over the last dimension (axis=-1)
	"""
	if not isinstance(logs, np.ndarray):
		logs = np.array(logs)

	logs_shape = np.shape(logs)

	if coeffs==None:
		coeffs = np.ones(logs_shape[-1],float)
	elif len(coeffs) != logs_shape[-1]:
		raise ValueError, "len(coeffs)=%d != np.shape(logs)[-1]=%d"%(len(coeffs), logs_shapes[axis])

	_max = np.max(logs, axis=-1)
	outer_max = np.reshape( np.outer( np.max(logs, axis=-1), np.ones(logs_shape[-1],float) ), logs_shape)

	ans = np.sum(coeffs*base**(logs-outer_max), axis=-1)

	if np.any(ans < 0):
		raise ValueError, "ans < 0 not allowed!"

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

###
def files_from_cache(cache, start, stop, suffix=".gwf"):
	"""
	selects files from a cache file
	files must have some overlap with [start, stop] to be included
	"""
	files = []
	file_obj = open(cache, "r")
	for line in file_obj:
		line = line.strip()
		s, d = [float(l) for l in line.strip(suffix).split("-")[-2:]]
		if (s+d >= start) and (s < stop):
			files.append( (line, s, d) )
	file_obj.close()

	return files

###
def vec_from_frames(frames, start, stop, verbose=False):
	"""
	returns a numpy array of the data inculded in frames between start and stop
	CURRENTLY ASSUME CONTIGUOUS DATA, but we should check this

	meant to be used with files_from_cache
	"""
	vecs = []
	dt = 0
	for frame, strt, dur in frames:
		if verbose: print frame
		s = max(strt, start)
		d = min(start+dur,stop) - s
		vec, gpstart, offset, dt, _, _ = Fr.frgetvect1d(frame, ifo_chan, start=s, span=d)
		vecs.append( vec )
	vec = np.concatenate(vecs)
	return vec, dt
 
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
		n_freqs = len(freqs)
		if dr != None:
			dx, dy, dz = dr
			dt = dx*sin_theta*cos_phi + dy*sin_theta*sin_phi + dz*cos_theta
			phs = 2*np.pi*np.outer(dt,freqs)
			phs = np.cos(phs) - 1j*np.sin(phs)
		else:
			phs = np.ones((n_pix,n_freqs),float)

		ones_freqs = np.ones((n_freqs),float)
		Fp = np.outer(Fp, ones_freqs) * phs
		Fx = np.outer(Fx, ones_freqs) * phs

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
		self.n_freqs = len_freqs
		if len(psd) != len_freqs:
			raise ValueError, "freqs and ps must have the same length"
		if not len_freqs:
			raise ValueError, "freqs and psd must have at least 1 entries"
		elif len_freqs == 1:
			freqs = np.array(2*list(freqs))
			psd = np.array(2*list(psd))
		self.freqs = freqs
		self.psd = psd

	###
	def check(self):
		return len(self.freqs) == len(self.psd)

	###
	def update(self, psd, freqs=None):
		if freqs!=None:
			if len(freqs)!=len(psd):
				raise ValueError, "len(freqs) != len(psd)"
			self.freqs = freqs[:]
			self.psd = psd[:]
		else:
			self.psd=psd[:] 

	###
	def get_psd(self):
		return self.psd

	###
	def get_freqs(self):
		return self.freqs

	###
	def interpolate(self, freqs):
		return np.interp(freqs, self.freqs, self.psd)

	###
	def draw_noise(self, freqs):
		"""
		draws a noise realization at the specified frequencies
		"""
		n_freqs = len(freqs)

		vars = self.interpolate(freqs) 
		amp = np.random.normal(size=(n_freqs))
		phs = np.random.random(n_freqs)*2*np.pi

		return (amp * vars**0.5) * np.exp(1j*phs)

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
	min{psd}=%.5fe%d  at freqs=%.5f"""%(np.min(self.freqs), np.max(self.freqs), len(self.freqs), min_psd*10**(-d), d, self.freqs[min_psd==self.psd][0])

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
	def snr(self, data, freqs=None):
		""" 
		returns the SNR for data using the PSD stored within the object 
		if freqs==None: assumes data corresponds to self.psd.freqs
		"""
		if freqs==None:
			freqs = self.get_psd().get_freqs()
		if len(data) != len(freqs):
			raise ValueError, "len(data) != len(freqs)"
		
		return ( 4*np.sum((data.real**2+data.imag**2) / self.get_psd().interpolate(freqs))*(freqs[1]-freqs[0]) )**0.5 ### return SNR

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
	def network_snr(self, data):
		"""
		computes network snr through delegation to self.snrs(data)
		"""
		return np.sum(self.snrs(data)**2)**0.5

	###
	def snrs(self, data):
		"""
		computes individual SNRs for each detector in the network
		returns a list of individual SNRs ordered according to self.detectors_list()
			delegates to Detector.snr()
		"""
		detectors_list = self.detectors_list()
		n_ifo = len(detectors_list)
		n_freqs = len(self.freqs)
		if np.shape(data) != (n_freqs, n_ifo):
			raise ValueError, "bad shape for data. expected (%d,%d). recieved "%(n_freqs, n_ifo) + np.shape(data)
		snrs = np.empty((n_ifo,),float)
		for ifo_ind, detector in enumerate(detectors_list):
			snrs[ifo_ind] = detector.snr(data[:,ifo_ind], freqs=self.freqs)

		return snrs
	
	###
	def draw_noise(self):
		"""
		generates a noise realization for the detectors in this network
		"""
		detectors_list = self.detectors_list()
		n_ifo = len(detectors_list)
		n_freqs = len(self.freqs)
		noise = np.empty((n_freqs, n_ifo), complex)

		N = 2*np.max(self.freqs)/(self.freqs[1]-self.freqs[0]) ### normalization for noise realization...
		                                                       ### fs*seglen
		                                                       ### strong chance this is wrong...

		for ifo_ind, detector in enumerate(detectors_list):
			noise[:,ifo_ind] = detector.get_psd().draw_noise(self.freqs)

		return N*noise

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

	###
	def A(self, theta, phi, psi, no_psd=False):
		"""computes the entire matrix"""
		n_pix, theta, phi, psi = check_theta_phi_psi(theta, phi, psi)

		if no_psd:
			a=np.zeros((n_pix, self.Np,self.Np))
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) #time shifts cancel within A so we supply no freqs
				for i in xrange(self.Np):
					a[:,i,i] += F[i].real**2 + F[i].imag**2
					for j in xrange(i+1,self.Np):
						_ = F[i].real*F[j].real + F[i].imag*F[j].imag
						a[:,i,j] += _
						a[:,j,i] += _
		else:  #if given a psd
			a=np.zeros((n_pix, len(self.freqs), self.Np, self.Np), float)  #initialize a 3-D array (frequencies x polarizations x polarizations)
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) #tuple of numbers (pols), time shifts cancel within A so we supply no freqs
				_psd = detector.psd.interpolate(self.freqs)**-1  #1-D array (frequencies)
				for i in xrange(self.Np):
					a[:,:,i,i] += np.outer(F[i]**2, _psd) ### freqs=None -> F is real
					for j in xrange(i+1,self.Np):
						_ = np.outer(F[i]*F[j], _psd) ### freqs=None -> F is real
						a[:,:,i,j] += _
						a[:,:,j,i] += _
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
#				aij += np.outer(np.conjugate(F[i])*F[j], _psd**-1)
				aij += np.outer(np.abs(F[i]*F[j]), _psd**-1)
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
			B = np.zeros((n_pix, len(self.freqs), self.Np, Nd), complex)
			for d_ind, detector in enumerate(sorted_detectors):
				F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs)
				for i in xrange(self.Np):
					B[:,:,i,d_ind] = np.conjugate(F[i])
		else:  #if given a psd
			B = np.zeros((n_pix, len(self.freqs), self.Np, Nd), complex) #initialize a 3-D array (freqs x polarizations x detectors)
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
                                        a[:,i,i] += F[i].real**2 + F[i].imag**2
					b[:,:,i,d_ind] = np.conjugate(F[i])
                                        for j in xrange(i+1,self.Np):
                                                _ = F[i].real*F[j].real + F[i].imag*F[j].imag
                                                a[:,i,j] += _
                                                a[:,j,i] += _
                else:  #if given a psd
                        a=np.zeros((n_pix, n_freqs, self.Np, self.Np), float)  #initialize a 3-D array (frequencies x polarizations x polarizations)
			b=np.zeros((n_pix, n_freqs, self.Np, Nd), complex)
			n_pix_ones = np.ones((n_pix,),float)
                        for d_ind, detector in enumerate(sorted_detectors):
                                F = detector.antenna_patterns(theta, phi, psi, freqs=self.freqs) #tuple of numbers (pols), time shifts cancel within A so we supply no freqs
                                _psd = np.outer(n_pix_ones, detector.psd.interpolate(self.freqs) ) #2-D array: (n_pix,n_freqs)
                                for i in xrange(self.Np):
					a[:,:,i,i] += (F[i].real**2 + F[i].imag**2) / _psd
					b[:,:,i,d_ind] =  np.conjugate(F[i]) / _psd
                                        for j in xrange(i+1,self.Np):
						_ = (F[i].real*F[j].real + F[i].imag*F[j].imag) / _psd ### we take only the real part
                                                a[:,:,i,j] += _
						a[:,:,j,i] += _
                if n_pix == 1:
                        return a[0], b[0]
                else:
                        return a, b

	###
	def A_dpf(self, theta, phi, A=None, no_psd=False, byhand=False):
		"""computes A in the dominant polarization frame. If A is supplied, it converts A to the dominant polarization frame"""
		if A==None:
			A = self.A(theta, phi, 0.0, no_psd=no_psd)

		if no_psd:
			npix, npol, npol = np.shape(A)
			a = np.empty((npix, 1, npol, npol), float)
			a[:,0,:,:] = A
			A = a

		if byhand:
			a = A[:,:,0,0]
			b = A[:,:,0,1]
			c = A[:,:,1,1]
			dpfA = np.zeros_like(A, float)
			x = ((a-c)**2 + 4*b**2)**0.5
			y = a+c
			dpfA[:,:,0,0] = 0.5*(y + x)
			dpfA[:,:,1,1] = 0.5*(y - x)
		else:
			dpfA=np.zeros_like(A, float)
			vals = linalg.eigvals(A)[:,:,::-1] ### order by decreasing eigenvalue
			for i in xrange(self.Np):
				dpfA[:,:,i,i] = vals[:,:,i]

		if no_psd:
			return dpfA[:,0,:,:]
		else:
			return dpfA

	###
	def Aii_dpf(self, i, theta, phi, A=None, no_psd=False, byhand=False):
		"""computes a single component of A in the dominant polarization frame. If A is supplied, it converts to the dominant polarizatoin frame"""
		if A==None:
			A = self.A(theta, phi, 0.0, no_psd=no_psd)

		if no_psd:
			npix, npol, npol = np.shape(A)
			a = np.empty((npix,1,npol,npol),float)
			a[:,0,:,:] = A
			A = a
		if byhand:
			a = A[:,:,0,0]
                        b = A[:,:,0,1]
                        c = A[:,:,1,1]
                        dpfA = np.zeros_like(A, float)
                        x = ((a-c)**2 + 4*b**2)**0.5
                        y = a+c
			if no_psd:
				return 0.5*(y[:,0] + (-1)**i * x[:,0])
			else:
				return 0.5*(y + (-1)**i * x)
		else:
			vals = linalg.eigvals(A)[:,:,::-1] ### order by decreasing eigenvalue
			if no_psd:
				return vals[:,0,i]
			else:
				return vals[:,:,i]

        ###
        def AB_dpf(self, theta, phi, AB=None, no_psd=False, byhand=False):
                """ computes A and B in the dominant polarization frame """
                if AB==None:
			A, B = self.AB(theta, phi, psi=0.0, no_psd=no_psd)
		else:
			A, B = AB

		if no_psd:
			npix, npol, npol = np.shape(A)
			a = np.empty((npix, 1, npol, npol), float)
			a[:,0,:,:] = A
			A = a

			npix, pol, nifo = np.shape(B)
			b = np.empty((npix, 1, npol, nifo), complex)
			b[:,0,:,:] = B
			B = b

		if byhand:
			a = A[:,:,0,0]
			b = A[:,:,0,1]
			c = A[:,:,1,1]
			
			x = ((a-c)**2 + 4*b**2)**0.5
                        y = a+c
			z = c-a
			s0 = 0.5*(y + x)
			s1 = 0.5*(y - x)

			y0 = (1 + 4*b**2/(z+x)**2)**-0.5
			x0 = 2*y0*b/(z+x)

			y1 = (1 + 4*b**2/(z-x)**2)**-0.5
			x1 = 2*y1*b/(z-x)

			dpfA = np.zeros_like(A, float)
			dpfA[:,:,0,0] = s0
                        dpfA[:,:,1,1] = s1

			dpfB = np.zeros_like(B, complex)
			dpfB[:,:,0,:] = B[:,:,0,:]*x0 + B[:,:,1,:]*y0
			dpfB[:,:,1,:] = B[:,:,0,:]*x1 + B[:,:,1,:]*y1
			
			vecs = np.empty_like(A, float)
			vecs[:,:,0,0] = x0
			vecs[:,:,0,1] = x1
			vecs[:,:,1,0] = y0
			vecs[:,:,1,1] = y1

			if no_psd:
				return dpfA[:,0,:,:], dpfB[:,0,:,:], vecs[:,0,:,:]
			else:
				return dpfA, dpfB, vecs
		else:
			dpfA=np.zeros_like(A, float)
			dpfB=np.zeros_like(B, complex)

                        vals, vecs = linalg.eig(A)
			vals = vals[:,:,::-1] ### order by decreasing eigenvalue
			vecs = vecs[:,:,:,::-1] ### change order as well

			n_ifo = np.shape(B)[-1]
                        for i in xrange(self.Np):
                                dpfA[:,:,i,i] = vals[:,:,i]
				for j in xrange(n_ifo):
					dpfB[:,:,i,j] = np.sum( np.conjugate(vecs[:,:,:,i])*B[:,:,:,j], axis=-1 )

			if no_psd:
				return dpfA[:,0,:,:], dpfB[:,0,:,:], vecs[:,0,:,:]
			else:
	                        return dpfA, dpfB, vecs

	###
	def B_dpf(self, theta, phi, AB=None, no_psd=False, byhand=False):
		"""computes B in the dominant polarization frame. If A_B=(A,B) is supplied, we use it to define the dominant polarization frame transformation"""
		if AB==None:
			A, B = self.AB(theta, phi, psi=0.0, no_psd=no_psd)
		else:
			A, B = AB

                if no_psd:
                        npix, npol, npol = np.shape(A)
                        a = np.empty((npix, 1, npol, npol), float)
                        a[:,0,:,:] = A
                        A = a

                        npix, pol, nifo = np.shape(B)
                        b = np.empty((npix, 1, npol, nifo), complex)
                        b[:,0,:,:] = B
                        B = b

		if byhand:
                        a = A[:,:,0,0]
                        b = A[:,:,0,1]
                        c = A[:,:,1,1]

                        x = ((a-c)**2 + 4*b**2)**0.5
                        y = a+c
                        z = c-a
                        s0 = 0.5*(y + x)
                        s1 = 0.5*(y - x)

                        y0 = (1 + 4*b**2/(z+x)**2)**-0.5
                        x0 = 2*y0*b/(z+x)

                        y1 = (1 + 4*b**2/(z-x)**2)**-0.5
                        x1 = 2*y1*b/(z-x)

                        dpfB = np.zeros_like(B, complex)
                        dpfB[:,:,0,:] = B[:,:,0,:]*x0 + B[:,:,1,:]*y0
                        dpfB[:,:,1,:] = B[:,:,0,:]*x1 + B[:,:,1,:]*y1

                        vecs = np.empty_like(A, float)
                        vecs[:,:,0,0] = x0
                        vecs[:,:,0,1] = x1
                        vecs[:,:,1,0] = y0
                        vecs[:,:,1,1] = y1

			if no_psd:
				return dpfB[:,0,:,:], vecs[:,0,:,:]
			else:
	                        return dpfB, vecs

		else:
                        dpfB=np.zeros_like(B, complex)

                        vals, vecs = linalg.eig(A)
                        vecs = vecs[:,:,:,::-1] ### change order to decreasing eigval

                        n_ifo = np.shape(B)[-1]
                        for i in xrange(self.Np):
                                for j in xrange(n_ifo):
                                        dpfB[:,:,i,j] = np.sum( np.conjugate(vecs[:,:,:,i])*B[:,:,:,j], axis=-1)

			if no_psd:
				return dpfB[:,0,:,:], vecs[:,0,:,:]
			else:
	                        return dpfB, vecs

	###
	def Bni_dpf(self, name, i, theta, phi, AB=None, no_psd=False, byhand=False):
		"""computes a single component of B in the dominant polarization frame. If A_B=(A,B) is supplied, we use it to define the dominant polarization frame"""
		if AB==None:
			A, B = self.AB(theta, phi, psi=0.0, no_psd=no_psd)
		else:
			A, B = AB

                if no_psd:
                        npix, npol, npol = np.shape(A)
                        a = np.empty((npix, 1, npol, npol), float)
                        a[:,0,:,:] = A
                        A = a

                        npix, pol, nifo = np.shape(B)
                        b = np.empty((npix, 1, npol, nifo), complex)
                        b[:,0,:,:] = B
                        B = b

		det_ind = dict((n,ind) for ind,n in enumerate(self.detector_name_list()))[name]

                if byhand:
                        a = A[:,:,0,0]
                        b = A[:,:,0,1]
                        c = A[:,:,1,1]

                        x = ((a-c)**2 + 4*b**2)**0.5
                        y = a+c
                        z = c-a
                        s0 = 0.5*(y + x)
                        s1 = 0.5*(y - x)

			if i==0:
	                        y = (1 + 4*b**2/(z+x)**2)**-0.5
        	                x = y0*b/(z+x)
			else:
	                        y = (1 + 4*b**2/(z-x)**2)**-0.5
        	                x = y1*b/(z-x)

			if no_psd:
				return B[:,0,0,det_ind]* + B[:,0,1,det_ind]*y
			else:
	                        return B[:,:,0,det_ind]*x + B[:,:,1,det_ind]*y
                else:
			if no_psd:
				return self.B_dpf(theta, phi, AB=(A,B), no_psd=no_psd, byhand=byhand)[:,i,det_ind]
			else:
				return self.B_dpf(theta, phi, AB=(A,B), no_psd=no_psd, byhand=byhand)[:,:,i,det_ind]









	#####################################################################################
	### not sure the following functionality will every be used... consider removing it?
	######################################################################################

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

#=================================================
#
# physical constants
#
#=================================================
c = 299792458.0 #m/s

