### written by R.Essick (ressick@mit.edu)

usage = """ a general utilities module for sky localization. All distances are measured in seconds """

import numpy as np
from numpy import linalg

#=================================================
#
#            general utilities
#
#=================================================
deg2rad = np.pi/180
rad2deg = 1.0/deg2rad

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
	Fp = 0.0 # without freqs, these are scalars
	Fx = 0.0
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
	if freqs:
		if dr:
			dx, dy, dz = dr
			dt = dx*sin_theta*cos_phi + dy*sin_theta*sin_phi + dz*cos_theta
		phs = np.exp(-1j*2*np.pi*freqs*dt)

		if isinstance(Fp, np.float): # this avoids weird indexing with outer and a single point
			Fp *= phs
			Fx *= phs
		else:
			Fp = np.outer(Fp, phs)
			Fx = np.outer(Fx, phs)

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
	freqs = np.array([])
	psd = np.array([])
	interp = None

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
		from scipy.interpolate import interp1d
		self.freqs = freqs
		self.psd = psd
		self.interp = interp1d(freqs, psd, kind=kind, copy=False)

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
		return interp(freqs)

#=================================================
#
#              detector class
#
#=================================================
class Detector(object):
	"""
	an object representing a gravitational wave detector. methods are meant to be convenient wrappers for more general operations. 
	"""
	name = None  # detector's name (eg: H1)
	dr = np.zeros((3,)) # r_detector - r_geocent
	nx = np.zeros((3,)) # direction of the x-arm
	ny = np.zeros((3,)) # direction of the y-arm
	psd = None   # the psd for network (should be power, not amplitude)

	###
	def __init__(self, name, dr, nx, ny, psd):
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
		return self.psd.get_psd()

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

#=================================================
#
#              network class
#
#=================================================
class Network(object):
	"""
	an object representing a network of gravitational wave detectors.
	"""
	detectors = dict()
	freqs = None
	Np = 2 # support only GR right now

	###
	def __init__(self, detectors=[]):
		for detector in detectors:
			self.__add_detector(detector)

	###
	def __len__(self):
		"""returns the number of detectors in the network"""
		return len(self.detectors)

	###
	def __str__(self):
		s = "detector network containing:"
		for detector in sorted(self.detectors.keys()):
			s += "\n\t%s"%detector
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
		if not self.freqs:
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

	###
	def contains_name(self, name):
		"""checks to see if name is associated with any detector in the network"""
		return self.detectors.has_key(name)

	###
	def A(self, theta, phi, psi, no_psd=False):
		"""computes the entire matrix"""
		a=np.zeros((self.Np,self.Np))
		if no_psd:
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) # time shifts cancel within A so we supply no freqs
				for i in xrange(self.Np):
					a[i,i] += np.abs(F[i])**2
					for j in xrange(i+1,self.Np):
						_ = np.conjugate(F[i])*F[j]
						a[i,j] += _
						a[j,i] += np.conjugate(_)
		else:
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None)
				_psd = detector.get_psd().interpolate(self.freqs)
				for i in xrange(self.Np):
					a[i,i] += np.abs(F[i])**2/_psd
					for j in xrange(i+1,self.Np):
						_ = np.conjugate(F[i])*F[j]/_psd
						a[i,j] += _
						a[j,k] += np.conjugate(_)
		return a

	###
	def Aij(self, i, j, theta, phi, psi, no_psd=False):
		"""computes a single component of the matrix"""
		aij = 0.0
		if no_psd:
			for detector in self.detectors.values():
				F = detector.antenna_patterns(theta, phi, psi, freqs=None) # time shifts cancel within A so we supply no freqs
				aij += np.conjugate(F[i])*F[j]
		else:
			for detector in self.detectors.values():
                                F = detector.antenna_patterns(theta, phi, psi, freqs=None)
				_psd = detector.get_psd().interpolate(self.freqs)
				aij += np.conjugate(F[i])*F[j]/_psd
		return aij

	###
	def B(self, theta, phi, psi, no_psd=False):
		"""computes the entire matrix"""
		sorted_detectors = self.detectors_list()
		Nd = len(sorted_detectors)
		B = np.zeros((Nd, 2))
		for ind, detector in enumerate(sorted_detectors):
			F = detector.antenna_patterns(theta, phi, psi, freqs=freqs)
			if no_psd:
				B[ind,:] = F
			else:
				B[ind,:] = F/detector.get_psd().interplate(self.freqs)
		return B

	###
	def Bni(self, name, i, theta, phi, psi, no_psd=False):
		"""computes a single component of the matrix"""
		if not self.contains_name(name):
			raise KeyError, "detector=%s not contained in this network"%name
		detector = self.detectors[name]
		F = detector.antenna_patterns(theta, phi, psi, freqs=freqs)
		if no_psd:
			return F[i]
		else:
			return F[i]/detector.get_psd().interpolate(self.freqs)

	###
	def A_dpf(self, theta, phi, A=None, no_psd=False):
		"""computes A in the dominant polarization frame. If A is supplied, it converts A to the dominant polarization frame"""
		if not A:
			A = self.A(theta, phi, 0.0, no_psd=no_psd)
		return np.diag(self.eigvals(A))

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
	def rank(self, A, tol=1e-10):
		"""wrapper for numpy.linalg.matrix_rank that computes the rank of A. Rank is defined as the number of eigenvalues larger than tol"""
		return linalg.matrix_rank(A, tol=tol)

	###
	def eigvals(self, A):
		"""wrappter for numpy.linalg.eigvals that computes the eigenvalues of A"""
		vals = linalg.eigvals(A)
		vals.sort() # sort the eigenvalues
		return np.diag(vals[::-1]) # return in order of decreasing eigenvalue
	
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

#=================================================
#
# DEFINE KNOWN DETECTORS
#
#=================================================
default_psd = PSD(np.array([0]), np.array([1]), kind="linear")

### Detector locations and orientations taken from Anderson, et all PhysRevD 63(04) 2003

__H_dr__ = np.array((-2.161415, -3.834695, +4.600350))*1e6/c # sec
__H_nx__ = np.array((-0.2239, +0.7998, +0.5569))
__H_ny__ = np.array((-0.9140, +0.0261, -0.4049))
#LHO = Detector("LHO", __H_dr__, __H_nx__, __H_ny__, default_psd)
LHO = Detector("H1", __H_dr__, __H_nx__, __H_ny__, default_psd)

__L_dr__ = np.array((-0.074276, -5.496284, +3.224257))*1e6/c # sec
__L_nx__ = np.array((-0.9546, -0.1416, -0.2622))
__L_ny__ = np.array((+0.2977, -0.4879, -0.8205))
#LLO = Detector("LLO", __L_dr__, __L_nx__, __L_ny__, default_psd)
LLO = Detector("L1", __L_dr__, __L_nx__, __L_ny__, default_psd)

__V_dr__ = np.array((+4.546374, +0.842990, +4.378577))*1e6/c # sec
__V_nx__ = np.array((-0.7005, +0.2085, +0.6826))
__V_ny__ = np.array((-0.0538, -0.9691, +0.2408))
#Virgo = Detector("Virgo", __V_dr__, __V_nx__, __V_ny__, default_psd)
Virgo = Detector("V1", __V_dr__, __V_nx__, __V_ny__, default_psd)



