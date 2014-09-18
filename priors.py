usage="""a module storing the methods to build priors.
				print p[i]
This includes:
	priors on strain decomposed into gaussian terms used in the analytic marginalization over all possible signals
	priors on angular position, such as the galactic plane
"""

print """WARNING:
	helpstrings are not necessarily accurate. UPDATE THESE

	implement a few other priors on h?
		Jeffrey's prior?
		truncated pareto amplitudes
"""

#=================================================

import utils
np = utils.np
linalg = utils.linalg
hp = utils.hp

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
	
	###
	def __init__(self, freqs=None, means=None, covariance=None, amplitudes=None, n_freqs=1, n_gaus=1, n_pol=2):
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
				n_gaus, n_pol, n_freqs
			otherwise these are ignored
		"""
		
		### set up placeholders that will be filled
		self.n_freqs = None
		self.n_pol = None
		self.n_gaus = None

		self.freqs = None
		self.means = None
		self.covariance = None
		self.amplitudes = None

		### set data
		if freqs != None:
			self.set_freqs(freqs)
		if means != None:
			self.set_means(means, n_freqs=n_freqs, n_pol=n_pol, n_gaus=n_gaus)
		if covariance != None:
			self.set_covariance(covariance, n_freqs=n_freqs, n_pol=n_pol, n_gaus=n_gaus)
		if amplitudes != None:
			self.set_amplitudes(amplitudes, n_gaus=n_gaus)

	###
	def set_freqs(self, freqs):
		""" check and set freqs """
		if not isinstance(freqs, (np.ndarray)):
                        freqs = np.array(freqs)
                if len(np.shape(freqs)) != 1:
                        raise ValueError, "bad shape for freqs"

                n_freqs = len(freqs)
                if not n_freqs:
                        raise ValueError, "freqs must have at least 1 entry"
		if self.n_freqs and (n_freqs != self.n_freqs):
			raise ValueError, "inconsistent n_freqs"

		self.n_freqs = n_freqs
		self.freqs = freqs
		self.df = freqs[1]-freqs[0]                                                

	###
	def set_means(self, means, n_freqs=1, n_pol=2, n_gaus=1):
		""" check and set means. n_freqs, n_gaus, n_pol are only used if they are not already defined within the object """
		if isinstance(means, (int, float)): ### scalar means
			if self.n_freqs:
				n_freqs = self.n_freqs
			if self.n_gaus:
				n_gaus = self.n_gaus
			if self.n_pol:
				n_pol = self.n_pol
			self.means = means * np.ones((n_freqs, n_pol, n_gaus), complex)
			self.n_freqs = n_freqs
			self.n_gaus = n_gaus
			self.n_pol = n_pol
		else: ### vector means
			if not isinstance(means, np.ndarray):
                                means = np.array(means)
                        if len(np.shape(means)) != 3:
                                raise ValueError, "bad shape for means"
                        n_freqs, n_pol, n_gaus = np.shape(means)
                        if self.n_freqs and (n_freqs != self.n_freqs):
                                raise ValueError, "inconsistent n_freqs"
			if self.n_pol and (n_pol != self.n_pol):
				raise ValueError, "inconsistent n_pol"
                        elif n_pol <= 0:
                                raise ValueError, "must have a positive definite number of polarizations"
			if self.n_gaus and (n_gaus != self.n_gaus):
				raise ValueError, "inconsistent n_gaus"
			self.means = means
			self.n_gaus = n_gaus
			self.n_pol = n_pol
			self.n_freqs = n_freqs
	
	###
	def set_covariance(self, covariance, n_freqs=1, n_pol=2, n_gaus=1):
		""" check and set covariance. n_freqs, n_gaus, n_pol are only used if they are not already defined within the object """
		if isinstance(covariance, (int,float)): ### scalar covariances
			if self.n_freqs:
				n_freqs = self.n_freqs
                        if self.n_gaus:
                                n_gaus = self.n_gaus
                        if self.n_pol:
                                n_pol = self.n_pol
			self.covariance = np.zeros((n_freqs, n_pol, n_pol, n_gaus), complex)
			for i in xrange(n_pol):
				self.covariance[:,i,i,:] = covariance
			self.n_freqs = n_freqs
			self.n_gaus = n_gaus
			self.n_pol = n_pol
		else: ### vector covariances
			if not isinstance(covariance, np.ndarray):
                                covariance = np.array(covariance)
                        if len(np.shape(covariance)) != 4:
                                raise ValueError, "bad shape for covariance"
                        n_freqs, n_pol, n_p, n_gaus = np.shape(covariance)
                        if self.n_freqs and (n_freqs != self.n_freqs):
                                raise ValueError, "shape mismatch between freqs and covariance"
                        if n_pol != n_p:
                                raise ValueError, "inconsistent shape within covariance"
                        if self.n_pol and (n_pol != self.n_pol):
				raise ValueError, "inconsistent n_pol"
			if self.n_gaus and (n_gaus != self.n_gaus):
                                raise ValueError, "inconsistent n_gaus"
			self.covariance = covariance
			self.n_freqs = n_freqs
			self.n_gaus = n_gaus
			self.n_pol = n_pol

		### set up inverse-covariance and det_invcovariance
		self.invcovariance = np.zeros_like(covariance, dtype=complex)
		self.detinvcovariance = np.zeros((n_freqs, n_gaus), dtype=complex)
                for n in xrange(n_gaus):
			invc = linalg.inv(self.covariance[:,:,:,n])
                        self.invcovariance[:,:,:,n] = invc
			self.detinvcovariance[:,n] = linalg.det(invc)
			
	###
	def set_amplitudes(self, amplitudes, n_gaus=1):
		""" check and set amplitudes """
		if isinstance(amplitudes, (int, float)):
			if self.n_gaus:
				n_gaus = self.n_gaus
                        self.amplitudes = amplitudes * np.ones((n_gaus,), float)
			self.n_gaus = n_gaus
                else:
                        if not isinstance(amplitudes, np.ndarray):
                                amplitudes = np.array(amplitudes)
                        if len(np.shape(amplitudes)) != 1:
                                raise ValueError, "bad shape for amplitudes"
			n_gaus = len(amplitudes)
                        if self.n_gaus and (n_gaus != self.n_gaus):
                                raise ValueError, "inconsistent n_gaus"
			self.amplitudes = amplitudes
			self.n_gaus = n_gaus

	###
	def lognorm(self, freq_truth):
		"""
		computes the proper normalization for this prior assuming a model (freq_truth)
		return log(norm)

		WARNING: this factor will act as an overall scale on the posterior (constant for all pixels) and is only important for the evidence
			==> getting this wrong will produce the wrong evidence
		"""
		return np.log( np.sum( self.amplitudes ) ) ### assumes individual kernals are normalized.
#		#                                               det|Z|                        df**n_pol            sum over freqs      use amplitudes
#		return -utils.sum_logs(np.sum(np.log(self.detinvcovariance[freq_truth]) + self.n_pol*np.log(self.df), axis=0), coeffs=self.amplitudes)

	###
	def norm(self, freq_truth):
		"""
		computes the proper normalizatoin for this prior assuming a model (freq_truth)
		"""
		return np.exp(self.lognorm(freq_truth))

	###
	def __call__(self, h):
		"""
		evaluates the prior for the strain "h"
			this call sums h into h_rss and uses the univariate decomposition
			we expect this to hold for a marginalized distribution on the vector {h}

		returns prior

                We require:
			*h is a 2-D array
				np.shape(h) = (self.n_freqs, self.n_pol)
			if h is a 1-D array, we check to see if the shape matches either n_freqs or n_pol. 
				if it does, we broadcast it to the correct 2-D array
			if h is a scalar, we broadcast it to the correct 2-D array
		"""

		### make sure h has the expected shape
		if isinstance(h, (int, float)): ### h is a scalar
			h = h * np.ones((self.n_freqs, self.n_pol), float)
		elif not isinstance(h, np.ndarray):
			h = np.array(h)

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
#		p = np.empty((self.n_gaus),float)
		for n in xrange(self.n_gaus): ### sum over all gaussian terms
			d = h - self.means[:,:,n] ### difference from mean values
			dc = np.conj(d)
			m = self.invcovariance[:,:,:,n] ### covariance matricies

			### compute exponential term			
			e = np.zeros_like(self.freqs, float)
			for i in xrange(self.n_pol): ### sum over all polarizations
				for j in xrange(self.n_pol):
					e -= np.real(dc[:,i] * m[:,i,j] * d[:,j]) ### we expect this to be a real number, so we cast it to reals

			### normalization for this gaussian term
#			norm = np.mean( np.log( np.real(linalg.det(m)) ) ) - self.n_pol*np.log( np.pi )	### taking the mean here implies that our normalization convention is not consistent!!!
#			norm = np.log( np.real(linalg.det(m)) ) - self.n_pol*np.log( np.pi ) ### this corresponds to our normalization conventions
#			norm = np.zeros((self.n_freqs,),float)

			### insert into prior array
#			p[n] = np.log(self.amplitudes[n]) + np.sum(e) #+ np.sum(norm)
			p += self.amplitudes[n] * np.exp( np.sum(e)*self.df )    ### NEW NORMALIZATION SCHEME CONSISTENT WITH CURRENT pareto_amps
			                                                 ### RETURNS THE SAME RESULT INDEPENDENT OF THE NUMBER OF BINS!

		return p
#		return utils.sum_logs(p) ### return log(prior)
#		return max( np.exp(utils.sum_logs(p)), 0.0)

        ###
        def plot(self, figname, xmin=1, xmax=10, npts=1001, ymin=None, ymax=None, grid=False):
                """
                generate a plot of the prior and save it to figname
                """
                ### generate plot
                fig_ind = 0
                fig = plt.figure(fig_ind)
		ax = plt.subplot(1,1,1)

		x = np.logspace(np.log10(xmin),np.log10(xmax),npts)/self.df

#		logp = np.array([self(X/(self.n_freqs*self.n_pol)**0.5) for X in x])
#		logp -= utils.sum_logs(logp)
#
#		p = np.exp(logp)
		p = np.array([self(X/(self.n_freqs*self.n_pol)**0.5) for X in x])

		ax.loglog(x, p )
#		ax.plot(x, logp/np.log(10.0))
#		ax.set_xscale('log')

		ax.set_xlabel("$\log_{10}(h_{rss})$")
		ax.set_ylabel("$p(h)$")
		
		ax.grid(grid, which="both")

		ax.set_xlim(xmin=xmin, xmax=xmax)
		if ymin:
			ax.set_ylim(ymin=ymin)
		if ymax:
			ax.set_ylim(ymax=ymax)

		fig.savefig(figname)
		plt.close(fig)

	###
	def __repr__(self):
		return self.__str__()

	###
	def __str__(self):
		s = """priors.hPrior object
	min{freqs}=%.5f
        max{freqs}=%.5f
        No. freqs =%d
	No. polarizations =%d
	No. gaussians =%d"""%(np.min(self.freqs), np.max(self.freqs), self.n_freqs, self.n_pol, self.n_gaus)
		return s

#=================================================
# prior on sky location
#=================================================
class angPrior(object):
	"""
        An object representing the prior on sky location.  
	this prior is stored in terms of the standard polar coordinates
		*theta : polar angle
		*phi   : azimutha angle
	both theta and phi are measured in radians, and refer to standard Earth-Fixed coordinates
	"""
	
	known_prior_type = ["uniform", "galactic_plane", "antenna_pattern"]

	# right now the default option is uniform over the sky
	# will want to add some sort of beam pattern option
	# eventually should add galaxy catalogues option
	
	###
	def __init__(self, nside_exp=7, prior_type='uniform', coord_sys="E", **kwargs):
		"""
		Initializes a prior with HEALPix decomposition defined by
			nside = 2**nside_exp

		prior_type defines the prior used:
			uniform [DEFAULT]
				constant independent of theta,phi
			galactic_plane
				a gaussian in galactic latitude
			antenna_pattern
				the maximum eigenvalue of the sensitivity matrix
	
		kwargs can be:
			gmst : float [GreenwichMeanSidereelTime]
				used to convert between Earth-fixed and Galactic coordinates 
			network : instance of utils.Network object
				used to compute antenna_pattern prior
			frequency : float [Hz]
				used to compute eigenvalues of sensitivity matrix for antenna pattern prior
			exp : float
				if prior_type=="galactic_plane":
					width of gaussian in galactic lattitude (in radians) 
					DEFAULT = np.pi/8
				if prior_type=="antenna_pattern":
					p ~ (F+^2 + Fx^2)**(exp/2.0)
					DEFAULT = 3.0

		we may want to add:
			galaxy catalogs (this is why gmst is an optional argument)
		"""
		### delegate to set methods
		self.set_nside(nside_exp)
		self.set_prior_type(prior_type, **kwargs)
		self.set_theta_phi(coord_sys="E", **kwargs)

	###
	def set_nside(self, nside_exp):
		""" check and set nside """
		if not isinstance(nside_exp, int):
                        raise ValueError, "nside_exp must be an integer"
                nside = 2**nside_exp
                self.nside = nside
		self.npix = hp.nside2npix(nside)

	###
	def set_prior_type(self, prior_type, **kwargs):
		""" check and set prior type """
		# Initialize prior type
                if not (prior_type in self.known_prior_type):
                        raise ValueError, "Unknown prior_type=%s"%prior_type
                self.prior_type = prior_type

                # store information needed to calculate prior for specific types
                if prior_type == "galactic_plane":
                        if kwargs.has_key("gmst"):
                                self.gmst = kwargs["gmst"]
                        else:
                                raise ValueError, "must supply \"gmst\" with prior_type=\"galactic_plane\""
                        if kwargs.has_key("exp"):
                                self.exp = kwargs["exp"]
                        else:
                                self.exp = np.pi/8

                elif prior_type == "antenna_pattern":
                        if kwargs.has_key("network"):
				network = kwargs["network"]
				if not isinstance(network, utils.Network):
					raise ValueError, "network must be an instance of utils.Network"
                                self.network = network
                        else:
                                raise ValueError, "must supply \"network\" with prior_type=\"antenna_pattern\""
                        if kwargs.has_key("frequency"):
                                self.frequency = kwargs["frequency"]
                        else:
                                raise ValueError, "must supply \"frequency\" with prior_type=\"antenna_pattern\""
                        if kwargs.has_key("exp"):
                                self.exp = kwargs["exp"]
                        else:
                                self.exp = 3.0
	
	### 
	def set_theta_phi(self, coord_sys="E", **kwargs):
		"""
		compute and store theta, phi
                delegates to utils.set_theta_phi
		"""
		if not self.nside:
                        raise ValueError, "set_angPrior() first"
		self.coord_sys=coord_sys
                theta, phi = utils.set_theta_phi(self.nside, coord_sys=coord_sys, **kwargs)
                self.theta = theta
                self.phi = phi
	
        ###             
        def angprior(self, normalize=False):
                """
                builds the normalized prior over the entire sky.
		if normalize:
			we ensure the prior is normalized by directly computing the sum
                """
		if not self.nside:
			raise ValueError, "set_nside() first"
                #Pixelate the sky
                npix = hp.nside2npix(self.nside)  #number of pixels
		
		### an array for sky positions
		if self.theta == None:
			raise ValueError, "set_theta_phi() first"

		### compute prior for all points in the sky
		angprior = self(self.theta, self.phi)

		if normalize:
			angprior /= np.sum(angprior) ### ensure normalization

		return angprior

	###
	def __call__(self, theta, phi, degrees=False):
		"""
		evalute the prior at the point defined by 
			theta, phi
		"""
		if isinstance(theta, (int,float)):
			theta = np.array([theta])
		elif not isinstance(theta, np.ndarray):
			theta = np.array(theta)
		if isinstance(phi, (int,float)):
			phi = np.array([phi])
		if not isinstance(phi, np.ndarray):
			phi = np.array(phi)

		if len(phi) != len(theta):
			raise ValueError, "theta, phi must have the same length"

		if degrees: ### convert to radians
			theta *= np.pi/180
			phi *= np.pi/180

		### check theta, phi for sanity
		if np.any((theta<0.0)*(theta>np.pi)):
			raise ValueError, "theta must be between 0 and pi"
		if np.any((phi<0.0)*(phi>2*np.pi)):
			raise ValueError, "phi must be between 0 and 2*pi"

		### compute prior
		if self.prior_type == "uniform":
			return np.ones_like(theta)/hp.nside2npix(self.nside)

		elif self.prior_type == "galactic_plane": 
			### need to convert from Earth-fixed -> galactic an apply gaussian prior on galactice latitude
			### expect self.gmst, self.exp to exist
			
			
			###
			# need to define a way to convert from earth-fixed to galactic coordinate
			# maybe astropy is a good solution?
			###
			print "WARNING: prior_type=\"galactic_plane\" is not implemented. Defaulting to \"uniform\""
			return np.ones_like(theta)/hp.nside2npix(self.nside)

		elif self.prior_type == "antenna_pattern":
			### need to compute max eigenvalue of sensitivity matrix (at some nominal frequency) and raise it so a given power
			### expect self.network, self.exp, self.frequency to exist

			###
			# WARNING: this implementation is likely to be slow! It can probably be optimized
			###

			prior = np.empty((len(theta),),float)
			A = self.network.A(theta, phi, 0.0, no_psd=False)
			evals = np.max(np.linalg.eigvals(A), axis=2)
			for ipix in xrange(len(theta)):
				prior[ipix] = np.interp(self.frequency, self.network.freqs, evals[ipix])**(self.exp/2.0)

			return prior

		else:
			raise ValueError, "unknown prior_type=%s"%self.prior_type

	###
	def plot(self, figname, title=None, unit=None, inj=None, est=None, graticule=False):
		"""
		generate a plot of the prior and save it to figname
		if inj != None:
			(theta,phi) = inj
			plot marker at theta,phi
		"""
		### generate plot
		fig_ind = 0
		fig = plt.figure(fig_ind)
		hp.mollview(self.angprior(normalize=True)/hp.nside2pixarea(self.nside), title=title, unit=unit, flip="geo", fig=fig_ind)
		if graticule:
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
	def __repr__(self):
		return self.__str__()

	###
	def __str__(self):
		s = """priors.angPrior object
	nside = %d
	npix  = %d
	prior_type = %s"""%(self.nside, self.npix, self.prior_type)

		if self.prior_type == "galactic_plane":
			s += """

	exp = %.3f
	gmst = %.5f"""%(self.exp, self.gmst)

		elif self.prior_type == "antenna_pattern":
			s += """

	exp = %.3f
	frequency = %.5f

	network = %s"""%(self.exp, self.frequency, str(self.network))

		return s

#=================================================
#
# Polarization constraints
#
#=================================================
#=================================================
# methods for generic constraints
#=================================================
def isotropic_to_constrained(covariances, alpha, psi, theta, r=1e-10):
	"""
	converts an isotropic (in polarization space) covariances to one with polarization constraints defined by
		h1 = tan(alpha)*e**(i*psi) * h2 = P * h2

	for example,
		linearly polarized in known frame : 
			alpha = alpha_o
			psi   =  0.0
		elliptically polarized in known frame :
			alpha = alpha_o
			psi   = np.pi/2

	if we know there are such constrains, but do not know the correct frame, etc, we can numerically marginalize over (alpha, psi)

	we work in the orthogonal basis
		/ a \ =                 /         cos(alpha)    sin(alpha)*e**(i*psi) \ / h1 \ 
		|   | = (1+|P|)**-0.5 * |                                             | |    |
		\ b /                   \ -sin(alpha)*e**(-i*phi)     cos(alpha)      / \ h2 /

	and expect the inverse-covariance in this basis to be
		         /  1/v       0   \ 
		invcov =  |                |
		         \   0    1/(r*v) /

	where v is the isotropic covariance : cov[i,i] = v

	In the h1,h2 basis, this means our inverse-covariance becomes
		        /     cos(alpha)**2/v + sin(alpha)**2/(r*v)      sin(alpha)*cos(alpha)*e**(i*psi)*(1 - 1/r)/v  \ 
		invcov = |                                                                                              |
                        \ sin(alpha)*cos(alpha)*e**(-i*psi)*(1 - 1/r)/v      sin(alpha)**2/v + cos(alpha)**2/(r*v)     /

	we also apply the rotation matrix
		    / sin(theta)   cos(theta) \ 
		R = |                         | = transpose(R)
		    \ cos(theta)  -sin(theta) /

	to rotate h_1,h_2 into some other frame, which results in the final invcov matrix

		rotated_invcov = R * invcov * R 

	returns constrained_covariances
	"""

	### check covariances
	if not isinstance(covariances, np.ndarray):
		covariances = np.array(covariances)
	if len(np.shape(covariances)) != 4:
		raise ValueError, "bad shape for covariances"
	n_freqs, n_pol, n_p, n_gaus = np.shape(covariances)
	if n_pol != n_p:
		raise ValueError, "inconsistent shape within covariances"
	if n_pol != 2:
		raise ValueError, "We only support polarization constrains for n_pol=2"

	### construce constrained covariances
	constrained_covariances = np.zeros_like(covariances, complex)

	cosalpha = np.cos(alpha)
	sinalpha = np.sin(alpha)
	cospsi = np.cos(psi)
	sinpsi = np.sin(psi)
	costheta = np.cos(theta)
	sintheta = np.sin(theta)

	### iterate over all covariance matricies and convert
	for f in xrange(n_freqs):
		for g in xrange(n_gaus):
			cov = covariances[f,:,:,g]
			### check that cov is diagonal and isotropic
			if (cov[0,0] != cov[1,1]) or cov[0,1] or cov[1,0]:
				raise ValueError, "we only support conversion of diagonal, isotropic covariance matrices"
			v = cov[1,1] ### pull out variance

			### compute constrained inverse-covariance
			constrained_invcov = np.empty_like(cov)
			a = cosalpha**2/v + sinalpha**2/(r*v)
			b = sinalpha*cosalpha*(cospsi + 1.0j*sinpsi)*(1-1.0/r)/v
			c = sinalpha*cosalpha*(cospsi - 1.0j*sinpsi)*(1-1.0/r)/v
			d = sinalpha**2/v + cosalpha**2/(r*v)

			### apply rotation matrix
			constrained_invcov[0,0] = a*sintheta**2 + b*sintheta*costheta +c*sintheta*costheta + d*costheta**2
			constrained_invcov[0,1] = a*sintheta*costheta - b*sintheta**2 + c*costheta**2 - d*sintheta*costheta
			constrained_invcov[1,0] = a*sintheta*costheta + b*costheta**2 - c*sintheta**2 - d*sintheta*costheta
			constrained_invcov[1,1] = a*costheta**2 - b*sitheta*costheta - c*sintheta*costheta + d*sintheta**2

			### fill in constrained_covariance
			constrained_covariances[f,:,:,g] = linalg.inv(constrained_invcov)

	return constrained_covariances

###
def istoropic_to_margenalized(covariances, alpha, psi, theta, r=1e-10):
	"""
	expands covariances to numerically marginalize over (alpha,psi,theta)
		n_gaus -> n_gaus*len(alpha)*len(psi)*len(theta)
	constrained_covariances are computed and stored appropriately
	"""
        ### check covariances
        if not isinstance(covariances, np.ndarray):
                covariances = np.array(covariances)
        if len(np.shape(covariances)) != 4:
                raise ValueError, "bad shape for covariances"
        n_freqs, n_pol, n_p, n_g = np.shape(covariances)
        if n_pol != n_p:
                raise ValueError, "inconsistent shape within covariances"
        if n_pol != 2:
                raise ValueError, "We only support polarization constrains for n_pol=2"

	### check alpha
	if isinstance(alpha, (int,float)):
		alpha = np.array([alpha])
	elif not isinstance(alpha, np.ndarray):
		alpha = np.array(alpha)
	if len(np.shape(alpha)) != 1:
		raise ValueError, "bad shape for alpha"
	n_alpha = len(alpha)

	### check psi
	if isinstance(psi, (int,float)):
		psi = np.array([psi])
	elif not isinstance(psi, np.ndarray):
		psi = np.array(psi)
	if len(np.shape(psi)) != 1:
		raise ValueError, "bad shape for psi"
	n_psi = len(psi)

	### check theta
	if isinstance(theta, (int,float)):
		theta = np.array([theta])
	elif not isinstance(theta, np.ndarray):
		theta = np.array(theta)
	if len(np.shape(theta)) != 1:
		raise ValueError, "bad shape for theta"
	n_theta = len(theta)

	### new number of gaussians
	n_gaus = n_g*n_alpha*n_psi*n_theta

        ### construct constrained_covariances
        constrained_covariances = np.empty((n_freqs, n_pol, n_pol, n_gaus), complex)

        ### iterate, compute, and fill in constrained_covariances
	ind = 0
        for a in alpha:
		for p in phi:
			for t in theta:
		                constraind_covariances[:,:,:,ind*n_g:(ind+1)*n_g] = isotropic_to_constrained(covariances, a, p, t, r=r) ### fill in appropriately
				ind += 1

	return constrained_covariances

###
def marginalized_amplitudes(amplitudes, alpha, psi, theta):
	"""
	returns amplitudes appropriately broadcast for isotropic_to_marginalized()
	"""
	### check amplitudes
	if not isinstance(amplitudes, np.ndarray):
		amplitudes = np.array(amplitudes)
	if len(np.shape(amplitudes)) != 1:
		raise ValueError, "bad shape for amplitudes"
	n_g = len(amplitudes)

       	### check alpha
        if isinstance(alpha, (int,float)):
                alpha = np.array([alpha])
        elif not isinstance(alpha, np.ndarray):
                alpha = np.array(alpha)
        if len(np.shape(alpha)) != 1:
                raise ValueError, "bad shape for alpha"
        n_alpha = len(alpha)

        ### check psi
        if isinstance(psi, (int,float)):
                psi = np.array([psi])
        elif not isinstance(psi, np.ndarray):
                psi = np.array(psi)
        if len(np.shape(psi)) != 1:
                raise ValueError, "bad shape for psi"
        n_psi = len(psi)
	
        ### check theta
        if isinstance(theta, (int,float)):
                theta = np.array([theta])
        elif not isinstance(theta, np.ndarray):
                theta = np.array(theta)
        if len(np.shape(theta)) != 1:
                raise ValueError, "bad shape for theta"
        n_theta = len(theta)

	### define new amplitudes
	marginalized_amplitudes = np.flatten( np.outer( amplitudes, np.ones((n_alpha*n_psi*n_theta),float)/(n_alpha*n_psi*n_theta) ) )

	return marginalized_amplitudes

###
def marginalizd_means(means, alpha, psi, theta):
	"""
	returns means appropriately broadcast for isotropic_to_marginalized()
	"""
	### check means
	if not isinstance(means, np.ndarray):
		means = np.array(means)
	if len(np.shape(means)) != 3:
		raise ValueError, "bad shape for means"
	n_freqs, n_pol, n_g = np.shape(means)
	if n_pol != 2:
		raise ValueError, "We only support polarization constrains for n_pol=2"

        ### check alpha
        if isinstance(alpha, (int,float)):
                alpha = np.array([alpha])
        elif not isinstance(alpha, np.ndarray):
                alpha = np.array(alpha)
        if len(np.shape(alpha)) != 1:
                raise ValueError, "bad shape for alpha"
        n_alpha = len(alpha)

        ### check psi
        if isinstance(psi, (int,float)):
                psi = np.array([psi])
        elif not isinstance(psi, np.ndarray):
                psi = np.array(psi)
        if len(np.shape(psi)) != 1:
                raise ValueError, "bad shape for psi"
        n_psi = len(psi)

        ### check theta
        if isinstance(theta, (int,float)):
                theta = np.array([theta])
        elif not isinstance(theta, np.ndarray):
                theta = np.array(theta)
        if len(np.shape(theta)) != 1:
                raise ValueError, "bad shape for theta"
        n_theta = len(theta)

	### new number of gaussians
	n_gaus = n_g*n_alpha*_n_psi*n_theta

	### define new means
	marginalized_means = np.empty((n_freqs,n_pol,n_gaus),complex)

	### iterate and fill in
	ind = 0
	for a in alpha:
		for p in phi:
			for t in theta:
				marginalized_means[:,:,ind*n_g:(ind+1)*n_g] = means
				ind += 1

	return marginalized_means

#=================================================
# methods for specific constraints
#=================================================

###
def isotropic_to_circular(means, covariances, amplitudes, r=1e-10, n_theta_marge=60):
	"""
	converts an isotropic covariances to circularly polarized covariances
	delegates to isotropic_to_constrained() with 
		alpha = np.pi/4
		psi   = np.pi/2

	delegates to marginalized_means(), isotropic_to_constrained(), marginalized_amplitudes()

	returns means, covariances, amplitudes
	"""
	theta = np.arange(n_theta_marge)*2*np.pi/n_theta_marge
	return marginalized_means(means, np.pi/4, np.pi/2, theta), isotropic_to_constrained(covariances, np.pi/4, np.pi/2, theta, r=r), marginalized_amplitudes(amplitudes, np.pi/4, np.pi/2, theta)

###
def isotropic_to_elliptical(means, covariances, amplitudes, r=1e-10, n_alpha_marge=60, n_theta_marge=60):
	"""
	converts an isotropic covariances to elliptical covariances
	numerically marginalizes over alpha with n_marge samples
		n_gaus -> n_gaus*n_marge

	delegates to marginalized_means(), isotropic_to_marginalized(), marginalized_amplitudes()

	returns means, covariances, amplitudes
	"""
	theta = np.arange(n_theta_marge)*2*np.pi/n_theta_marge
	alpha = np.arange(n_alpha_marge)*2*np.pi/n_alpha_marge
	return marginalized_means(means, alpha, np.pi/2, theta), isotropic_to_marginalized(covariances, alpha, np.pi/2, theta, r=r), marginalized_amplitudes(amplitudes, alpha, np.pi/2, theta)

###
def isotropic_to_linear(means, covariances, amplitudes, r=1e-10, n_alpha_marge=60, n_theta_marge=60):
	"""
	converts an isotropic covariances to linear covariances
	numerically marginalizes over alpha with n_marge samples
		n_gaus -> n_gaus*n_marge

	delegates to marginalized_means(), isotropic_to_marginalized(), marginalized_amplitudes()

	returns means, covariances, amplitudes
	"""
	theta = np.arange(n_theta_marge)*2*np.pi/n_theta_marge
	alpha = np.arange(n_alpha_marge)*2*np.pi/n_alpha_marge
	return marginalized_means(means, alpha, 0.0, theta), isotropic_to_marginalized(covariances, alpha, 0.0, theta, r=r), marginalized_amplitudes(amplitudes, alpha, 0.0, theta)

#=================================================
#
# Methods to compute standard priors
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
def pareto(a, n_freqs, n_pol, variances):
	"""
	computes the required input for a Prior object using the pareto distribution
		p(h_rss) = h_rss**-a

	delegates decomposition into gaussians to pareto_amplitudes

	returns amplitudes, means, covariance
	"""
        if not isinstance(variances, np.ndarray):
                variances = np.array(variances)
        if len(np.shape(variances)) != 1:
                raise ValueError, "bad shape for variances"
        n_gaus = len(variances)

	### compute amplitudes
	amplitudes = pareto_amplitudes(a, variances, n_pol=n_pol)

	### compute covariance in correct array format
	covariances = np.zeros((n_freqs,n_pol,n_pol,n_gaus),float)
	for i in xrange(n_pol):
		covariances[:,i,i,:] = 1
	for n in xrange(n_gaus):
		covariances[:,:,:,n] *= 2*variances[n]
	
	### instantiate means in correct array format
	means = np.zeros((n_freqs, n_pol, n_gaus), float)

	return means, covariances, amplitudes

###
def pareto_amplitudes(a, variances, n_pol=1):
	"""
	computes the amplitudes corresponding to the supplied variances to optimally reconstruct a pareto distribution with exponent "a"
	p(x) = x**-a ~ \sum_n C_n * exp( -x**2/(2*variances[n]) )

	We require:
		*variances is a 1-D array
			np.shape(variances) = (N,)

	returns C_n as a 1-D array


	amplitudes are defined (up to an arbitrary constant) through a chi2-minimization

		chi2 = \int dx [ (f - fhat) / f ]**2
	
	where 
		f = x**-a
		fhat = \sum C[n] exp( -x**2/(2*variances[n]) )

	a minimization with respect to C_n yields

		C[n] = np.inv(M)[n,n] * (2*np.pi*variances[n])**-0.5

	where

		M[m,n] = \int dx [ (2*np.pi*variances[m])**-0.5 * (2*np.pi*variances[n])**-0.5 * exp( -(x**2/2) * (variances[m]**-1 + variances[n]**-1) ) / f ]
		       \propto (variances[m]*variances[n])**-0.5 * (variances[m]**-1 + variances[n]**-1)**(-0.5*(a+1))

	where the proportionality constant is independent of m,n

	we return a list of coefficients normalized so that

		\int dx [ \sum C[n] * exp( -x**2/(2*variances[n]) ) ] = 1
	"""
	if not isinstance(variances, np.ndarray):
                variances = np.array(variances)
        if len(np.shape(variances)) != 1:
                raise ValueError, "bad shape for variances"
        n_gaus = len(variances)


	### hopefully this is the correct form
	"""
	we attempt to fit a univariate distribution (in h_rss) using a chi2 minimization

	chi2 = int dx ( ( x**(-a) - sum C_n K_n ) / (x**-a) )**2

	where K_n = (2*pi*v_n)**-0.5 * exp( - x**2 / 2*v_n )

	=> int x**a Km = sum C_n int x**(2a) K_n K_m
           v_m**(a/2) I(a) = sum C_n v_mn**((1+2a)/2) (v_m*v_n)**(-1) Y(a)    where I(a) and Y(a) are non-dimensional integrals
                                                                              v_mn = v_m*v_m / (v_m + v_n)
           The C_n are determined through straightforward linear algebra
	"""

	### Distribution for normalized univariate kernals:
	### we may want to handle the smll numbers more carefully, because we start to run into problems with float precision (setting things to zero).
	###   we can do something like utils.sum_logs where we subtract out the maximum value, and then do the manipulation, only to put in the maximum value at the end.
	###   for right now, this appears to work well enough.

	### build matrix from RHS
        M = np.empty((n_gaus, n_gaus), float)
        for m in xrange(n_gaus):
                v_m = variances[m]
                M[m,m] = 2**-(0.5+a) * v_m**(a-0.5)
                for n in xrange(m+1, n_gaus):
                        v_n = variances[n]
                        M[n,m] = M[m,n] = (v_m+v_n)**-(0.5+a) * (v_m*v_n)**a

        ### invert matrix from RHS
        invM = linalg.inv(M)

        ### compute coefficients
        vec = variances**(0.5*a)
        C_n = np.sum( linalg.inv(M)*(variances**(0.5*(1+a))), axis=1) ### take the inverse and matrix product

        ### normalize coefficients?
        C_n /= np.sum(C_n)

        return C_n


	'''
	Decomposition for un-normalized univariate kernals

	### build matrix from RHS
	M = np.empty((n_gaus, n_gaus), float)
	for m in xrange(n_gaus):
		v_m = variances[m]
		M[m,m] = (0.5*v_m)**(0.5+a)
		for n in xrange(m+1, n_gaus):
			v_n = variances[n]
			v_mn = v_m*v_n / (v_m + v_n)
			M[n,m] = M[m,n] = v_mn**(0.5+a)

	### invert matrix from RHS
	invM = linalg.inv(M)

	### compute coefficients
	vec = variances**(0.5*(1+a))
	C_n = np.sum( linalg.inv(M)*(variances**(0.5*(1+a))), axis=1) ### take the inverse and matrix product

	### normalize coefficients?
	C_n /= np.max(C_n)

	return C_n
	'''

	'''

	OLD PROCEEDURE THAT IS BELIEVED TO BE FAULTY


	"""
	a -= 1 ### Not sure where this factor comes from...but it makes p(h) scale correctly...

	### construct the matrix M
	M = np.empty((n_gaus,n_gaus),float) ### matrix representing the inner product in our gaussian basis
	for i in xrange(n_gaus):
		vi = variances[i]
		M[i,i] = 2**(-0.5*(a+1)) * vi**(0.5*(a-1))
		for j in xrange(i+1,n_gaus):
			vj = variances[j]
			M[i,j] = M[j,i] = (vi*vj)**-0.5 * (vi**-1 + vj**-1)**(-0.5*(a+1))
	"""
#	a *= 2
	M = np.empty((n_gaus, n_gaus), float)
	for i in xrange(n_gaus):
		vi = variances[i]
		M[i,i] = 2**-(n_pol+0.5*a) * vi**-(n_pol-0.5*a)
		for j in xrange(i+1,n_gaus):
			vj = variances[j]
			M[i,j] = M[j,i] = (vi+vj)**-(n_pol+0.5*a) * (vi*vj)**(0.5*a)




	### invert and extract diagonal, multiply by normalization factor
#	C = np.diagonal(linalg.inv(M))
	C = np.sum(linalg.inv(M), axis=0) ### sum over each row

	### return coefficients that will produce a normalized 1-D prior
	### we first multiply by the normalization factor to get correct scaling
#	return C * (2*np.pi*variances)**-0.5 / np.sum(C) # = C * (2*np.pi*variances)**-0.5 / np.sum( C * (2*np.pi*variances)**-0.5 * (2*np.pi*variances)**0.5 )

	### if gaussian kernels are L1 normalized, then we shouldn't include that normalization in the definition of our amplitudes
	### this makes things simpler for functional integration. We normalize each kernel separately, so we don't have to worry about any of those factors post facto
	return C / np.sum(C) # = C * (2*np.pi*variances)**-0.5 / np.sum( C * (2*np.pi*variances)**-0.5 * (2*np.pi*variances)**0.5 )
	'''
