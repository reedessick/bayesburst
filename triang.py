### written by R.Essick (ressick@mit.edu)

usage = """triang.py is a module that contains basic methods for triangulation, including a TimingNetwork that does most of the heavy lifting"""

import utils
np = utils.np
import pdf_estimation as pdfe

import healpy as hp
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#===================================================================================================
#
#                                                utilities
#
#===================================================================================================
#========================
# pixelization reference
#========================
def __pixarray(npix, nside=None):
	"""
	constructs an array that maps ipix -> theta,phi for reference
	"""
	pixarray = np.empty((npix, 2), float)
	if not nside:
		nside = hp.npix2nside(npix)
	for ipix in xrange(npix):
		pixarray[ipix] = hp.pix2ang(nside, ipix)
	return pixarray

#========================
# likelihoods
#========================
def gaussian_likelihood(toa, tof_map, err):
	"""
	computes the likelihood that a signal came from the points in tof_map and produced toa, assuming gaussian errors defined in err_map
	require:	shape(toa)     = (N,)
			shape(tof_map) = (Npix,1+N)
			shape(err_map) = (Npix,1+N)
	"""
	Npix,N = np.shape(tof_map)
	if (N,) != np.shape(err):
		raise ValueError, "tof_map and err_map must have the same shape!"
	elif N != len(toa):
		raise ValueError, "shape mismatch between toa and tof_map"

	return np.exp( -0.5*sum( np.transpose((toa-tof_map[:,:])/err)**2 ) ) # transposition allows us to sum over detector pairs

###
def kde_likelihood(toa, tof_map, kde):
	"""
	computes the likelihood that a signal came from the points in tof_map and produced toa, assuming the measured distribution in kde
	only allows a single distribution of errors, stored in kde
	require:	shape(toa)	= (N,)
			shape(tof_map)	= (Npix,1+N)
			kde to be callable (preferably some sort of interpolator) and can handle an array-like object as input
	"""
	Npix,N = np.shape(tof_map)
	if N != len(toa):
		raise ValueError, "shape mismatch between toa and tof_map"

	return kde(toa-tof_map[:,:])

#========================
# Likelihood (from timing)
#========================
class Likelihood(object):
	"""
	a callable object that computes the likelihood from timing
	stores 
		binnedTimingNetwork, which it references for error distributions
		npix, pixarray which define the specific pixelization used (HEALPix decomposition)
		tof_map which stores the expected tof for each binned distribution from binnedTimingNetwork for each pixel in the sky
	"""

	###
	def __init__(self, binnedTimingNetwork, npix, pixarray=None):
		self.binnedTimingNetwork = binnedTimingNetwork

		if binnedTimingNetwork.error_approx == "gaussian":
			self.likelihood_func = gaussian_likelihood
		elif binnedTimingNetwork.error_approx == "kde":
			self.likelihood_func = kde_likelihood
		else:
			raise ValueError, "error-approx=%s not understood"%binnedTimingNetwork.error_approx

		self.set_tof_map(npix, pixarray=pixarray)


	###
	def __call__(self, toa, snr, freq, bandwidth, verbose=False):
                """ computes the likelihood of observing toa """
		if self.binnedTimingNetwork.errors==None:
			raise ValueError, "self.errors==None. Use self.toacache_to_errors() to build self.errors before calling likelihood"

		snr_truth = (snr >= self.binnedTimingNetwork.snr_bin_edges[:-1])*(snr < self.binnedTimingNetwork.snr_bin_edges[1:])
		snr_bin_ind = np.arange(self.binnedTimingNetwork.n_snr_bin)[snr_truth]

		freq_truth = (freq >= self.binnedTimingNetwork.freq_bin_edges[:-1])*(freq < self.binnedTimingNetwork.freq_bin_edges[1:])
		freq_bin_ind = np.arange(self.binnedTimingNetwork.n_freq_bin)[freq_truth]

		bandwidth_truth = (bandwidth >= self.binnedTimingNetwork.bandwidth_bin_edges[:-1])*(bandwidth < self.binnedTimingNetwork.bandwidth_bin_edges[1:])
		bandwidth_bin_ind = np.arange(self.binnedTimingNetwork.n_bandwidth_bin)[bandwidth_truth]

		if verbose: 
			print "snr_bin  : %d/%d"%(snr_bin_ind, self.binnedTimingNetwork.n_snr_bin)
			print "freq_bin : %d/%d"%(freq_bin_ind, self.binnedTimingNetwork.n_freq_bin)
			print "bw_bin   : %d/%d"%(bandwidth_bin_ind, self.binnedTimingNetwork.n_bandwidth_bin)
		
                toa = np.dot(toa, self.binnedTimingNetwork.basis[snr_bin_ind][freq_bin_ind][bandwidth_bin_ind]) # convert to correct basis
                if verbose: print "tof (basis):\t\t",toa

		return self.likelihood_func(toa, self.tof_map[snr_bin_ind][freq_bin_ind][bandwidth_bin_ind], self.binnedTimingNetwork.errors[snr_bin_ind][freq_bin_ind][bandwidth_bin_ind])

	###
	def set_tof_map(self, npix, pixarray=None):
		""" builds tof map and converts it inot the correct basis (defined in self.binnedTimingNetwork). if basis=None, we assume the detector basis is correct. Resulting map stored within """
		tof_names = self.binnedTimingNetwork.get_tof_names()
		ntof = len(tof_names)
		if self.binnedTimingNetwork.basis == None:
			self.binnedTimingNetwork.basis = np.eye(n_tof)
		if pixarray == None:
			pixarray = __pixarray(npix, nside=None)
		elif len(pixarray) != npix:
			raise ValueError, "len(pixarray)!=npix"
		self.pixarray = pixarray
		self.npix = npix

		tof_map = np.empty((npix, ntof), float) # tau for each combination of detectors
		for ind, (name1, name2) in enumerate(tof_names):
			for ipix, (theta, phi) in enumerate(pixarray):
				tof_map[ipix,ind] = self.binnedTimingNetwork.get_tof(name1,name2,theta,phi)

		self.tof_map = []
		for snr_bin_ind in xrange(self.binnedTimingNetwork.n_snr_bin):
			snr_tof_map = []
			for frq_bin_ind in xrange(self.binnedTimingNetwork.n_freq_bin):
				frq_tof_map = []
				for bdw_bin_ind in xrange(self.binnedTimingNetwork.n_bandwidth_bin):
					basis = self.binnedTimingNetwork.basis[snr_bin_ind][frq_bin_ind][bdw_bin_ind]
					if basis != None:
						frq_tof_map.append( np.dot( tof_map, basis ) ) 
					else:
						frq_tof_map.append( tof_map )
				snr_tof_map.append( frq_tof_map )
			self.tof_map.append( snr_tof_map )

#========================
# Prior (from antenna patterns)
#========================
class Prior(object):
	"""
	a callable object that returns the effective prior (antenna patterns)
	stores
		binnedTimingNetwork, which it references for detectors and their PSDs, antenna patterns
		npix, pixarray which define the specific pixelization used (HEALPix decomposition)
                antenna_patterns which separates each element of the sensitivity matrix by detector and evaluates the antenna patterns according to pixarray
		currnet_prior, current_freq, current_exp all store the current prior, so it doesn't have to be re-computed for each call
	"""

	###
	def __init__(self, binnedTimingNetwork, npix, pixarray=None, freq=100, exp=2.0):
		self.binnedTimingNetwork = binnedTimingNetwork
		self.set_antenna_patterns(npix, pixarray=pixarray, freq=freq, exp=exp)

	###
	def __call__(self, freq, exp=2.0, verbose=False):
		if (self.current_freq == freq) and (self.current_exp == exp):
			return self.current_prior
		elif self.current_freq == freq:
			self.current_exp = exp
			return self.current_prior**(exp/self.current_exp)
		else:
			if verbose: 
				print "computing maximum noise-weighted sensitivity eigenvalue\n\tfreq=%.3f\n\texp=%.3f"%(freq,exp)
			self.current_exp = exp
			self.current_freq = freq

			prior = np.zeros((self.npix,2,2), float)

			### find psd value at this freq, update sensitivity matrix by hand		
			for idet, detector in enumerate(self.binnedTimingNetwork.detectors_list()):
				psd = detector.get_psd().interpolate(freq)
				prior[:,0,0] += self.antenna_patterns[:,idet,0]/psd ### Fp**2/psd
				prior[:,1,1] += self.antenna_patterns[:,idet,2]/psd ### Fx**2/psd
				prior[:,0,1] += self.antenna_patterns[:,idet,1]/psd ### Fp*Fx/psd
#			prior[:,1,0] = prior[:,0,1]

			### compute maximum eigenvalue by hand
			if verbose: 
				print "computing maximum eigenvalues (by hand)"
			prior = 0.5*( prior[:,0,0] + prior[:,1,1] + ( (prior[:,0,0] - prior[:,1,1])**2 + 4*prior[:,0,1]**2 )**0.5 )
		
			### raise max eig to appropriate power
			if verbose:
				print "raising maximum eigenvalue to the (%.3f/2.0) power"%exp
				prior = prior**(exp/2.0)

			return prior
	
	###
	def set_antenna_patterns(self, npix, pixarray=None, freq=100, exp=2.0):
		""" computes antenna patterns once and stores them """
		if pixarray == None:
			pixarray = __pixarray(npix, nside=None)
		if len(pixarray) != npix:
			raise ValueError, "len(pixarray)!=npix"
		self.pixarray = pixarray
		self.npix = npix

		detectors = self.binnedTimingNetwork.detectors_list()
		ndet = len(detectors)
		self.antenna_patterns = np.empty((npix, ndet, 3), float)
		for idet, detector in enumerate(detectors):
			Fp, Fx = detector.antenna_patterns(pixarray[:,0], pixarray[:,1], psi=0.0, dt=0.0) ### compute antenna patterns for all points in the sky
			self.antenna_patterns[:,idet,0] = Fp**2
			self.antenna_patterns[:,idet,1] = Fp*Fx
			self.antenna_patterns[:,idet,2] = Fx**2

		self.current_freq = None
		self.current_exp = None
		self.current_prior = self.__call__(freq, exp=exp, verbose=False)

#========================
# BinnedTimingNetwork
#========================
class BinnedTimingNetwork(utils.Network):
	"""
	an extension of utils.Network that will include time-of-flight errors between detectors
	timing errors are estimated based on binned sample distributions
		we bin according to "snr", "freq", "bandwidth"
	error distributions are stored and referenced by Likelihood objects to compute the timing likelihood
	"""

	###
	def __init__(self, error_approx, snr_bin_edges=[], freq_bin_edges=[], bandwidth_bin_edges=[], detectors=[], freqs=None, Np=2):
		utils.Network.__init__(self, detectors, freqs, Np)

		### error approximations
		self.error_approx = error_approx
		self.errors = None
		self.snr_bin_edges = self.__set_bin_edges(snr_bin_edges)
		self.n_snr_bin = len(self.snr_bin_edges)-1
		self.freq_bin_edges = self.__set_bin_edges(freq_bin_edges)
		self.n_freq_bin = len(self.freq_bin_edges)-1
		self.bandwidth_bin_edges = self.__set_bin_edges(bandwidth_bin_edges)
		self.n_bandwidth_bin = len(self.bandwidth_bin_edges)-1

		self.basis = [ [ [None for bdw_bin_ind in xrange(self.n_bandwidth_bin)] for frq_bin_ind in xrange(self.n_freq_bin) ] for snr_bin_ind in xrange(self.n_snr_bin) ]
	
	###
	def __set_bin_edges(self, bin_edges):
		""" a helper function that sets up bin edges so they will span all possible values """
		if not isinstance(bin_edges, list):
			bin_edges = list(bin_edges)
		if not len(bin_edges):
			bin_edges =  np.array([0,np.infty])
		elif bin_edges[0] > 0:
			bin_edges.insert(0, 0.0)
		elif bin_edges[-1] < np.infty:
			bin_edges.append( np.infty )
		return np.array(bin_edges)

        ###
        def get_tof(self, name1, name2=None, theta=0.0, phi=0.0):
                """compute the expected time-of-flight from name1 -> name2"""
                if name2 == None: # only one name supplied, so we interpret it as a tuple
                        name1,name2 = name1
                return utils.time_of_flight(theta, phi, self.detectors[name2].dr - self.detectors[name1].dr)

        ###
        def get_tof_names(self, names=[]):
                """returns an ordered list of detector combinations"""
                if not len(names):
                        names = self.detectors.keys()
                keys = []
                for ind, name1 in enumerate(names):
                        for name2 in names[ind+1:]:
                                keys.append( tuple(sorted([name1,name2])) )
                return sorted(keys)

	###
	def toacache_to_errs(self, toacache, dt=1e-5, verbose=False, timing=False, hist_errors=False, scatter_errors=False, output_dir="./", tag="", diag=False):
		""" builds errors structure and computes basis from toacache. Results are stored within network """
        	self.errors = [ [ [[] for bdw_bin_ind in xrange(self.n_bandwidth_bin)] for frq_bin_ind in xrange(self.n_freq_bin) ] for snr_bin_ind in xrange(self.n_snr_bin) ]
	        n_err = len(toacache)

		tof_names = self.get_tof_names()
		n_tof = len(tof_names)

		### build total_tof_err to store all observed errors
		tof_errs = np.empty((n_tof, n_err))

	        bound = 0.0
		for tof_ind, (name1, name2) in enumerate(tof_names):
			tof_err = np.empty((n_err,))
			for toa_ind, toa_event in enumerate(toacache):
				### recovered tof
	                        tof = toa_event[name1] - toa_event[name2]
	                        ### injected tof
	                        tof_inj = toa_event[name1+"_inj"] - toa_event[name2+"_inj"]
	                        ### error in tof
	                        tof_err[toa_ind] = tof - tof_inj
			tof_errs[tof_ind,:] = tof_err
			dr = self.detectors[name1].dr - self.detectors[name2].dr
			b = (n_tof)**0.5 * 2*np.dot(dr,dr)**0.5
			if b > bound:
				bound = b

		### iterate and store snrs, freqs, bandwidths
		snrs = np.empty((n_err,),float)
		frqs = np.empty((n_err,),float)
		bdws = np.empty((n_err,),float)
		for toa_ind, toa_event in enumerate(toacache):
			snrs[toa_ind] = toa_event["snr"]
			frqs[toa_ind] = toa_event["freq"]
			bdws[toa_ind] = toa_event["bandwidth"]

		### iterate through bins, downselecting data as we go
		for snr_bin_ind in xrange(self.n_snr_bin):
			msnr = self.snr_bin_edges[snr_bin_ind]
			Msnr = self.snr_bin_edges[snr_bin_ind+1]
		
			snr_truth = (msnr <= snrs)*(snrs < Msnr) ### indexes for events in this snr bin

			for freq_bin_ind in xrange(self.n_freq_bin):
				mfrq = self.freq_bin_edges[freq_bin_ind]
				Mfrq = self.freq_bin_edges[freq_bin_ind+1]

				frq_truth = (mfrq <= frqs)*(frqs < Mfrq) ### indexes for events in this freq bin

				for bdw_bin_ind in xrange(self.n_bandwidth_bin):
					mbdw = self.bandwidth_bin_edges[bdw_bin_ind]
					Mbdw = self.bandwidth_bin_edges[bdw_bin_ind+1]

					bdw_truth = (mbdw <= bdws)*(bdws < Mbdw) ### indexes for events in this bandwidth bin
					
					_tof_errs = tof_errs[:,snr_truth*frq_truth*bdw_truth] ### only those events that live in all selected bins
					_n_err = len(_tof_errs[0])
					if verbose:
						print "snr_bin:\t%d/%d\nfreq_bin:\t%d/%d\nbandwidth_bin:\t%d/%d"%(snr_bin_ind, self.n_snr_bin, frq_bin_ind, self.n_freq_bin, bdw_bin_ind, self.n_bandwidth_bin)
						print "%d / %d events"%(_n_err, n_err)

					### compute covariance matrix and diagonalize
					cov = np.cov(_tof_errs)
					if verbose: print "cov:\n", cov
					if diag and (n_tof > 1):
						if verbose: print "diagonalizing covariance matrix and defining linear combinations of timing errors"
						eigvals, _eigvecs = np.linalg.eig( cov )
						cov = np.cov( np.dot(np.transpose(_eigvecs), tof_errs) )
						if verbose:
							print "eigval:\n", np.diag(eigvals)
							print "basis:\n" , _eigvecs
							print "new cov:\n", cov
	

						### reduce the number of dimensions if needed
						truth = eigvals >= dt**2
						if not truth.all():
							if verbose: print "reducing the number of degrees of freedom in the system because %d eigvals are too small to measure (dt=%.3fms)"%(n_tof-sum(truth), dt*1e3)
							truth = eigvals >= dt**2 # figure out which eigvals are measurable
							eigvecs = np.empty((n_tof,sum(truth))) # array for new basis
							eig_ind = 0
							for tof_ind in xrange(_n_tof):
								if truth[tof_ind]: # variance is big enough to measure
									eigvecs[:,eig_ind] = _eigvecs[:,tof_ind]
									eig_ind += 1
								else: # variance is too small to measure. We drop it
									pass
							cov = np.cov( np.dot(np.transpose(eigvecs), _tof_errs) )
							if verbose:
								print "eigval:\n", eigvals[truth]
								print "basis:\n", eigvecs
								print "new cov:\n", cov
						else:
							eigvecs = _eigvecs
							eig_ind = _n_tof
					else: 
						eigvecs = np.eye(n_tof)
						eig_ind = n_tof # all the eigvals are allowable

					self.basis[snr_bin_ind][frq_bin_ind][bdw_bin_ind] = eigvecs ### establish basis for this bin

					### iterate through basis and compute errors
					if scatter_errors:
						tof_errs_summary = []

					for tof_ind in xrange(eig_ind):
						if verbose: 
							s = "\tbasis:\n"
							for i, (name1,name2) in enumerate(tof_names):
								s += "\t\t%.4f*(t_{%s}-t_{%s})\n"%(eigvecs[i,tof_ind], name1, name2)
							print s[:-1]

						e_tof_err = np.dot(eigvecs[:,tof_ind], _tof_errs) # transform into the correct basis
	
						m = np.mean(e_tof_err)
						e = np.std(e_tof_err)
		
	        			        z = 0.1 # consistency check to make sure the tof errors are not crazy
			        	        if abs(m) > z*e:
                				        ans = raw_input("\nmeasured mean (%f) is larger than %.3f of the standard deviation (%f) for tof_ind: %d\n\tcontinue? [Y/n] "%(m,z,e,tof_ind))
			                        	if ans != "Y":
                        			        	raise ValueError, "measured mean (%f) is larger than %.3f of the standard deviation (%f) for tof: %d"%(m,z,e,tof_ind)
				                        else:
        	        			                pass

			                	### add errors to the errs
				                if self.error_approx == "gaussian":
			        	                self.errors[snr_bin_ind][frq_bin_ind][bdw_bin_ind].append( e )
							kde = None # for hist_errors

				                elif self.error_approx == "kde": ### single kde estimate for the entire sky
			        	                samples = np.arange(-bound, bound+dt, dt)
#							samples = np.linspace(-10*e, 10*e, 1e5+1)
							kde = singlekde(samples, e_tof_err, e, verbose=verbose, timing=timing)
							self.errors[snr_bin_ind][frq_bin_ind][bdw_bin_ind].append( kde )
			                	else:
			                        	raise ValueError, "error-approx=%s not understood"%self.binnedTimingNetwork.error_approx

					        if hist_errors: # generate histogram of errors
					                label = "t_{%d}"%tof_ind
	
        					        if verbose: print "\thistogram for %s\n\t\tm=%.4fms\n\t\te=%.4fms"%(label, m, e)
				
			                		fig, ax = self.__hist_errors(e_tof_err, label, n_err=_n_err, m=m, e=e, kde=kde)

		        			        figname = output_dir+"/s-%d_f-%d_b-%d__tof-err_%s%s.png"%(snr_bin_ind, frq_bin_ind, bdw_bin_ind, label,tag)
			        		        if verbose: print "\tsaving", figname
			                		fig.savefig(figname)
				                	plt.close(fig)


						if scatter_errors: # generate scatter and projected histograms
							for _tof_ind, (_e_tof_err, _m, _e) in enumerate(tof_errs_summary):
								label = "t_{%d}"%tof_ind
								_label = "t_{%d}"%_tof_ind
								p = np.dot(e_tof_err-m, _e_tof_err-_m)/(_n_err*e*_e) # should be vanishingly small, but we'll check

								if verbose: print "\tscatter for %s vs %s\n\t\tpearson=%.5f"%(label,_label,p)

								fig, ax =  self.__scatter_errors(e_tof_err, label, _e_tof_err, _label, n_err=_n_err, m=m, e=e, _m=_m, _e=_e)

								fig.text(0.15+0.025, 0.70-0.025, "$\\rho=%.5f$\n$N=%d$"%(p,n_err), ha="left", va="top", color="b", fontsize=12)
			                        	        fig.text(0.15+0.025, 0.925-0.025, "$\mu=%.3f\mathrm{ms}$\n$\sigma=%.3f\mathrm{ms}$"%(m*1e3, e*1e3), ha="left", va="top", color="b", fontsize=12)
			                                	fig.text(0.975-0.025, 0.700-0.025, "$\mu=%.3f\mathrm{ms}$\n$\sigma=%.3f\mathrm{ms}$"%(_m*1e3, _e*1e3), ha="right", va="top", color="b", fontsize=12)

								figname = output_dir+"//s-%d_f-%d_b-%d__tof-scat_%d_%d%s.png"%(snr_bin_ind, frq_bin_ind, bdw_bin_ind, tof_ind, _tof_ind, tag)
			        	                        if verbose: print "\tsaving", figname
			                	                fig.savefig(figname)
                        				        plt.close(fig)

							tof_errs_summary.append( (e_tof_err, m, e) ) ### add only at the end to we don't plot auto-correlations

					if self.error_approx == "gaussian":
						self.errors = np.array(self.errors)

					elif self.error_approx == "kde":
						self.errors = [ [ [ IndependentKDE( self.errors[snr_bin_ind][frq_bin_ind][bdw_bin_ind] ) for bdw_bin_ind in xrange(self.n_bandwidth_bin)] for frq_bin_ind in xrange(self.n_freq_bin)]for snr_bin_ind in xrange(self.n_snr_bin)]


	#========================
	# plotting error distributions
	#========================
	def __hist_errors(self, tof_err, label, n_err=None, m=None, e=None, kde=None):
		"""
		generate a histogram of the observed error distribution
		if kde!=None:
			we also overlay the kde estimate
		"""
        	if n_err == None:
	                n_err = len(tof_err)
        	if m == None:
	                m = np.mean(tof_err)
        	if e == None:
	                e = np.std(tof_err)

        	fig = plt.figure()
	        ax  = plt.subplot(1,1,1)

        	ax.hist(tof_err*1e3, bins=n_err/10, histtype="step", log=True, normed=True, label="$N=%d$\n$\mu=%.3f\mathrm{ms}$\n$\sigma=%.3f\mathrm{ms}$"%(len(tof_err), m*1e3, e*1e3))
	        if kde:
        	        ylim = ax.get_ylim()
                	xlim = ax.get_xlim()
#	               ax.plot(kde.__dict__['x']*1e3, kde.__dict__['y'], color="r", alpha=0.5, label="kde estimate")
        	        ax.plot(kde[0]*1e3, kde[1], color="r", alpha=0.5, label="kde estimate")
                	ax.set_ylim(ymin=ylim[0])
	                ax.set_xlim(xlim)
        	ax.grid(True)
	        ax.set_xlabel("$\Delta\left(%s\\right) [\mathrm{ms}]$"%(label))
        	ax.set_ylabel("probability density")
	        ax.legend(loc="upper left")

        	return fig, ax

	###
	def __scatter_errors(self, tof_err, label, _tof_err, _label, n_err=None, m=None, e=None, _m=None, _e=None):
		"""
		generates a scatter of the observed error distributions
		"""
        	if n_err == None:
	                n_err = len(tof_err)
        	if m == None:
	                m = np.mean(tof_err)
        	if e == None:
	                e = np.std(tof_err)
        	if _m == None:
	                _m = np.mean(_tof_err)
        	if _e == None:
	                _e = np.std(_tof_err)

        	fig = plt.figure()

	        ax  = fig.add_axes([0.150, 0.100, 0.600, 0.600])
        	axu = fig.add_axes([0.150, 0.725, 0.600, 0.200])
	        axr = fig.add_axes([0.775, 0.100, 0.200, 0.600])

        	bins = n_err/10
	        ax.plot(tof_err, _tof_err, marker="o", markerfacecolor="none", markeredgecolor="b", markersize=2, linestyle="none")
        	axu.hist(tof_err, bins, histtype="step", log=True)
	        axr.hist(_tof_err, bins, histtype="step", log=True, orientation="horizontal")

        	ax.grid(True)
	        axu.grid(True)
        	axr.grid(True)

	        ax.set_xlabel("$\Delta\left(%s\\right) [\mathrm{ms}]$"%(label))
        	ax.set_ylabel("$\Delta\left(%s\\right) [\mathrm{ms}]$"%(_label))

	        axu.set_ylabel("count")
        	plt.setp(axu.get_xticklabels(), visible=False)

	        axr.set_xlabel("count")
        	plt.setp(axr.get_yticklabels(), visible=False)

	        return fig, ax

#========================
# kde interpolation object
#========================
class IndependentKDE(object):
	"""
	a callable object that stores several singlekde decompositions
	computes the joint distribution assuming each singlekde distribution is independent
	"""
        def __init__(self, kdes):
                self.kdes = kdes

        def add_kde(self, kde):
                self.kdes.append( kde )

        def __call__(self, x):
                shape_x = np.shape(x)
                len_shape_x = len(shape_x)
                if len_shape_x == 1: # intepreted as a single vector
                        if len(x) != len(self.kdes):
                                raise ValueError, "bad shape for x", shape_x
                        p = 1.0
                        for _x, _kde in zip(x,self.kdes):
                                p *= _kde(_x)
                        return p
                elif len_shape_x == 2: # interpreted as a list of vectors
                        if shape_x[1] != len(self.kdes):
                                raise ValueError, "bad shape for x", shape_x
                        p = np.ones((shape_x[0],))
                        for ind, (samples, samples_kde) in enumerate(self.kdes):
                                p *= np.interp(x[:,ind], samples, samples_kde)

                        return p
                else:
                        raise ValueError, "bad shape for x :", shape_x

#========================
# helper functions for error estimation
#========================
def singlekde(samples, tof_err, e, precision_limit=0.001, max_iters=5, verbose=False, timing=False):
	"""
	builds a singlekde out of the observed tof_err 
	sampled at samples
	"""
	frac = (0.005)**2 # the fraction of the entire distribution's width used in kde estimate
        if verbose:
		print "\tfixed_bandwidth kde"
		if timing:
			t1=time.time()
	### fixed_bandwidth to start
	samples_kde = pdfe.fixed_bandwidth_gaussian_kde(samples, tof_err, v=frac*e)
	fbw_samples_kde = samples_kde
	if verbose and timing: print "\t\t", time.time()-t1, "sec"

	### iterate with point_wise kde and look for convergence

	z = 1000/(1.0+len(tof_err)) ### fraction of e used as scale in point_wise kde algorithm
	                            ### ad hoc formula that will use larger scales for smaller sample sets
	                            ### may not be optimal...

	for _ in range(max_iters):
		if verbose:
			print "\tpoint_wise kde"
			if timing: t1=time.time()
		old_samples_kde = samples_kde

		samples_kde = pdfe.point_wise_gaussian_kde(samples, tof_err, scale=z*e, pilot_x=samples, pilot_y=samples_kde)
		precision = 1 - sum(samples_kde*old_samples_kde)/(sum(samples_kde**2) * sum(old_samples_kde**2))**0.5
		if verbose:
			print "\t\tprecision=%.6f"%precision
			if timing: print "\t\t", time.time()-t1, "sec"
		if precision_limit > precision:
			break
	else:
		if verbose:
			print "\tprecision_limit=%.6f not reached after %d iterations. Falling back to fixed_bandwidth kde"%(precision_limit, max_iters)
			samples_kde = fbw_samples_kde

	### build interpolation object and add it to the network
	return samples, samples_kde

