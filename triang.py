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
	if (N-1,) != np.shape(err):
		raise ValueError, "tof_map and err_map must have the same shape!"
	elif N != len(toa)+1:
		raise ValueError, "shape mismatch between toa and tof_map"

	return np.exp( -0.5*sum( np.transpose((toa-tof_map[:,1:])/err)**2 ) ) # transposition allows us to sum over detector pairs

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
	if N != len(toa)+1:
		raise ValueError, "shape mismatch between toa and tof_map"

	return kde(toa-tof_map[:,1:])

#========================
# priors
#========================
def prior(ap_map, which="shells"):
	"""
	computes the prior assigned to each point using the antenna patterns stored in ap_map
	"""
	if which == "flat":
		### no prior
		return np.ones((len(ap_map[:,-1]),))
	elif which == "shells":
		### uniform in volume (prior follows maximum eigenvalue of sensitivity matrix)
		return ap_map[:,-1] # shells in the sky
	elif which == "spheres":
		return ap_map[:,-1]**(3/2) # spheres in the sky
	else:
		try:
			return ap_map[:,-1]**float(which)
		except:
			raise ValueError, "type=%s not understood"%which 

#========================
# skymaps
#========================
def __pixarray(nside):
	"""
	defines a pixelization array as a short-cut between ipix and (theta,phi)
	"""
	npix = hp.nside2npix(nside)
        pixarray = np.zeros((npix,3))
        for ipix in xrange(npix):
                theta, phi = hp.pix2ang(nside, ipix)
                pixarray[ipix,:] = np.array([ipix, theta, phi])
        return pixarray

###
def tof_map(npix, network, pixarray=None, ntof=None):
	"""
	computes time-of-flight between detectors for different source positions
	"""
	if pixarray == None:
		pixarray = __pixarray(npix)
	if not ntof:
		ntof = len(network.get_tof_names())

        tof_map = np.zeros((npix,1+ntof)) # ipix, tau for each combination of detectors
        tof_map[:,0] = pixarray[:,0]
        for ind, (name1, name2) in enumerate(network.get_tof_names()):
                for ipix, theta, phi in pixarray:
                        tof_map[ipix,1+ind] = network.get_tof(name1,name2,theta,phi)
	return tof_map

###
def __ap_map(npix, network, pixarray=None, no_psd=True):
	"""
	computes the antenna pattern map used to weight to compute the prior
	"""
	if pixarray == None:
		pixarray = __pixarray(npix)

        ap_map = np.zeros((npix,2)) # ipix, |F|
        for ipix, theta, phi in pixarray:
		a = network.A_dpf(theta, phi, no_psd=no_psd) # get sensitivity matrix in dominant polarization frame
		F = np.amax(a) # get the maximum eigenvalue
                ap_map[ipix] = np.array([ipix,F])

	return ap_map

#========================
# error estimation
#========================
def __hist_errors(tof_err, label, n_err=None, m=None, e=None, kde=None):
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
#		ax.plot(kde.__dict__['x']*1e3, kde.__dict__['y'], color="r", alpha=0.5, label="kde estimate")
		ax.plot(kde[0]*1e3, kde[1], color="r", alpha=0.5, label="kde estimate")
		ax.set_ylim(ymin=ylim[0])
		ax.set_xlim(xlim)
	ax.grid(True)
	ax.set_xlabel("$\Delta\left(%s\\right) [\mathrm{ms}]$"%(label))
	ax.set_ylabel("probability density")
	ax.legend(loc="upper left")

	return fig, ax

###
def __scatter_errors(tof_err, label, _tof_err, _label, n_err=None, m=None, e=None, _m=None, _e=None):
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

###
def toacache_to_errs(toacache, timing_network, error_approx="gaussian", dt=1e-5, verbose=False, timing=False, hist_errors=False, scatter_errors=False, output_dir="./", tag="", diag=False):
        """
        loads observed time-fo-arrival information and builds errors suitable to be loaded into TimingNetwork
        """
        errs = []
        n_err = len(toacache)

	tof_names = timing_network.get_tof_names()
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
		dr = timing_network.detectors[name1].dr - timing_network.detectors[name2].dr
		b = (n_tof)**0.5 * 2*np.dot(dr,dr)**0.5
		if b > bound:
			bound = b
		
	### compute covariance matrix and diagonalize
	cov = np.cov(tof_errs)
	if verbose: print "cov:\n", cov
	if diag and (n_tof > 1):
		if verbose: 
			print "diagonalizing covariance matrix and defining linear combinations of timing errors"
		eigvals, _eigvecs = np.linalg.eig( cov )
		cov = np.cov( np.dot(np.transpose(_eigvecs), tof_errs) )
		if verbose:
			print "eigval:\n", np.diag(eigvals)
			print "basis:\n" , _eigvecs
			print "new cov:\n", cov
	
		### reduce the number of dimensions if needed
		truth = eigvals >= dt**2
		if not truth.all():
			if verbose:
				print "reducing the number of degrees of freedom in the system because %d eigvals are too small to measure (dt=%.3fms)"%(n_tof-sum(truth), dt*1e3)
			truth = eigvals >= dt**2 # figure out which eigvals are measurable
			eigvecs = np.empty((n_tof,sum(truth))) # array for new basis
			eig_ind = 0
			for tof_ind in xrange(n_tof):
				if truth[tof_ind]: # variance is big enough to measure
					eigvecs[:,eig_ind] = _eigvecs[:,tof_ind]
					eig_ind += 1
				else: # variance is too small to measure. We drop it
					pass
			cov = np.cov( np.dot(np.transpose(eigvecs), tof_errs) )
			if verbose:
				print "eigval:\n", eigvals[truth]
				print "basis:\n", eigvecs
				print "new cov:\n", cov
		else:
			eigvecs = _eigvecs
			eig_ind = n_tof
	else: 
		eigvecs = np.eye(n_tof)
		eig_ind = n_tof # all the eigvals are allowable

	### iterate through basis and compute singlekde
	if scatter_errors:
		tof_errs_summary = []

	for tof_ind in xrange(eig_ind):
		if verbose: 
			s = "\tbasis:\n"
			for i, (name1,name2) in enumerate(tof_names):
				s += "\t\t%.4f*(t_{%s}-t_{%s})\n"%(eigvecs[i,tof_ind], name1, name2)
			print s[:-1]

		tof_err = np.dot(eigvecs[:,tof_ind], tof_errs) # transform into the correct basis

		m = np.mean(tof_err)
		e = np.std(tof_err)
		
                z = 0.1 # consistency check to make sure the tof errors are not crazy
                if abs(m) > z*e:
                        ans = raw_input("measured mean (%f) is larger than %.3f of the standard deviation (%f) for tof_ind: %d\n\tcontinue? [Y/n] "%(m,z,e,tof_ind))
                        if ans != "Y":
                                raise ValueError, "measured mean (%f) is larger than %.3f of the standard deviation (%f) for tof: %d"%(m,z,e,tof_ind)
                        else:
                                pass

                ### add errors to the errs
                if error_approx == "gaussian":
                        errs.append( e )
			kde = None # for hist_errors

                elif error_approx == "kde": ### single kde estimate for the entire sky
                        samples = np.arange(-bound, bound+dt, dt)
#                        samples = np.linspace(-10*e, 10*e, 1e5+1)
			kde = singlekde(samples, tof_err, e, verbose=verbose, timing=timing)
			errs.append( kde )

                else:
                        raise ValueError, "error-approx=%s not understood"%opts.e_approx

	        if hist_errors: # generate histogram of errors
	                label = "t_{%d}"%tof_ind

        	        if verbose: print "\thistogram for %s\n\t\tm=%.4fms\n\t\te=%.4fms"%(label, m, e)
			
                	fig, ax = __hist_errors(tof_err, label, n_err=n_err, m=m, e=e, kde=kde)

	                figname = output_dir+"/tof-err_%s%s.png"%(label,tag)
        	        if verbose: print "\tsaving", figname
                	fig.savefig(figname)
	                plt.close(fig)


		if scatter_errors: # generate scatter and projected histograms
			for _tof_ind, (_tof_err, _m, _e) in enumerate(tof_errs_summary):
				label = "t_{%d}"%tof_ind
				_label = "t_{%d}"%_tof_ind
				p = np.dot(tof_err-m, _tof_err-_m)/(n_err*e*_e) # should be vanishingly small, but we'll check

				if verbose: print "\tscatter for %s vs %s\n\t\tpearson=%.5f"%(label,_label,p)

				fig, ax =  __scatter_errors(tof_err, label, _tof_err, _label, n_err=n_err, m=m, e=e, _m=_m, _e=_e)

				fig.text(0.15+0.025, 0.70-0.025, "$\\rho=%.5f$\n$N=%d$"%(p,n_err), ha="left", va="top", color="b", fontsize=12)
                                fig.text(0.15+0.025, 0.925-0.025, "$\mu=%.3f\mathrm{ms}$\n$\sigma=%.3f\mathrm{ms}$"%(m*1e3, e*1e3), ha="left", va="top", color="b", fontsize=12)
                                fig.text(0.975-0.025, 0.700-0.025, "$\mu=%.3f\mathrm{ms}$\n$\sigma=%.3f\mathrm{ms}$"%(_m*1e3, _e*1e3), ha="right", va="top", color="b", fontsize=12)

				figname = output_dir+"/tof-scat_%d_%d%s.png"%(tof_ind, _tof_ind, tag)
                                if verbose: print "\tsaving", figname
                                fig.savefig(figname)
                                plt.close(fig)

			tof_errs_summary.append( (tof_err, m, e) ) ### add only at the end to we don't plot auto-correlations

	if error_approx == "gaussian":
		errs = np.array(errs)

	elif error_approx == "kde":
		errs = IndependentKDE( errs )

	return errs, eigvecs

###
def singlekde(samples, tof_err, e, precision_limit=0.001, max_iters=5, verbose=False, timing=False):
	"""
	builds a singlekde out of the observed tof_err sampled at samples
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

	### iterate with point_wide kde and look for convergence

#	z = 1.0 ### fraction of e used as scale 
	z = 1000/(1.0+len(tof_err)) ### fraction of e used as scale
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
#	return pdfe.scipy.interpolate.interp1d(samples, samples_kde, kind="linear")

###
class IndependentKDE(object):

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
#                        for ind, kde in enumerate(self.kdes):
#				p *= kde(x[:,ind])
                        for ind, (samples, samples_kde) in enumerate(self.kdes):
                                p *= np.interp(x[:,ind], samples, samples_kde)

                        return p
                else:
                        raise ValueError, "bad shape for x :", shape_x


#========================
# TimingNetwork
#========================
class TimingNetwork(utils.Network):
	"""
	an extension of utils.network that will include time-of-flight errors between detectors
	we assume the timing errors are gaussian so we can describe them with a single number (the standard deviation)
	"""

	###
	def __init__(self, error_approx, detectors=[], freqs=None, Np=2):
		utils.Network.__init__(self, detectors, freqs, Np) # call parent's constructor

		self.error_approx = error_approx
		self.basis = None
		self.errors = None
		self.tof_map = None

	###
	def toacache_to_errs(self, toacache, dt=1e-5, verbose=False, timing=False, hist_errors=False, scatter_errors=False, output_dir="./", tag="", diag=False):
		""" builds errors structure and computes basis from toacache. Results are stored within network """
		self.errors, self.basis = toacache_to_errs(toacache, self, error_approx=self.error_approx, dt=dt, verbose=verbose, timing=timing, hist_errors=hist_errors, scatter_errors=scatter_errors, output_dir=output_dir, tag=tag, diag=diag)
	
	###	
	def set_tof_map(self, npix, pixarray=None):
		""" builds tof map and converts it into the correct basis. If basis==None, we assume the detector basis is correct (identity matrix). Resulting map is stored within network """
		if self.basis == None:
			n_tof = len(self.get_tof_names)
			self.basis = np.eye(n_tof)
		self.tof_map = np.empty((npix,1+np.shape(self.basis)[1]))
                self.tof_map[:,0] = pixarray[:,0]
		
                self.tof_map[:,1:] = np.dot( tof_map(npix, self, pixarray=pixarray, ntof=len(self.get_tof_names()))[:,1:], self.basis ) # dot product to convert basis

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
	def likelihood(self, toa, verbose=False):
		""" computes the likelihood of observing toa """
		toa = np.dot(toa, self.basis) # convert to correct basis
		if verbose: print "tof (basis):\t\t",toa
		if self.error_approx == "gaussian":
                        return gaussian_likelihood(toa, self.tof_map, self.errors)

                elif self.error_approx == "kde":
                        return kde_likelihood(toa, self.tof_map, self.errors)
                else:
                        raise ValueError, "error-approx=%s not understood"%self.error_approx


