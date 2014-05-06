### written by R.Essick (ressick@mit.edu)

usage = """triangulation.py [--options] detector_name1 detector_name2 detector_name3 ...
written to compute posteriors over the sky using triangulation and modulation with antenna patterns"""

import utils
np = utils.np
import healpy as hp
import pickle

#=================================================
#
#                  utilities
#
#=================================================
def gaussian_likelihood(toa, tof_map, err_map):
	"""
	computes the likelihood that a signal came from the points in tof_map and produced toa, assuming gaussian errors defined in err_map
	require:	shape(toa)     = (N,)
			shape(tof_map) = (Npix,1+N)
			shape(err_map) = (Npix,1+N)
	"""
	Npix,N = np.shape(tof_map)
	if (Npix,N) != np.shape(err_map):
		raise ValueError, "tof_map and err_map must have the same shape!"
	elif N != len(toa)+1:
		raise ValueError, "shape mismatch between toa and tof_map"

	return np.exp( -0.5*sum(np.transpose( ((toa-tof_map[:,1:])/err_map[:,1:])**2 )) ) # transposition allows us to sum over detector pairs
#	return (2*np.pi*err_map[:,1:]**2)**0.5 * np.exp( -(toa-tof_map[:,1:])**2/(2*err_map[:,1:]**2) )

###
def kde_likelihood(toa, tof_map, kde_map):
	"""
	computes the likelihood that a signal came from the points in tof_map and produced toa, assuming the measured error distributions in kde_map
	allows for different distributions at each point in the sky (through kde_map)
	require: 	shape(toa)	= (N,)
			shape(tof_map)	= (Npix,1+N)
			shape(kde_map)  = (Npix,1+N)
	"""
	Npix,N = np.shape(tof_map)
	if (Npix,N) != np.shape(err_map):
		raise ValueError, "tof_map and err_map must have the same shape!"
	elif N != len(toa)+1:
		raise ValueError, "shape mismatch between toa and tof_map"

	likelihood = np.empty((Npix,))
	for ipix, dtof in enumerate(toa-tof_map[:,1:]): # errors in time-of-flight for every pixel in the tof_map
		likelihood[ipix] = kde_map[ipix](dtof)

	return likelihood

###
def singlekde_likelihood(toa, tof_map, kde):
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

###
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

###
class IndependentKDE(object):
	kdes = []

	def __init__(self, kdes):
		self.kdes = kdes

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
			for ind, _x in enumerate(x):
				_p = 1.0
				for _X, _kde in zip(_x,self.kdes):
					_p *= _kde(_X)
				p[ind] = _p
			return p
		else:
			raise ValueError, "bad shape for x :", shape_x

###
class TimingNetwork(utils.Network):
	"""
	an extension of utils.network that will include time-of-flight errors between detectors
	we assume the timing errors are gaussian so we can describe them with a single number (the standard deviation)
	"""
	errors = dict()

	def __init__(self, errs={}):
		for (detector1,detector2), err in errs.items():
			self.add_err(detector1,detector2,err)
		self.check()

	def check(self):
		all_names = sorted(self.detectors.keys())
		n = len(all_names)
		for ind1 in xrange(n):
			name1 = all_names[ind1]
			for ind2 in xrange(ind1+1,n):
				names = (name1, all_names[ind2])
				if not self.errors.has_key(names):
					raise KeyError, "TimingNetwork is missing err for %s and %s"%(name1,all_names[ind2])

	def add_err(self, detector1, detector2, err):
		"""adds detector1,detector2 with err to the network. DOES NOT CHECK TO MAKE SURE ALL POSSIBLE DETECTOR COMBINATIONS ARE PRESENT"""
		name1 = detector1.name
		name2 = detector2.name
		self.detectors[name1] = detector1
		self.detectors[name2] = detector2
		names = tuple(sorted([name1,name2]))
		self.errors[names] = err

	def set_err(self, name1, name2, err, theta=0.0, phi=0.0):
		"""associates err with name1 and name2. Does not checks whether they are in the network first"""
		_name1 = self.contains_name(name1)
		_name2 = self.contains_name(name2)
		if _name1 and _name2:
			names = tuple(sorted([name1, name2]))
			self.errors[names] = err
		elif _name1:
			raise KeyError, "TimingNetwork does not contain %s"%name2
		elif _name2:
			raise KeyError, "TimingNetwork does not contain %s"%name1
		else:
			raise KeyError, "TimingNetwork does not contain either %s or %s"%(name1,name2)

	def get_err(self, name1, name2, theta=0.0, phi=0.0):
		"""returns the associated error for name1,name2"""
		_name1 = self.contains_name(name1)
		_name2 = self.contains_name(name2)
		if _name1 and _name2:
			names = tuple(sorted([name1, name2]))
			return self.errors[names]
		elif _name1:
			raise KeyError, "TimingNetwork does not contain %s"%name2
		elif _name2:
			raise KeyError, "TimingNetwork does not contain %s"%name1
		else:
			raise KeyError, "TimingNetwork does not contain either %s or %s"%(name1,name2)

	def get_tof(self, name1, name2, theta, phi):
		"""compute the expected time-of-flight from name1 -> name2"""
		return utils.time_of_flight(theta, phi, self.detectors[name2].dr - self.detectors[name1].dr)

	def get_tof_names(self, names=[]):
		"""returns an ordered list of detector combinations"""
		if len(names):
			keys = []
			for ind, name1 in enumerate(names):
				for name2 in names[ind+1:]:
					keys.append( tuple(sorted([name1,name2])) )
			return sorted(keys)
		else:
			return sorted(self.errors.keys())

#=================================================
#
#                    MAIN
#
#=================================================
if __name__ == "__main__":
	from optparse import OptionParser
	parser = OptionParser(usage=usage)

	parser.add_option("-v", "--verbose", dest="verbose", default=False, action="store_true")

	parser.add_option("-a", "--arrivals-cache", dest="a_cache", default="arrivals.cache", type="string", help="a cache file containing the times-of-arrival for the detectors in this network")
	parser.add_option("", "--deg", default=False, action="store_true", help="if True, we convert injected phi,theta to radians. only important for --arrivals-cache (for which posteriors are calculated).")

	parser.add_option("-e", "--errors-cache", dest="e_cache", default="errors.cache", type="string", help="a cache file containing errors in time-of-flight measurements between detectors")
	parser.add_option("", "--error-approx", dest="e_approx", default="gaussian", type="string", help="how the triangulation code estimates time-of-flight errors")

	parser.add_option("", "--hist-errors", default=False, action="store_true", help="histogram the tof errors observed in --errors-cache")

	parser.add_option("-n", "--nside-exp", default=7, type="int", help="HEALPix NSIDE parameter for pixelization is 2**opts.nside_exp")

	parser.add_option("-w", "--write-posteriors", default=False, action="store_true", help="generate FITs files for posteriors")
	parser.add_option("-p", "--plot-posteriors", default=False, action="store_true", help="generate plots for posteriors")
	parser.add_option("-s", "--stats", default=False, action="store_true", help="compute basic statistics about the reconstruction.")

	parser.add_option("", "--scatter", default=False, action="store_true", help="generate a scatter plot of the entire population")

	parser.add_option("-o", "--output-dir", default="./", type="string")
	parser.add_option("-t", "--tag", default="", type="string")

	parser.add_option("", "--time", default=False, action="store_true")

	parser.add_option("", "--prior", default="shells", type="string", help="flat, shells, spheres, or a float")

	opts, args = parser.parse_args()
	if not len(args):
		raise ValueError, "supply at least 2 detector names as arguments"

	args = sorted(args)

	nside = 2**opts.nside_exp

	if opts.tag:
		opts.tag = "_%s"%opts.tag

	if opts.plot_posteriors:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

	if opts.time:
		import time

	if opts.hist_errors:
		import matplotlib
		matplotlib.use("Agg")
		from matplotlib import pyplot as plt

	if "kde" in opts.e_approx:
		import pdf_estimation as pdfe

	### get list of detectors from arguments
	if opts.verbose: 
		print "loading list of detectors"
		if opts.time: to=time.time()
	detectors = {}
	for arg in args:
		if arg == "H1": #"LHO":
			detectors[arg] = utils.LHO
		elif arg == "L1": #"LLO":
			detectors[arg] = utils.LLO
		elif arg == "V1": #"Virgo":
			detectors[arg] = utils.Virgo
		else:
			raise ValueError, "detector=%s not understood"%arg
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	ndetectors = len(detectors) # number of detectors
        ntof = ndetectors*(ndetectors-1)/2 # number of combinations of detectors

	### instantiate an empty TimingNetwork to use some functions
	network = TimingNetwork()

	### get set of timing errors from error_cache
	###   we select only those errors which are relevant to this network
	if opts.verbose:
		print "computing error distributions from", opts.e_cache
		if opts.time: to = time.time()
	err_file = open(opts.e_cache, "r")
	e_cache = pickle.load(err_file)
	err_file.close()

	n_err = len(e_cache)
	for name1, name2 in network.get_tof_names(names=detectors.keys()):
		tof_err = np.empty((n_err,))
		for ind, toa_event in enumerate(e_cache):	
			### recovered tof
			tof = toa_event[name1] - toa_event[name2]
			### injected tof
			tof_inj = toa_event[name1+"_inj"] - toa_event[name2+"_inj"]
			### error in tof
			tof_err[ind] = tof - tof_inj
		m = np.mean(tof_err)
		e = np.std(tof_err) # get standard deviation of this distribution

		z = 0.1 # consistency check to make sure the tof errors are not crazy
		if abs(m) > z*e:
			ans = raw_input("measured mean (%f) is larger than %.3f of the standard deviation (%f) for tof:%s-%s\n\tcontinue? [Y/n] "%(m,z,e,name1,name2))
			if ans != "Y":
				raise ValueError, "measured mean (%f) is larger than %.3f of the standard deviation (%f) for tof:%s-%s"%(m,z,e,name1,name2)
			else:
				pass

		### add errors to the network
		if opts.e_approx == "gaussian":
			network.add_err(detectors[name1],detectors[name2],e)

		elif opts.e_approx == "singlekde": ### single kde estimate for the entire sky
			num_samples = 1001 # number of samples used in kde estimate
			bound = 0.040 # 40 ms as maximum error... should be ~twice the possible error
			samples = np.linspace(-bound, bound, num_samples)

			### build kde estimate
			frac = (0.005)**2 # the fraction of the entire distribution's width used in kde estimate
			kde = pdfe.scipy.interpolate.interp1d(samples, pdfe.fixed_bandwidth_gaussian_kde(samples, tof_err, v=frac*e), kind="linear")

			network.add_err(detectors[name1],detectors[name2],kde)

		elif opts.e_approx == "kde_map": ### kde estimates for each point in the sky
			raise StandardError, "write kde_map"
		else:
			raise ValueError, "error-approx=%s not understood"%opts.e_approx

                if opts.hist_errors: # generate histogram of errors
                        if opts.verbose: print "\thistogram for %s-%s"%(name1,name2)
                        fig = plt.figure()
                        ax  = plt.subplot(1,1,1)
                        ax.hist(tof_err*1e3, bins=n_err/10, histtype="step", log=True, normed=True, label="$N=%d$\n$\mu=%.3f$\n$\sigma=%.3f$"%(len(tof_err), m*1e3, e*1e3))
			if opts.e_approx == "singlekde":
				ylim = ax.get_ylim()
				xlim = ax.get_xlim()
				estimate = kde(samples)
				ax.plot(samples*1e3, estimate, color="r", alpha=0.5, label="kde estimate")
				ax.set_ylim(ymin=ylim[0])
				ax.set_xlim(xlim)
                        ax.grid(True)
                        ax.set_xlabel("t_%s - t_%s [ms]"%(name1,name2))
                        ax.set_ylabel("probability density")
                        ax.legend(loc="upper left")
                        figname = opts.output_dir+"/tof-err_%s-%s%s.png"%(name1,name2,opts.tag)
                        if opts.verbose: print "\tsaving", figname
                        plt.savefig(figname)
                        plt.close(fig)


	network.check() # checks network for consistency. If it isn't consistent, raises a KeyError
	
	if opts.verbose:
		print "built TimingNetwork\n\t", network
		if opts.time: print "\t", time.time()-to, "sec"

	### number of pixels in the sky map
	npix = hp.nside2npix(nside)
	pixarea = hp.nside2pixarea(nside)
	pixarea_deg = pixarea/utils.deg2rad**2
        if opts.verbose: 
		print "pixelating the sky with nside=%d ==> %d pixels"%(nside,npix)
		if opts.time: to = time.time()
	pixarray = np.zeros((npix,3))
	for ipix in np.arange(npix):
		theta, phi = hp.pix2ang(nside, ipix)
		pixarray[ipix,:] = np.array([ipix, theta, phi])
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	### build sky map of expected time-of-flights
	if opts.verbose: 
		print "computing tof_map"
		if opts.time: to=time.time()
	tof_map = np.zeros((npix,1+ntof)) # ipix, tau for each combination of detectors
	tof_map[:,0] = pixarray[:,0]
	for ind, (name1, name2) in enumerate(network.get_tof_names()):
		for ipix, theta, phi in pixarray:
			tof_map[ipix,1+ind] = network.get_tof(name1,name2,theta,phi)
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	### build error maps
	if opts.verbose:
		print "computing err_map"
		if opts.time: to=time.time()
	if opts.e_approx == "gaussian":
		err_map = np.zeros((npix,1+ntof)) #ipix, tau for each combination of detectors
		err_map[:,0] = pixarray[:,0]
		for ind, (name1, name2) in enumerate(network.get_tof_names()):
			for ipix, theta, phi in pixarray:
				err_map[ipix,1+ind] = network.get_err(name1,name2,theta,phi)

	elif opts.e_approx == "singlekde":
		kdes = []
		for ind, (name1, name2) in enumerate(network.get_tof_names()):
			kdes.append( network.get_err(name1,name2,theta,phi) )
		kde = IndependentKDE(kdes) # single kde object for the entire sky

	elif opts.e_approx == "kde_map":
	 	err_map = [ [ipix]+[1.0 for _ in range(ntof)] for ipix in pixarray[:,0] ] #ipix, tau for each combination of detectors
		for ind, (name1, name2) in enumerate(network.get_tof_names()):
			for ipix, theta, phi in pixarray:
				err_map
		raise ValueError, "write kde_map"

	else:
		raise ValueError, "error-approx=%s not understood"%opts.e_approx

	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"
	
	### build sky map of antenna patterns
	if opts.verbose: 
		print "computing prior_map :",opts.prior
		if opts.time: to=time.time()
	ap_map = np.zeros((npix,2)) # ipix, |F|
	for ipix, theta, phi in pixarray:
		a = network.A(theta, phi, 0.0, no_psd=True) # get sensitivity matrix with psi set to 0.0 for convenience. Also, do not include time shifts or psd in antenna patterns
		a00 = a[0,0]
		a11 = a[1,1]
		a01 = a[0,1]
		a10 = a[1,0]
		F = 0.5*(a00+a11+((a00-a11)**2 + 4*a01*a10)**0.5) # maximum eigenvalue of the sensitivity matrix
	        ap_map[ipix] = np.array([ipix,F])
	prior_map = prior(ap_map, which=opts.prior)
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	### load in list of arrival times
	if opts.verbose: 
		print "loading a_cache from", opts.a_cache
		if opts.time: to=time.time()
	toa_file = open(opts.a_cache, "r")
	a_cache = pickle.load(toa_file)
	toa_file.close()

	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	### compute posteriors for each event in a_cache
	if opts.verbose: print "computing posteriors"
	n_toa = len(a_cache)
	if opts.scatter:
		estangs = np.empty((n_toa,2),np.float)
		injangs = np.empty((n_toa,2),np.float)

	for toa_ind, toa_event in enumerate(a_cache):
		inj_theta = toa_event['theta_inj'] # deg
		inj_phi = toa_event['phi_inj'] # deg
		if opts.deg:
			inj_theta *= utils.deg2rad
			inj_phi   *= utils.deg2rad

		if opts.verbose: 
			print "%d / %d\ntoa ="%(toa_ind+1,n_toa), toa_event
			print "\tcomputing posterior"
			if opts.time: to=time.time()

		### build observed time-of-flight vector
#		toa = np.array([toa_event[name1+"_inj"]-toa_event[name2+"_inj"] for name1,name2 in network.get_tof_names()])
		toa = np.array([toa_event[name1]-toa_event[name2] for name1,name2 in network.get_tof_names()])

		### build posteriors for each point in the sky
		posterior = np.zeros((npix,2)) # ipix, p(ipix|d)
		posterior[:,0] = pixarray[:,0]
		if opts.e_approx == "gaussian":
			posterior[:,1] = gaussian_likelihood(toa, tof_map, err_map) * prior_map
		elif opts.e_approx == "singlekde":
			posterior[:,1] = singlekde_likelihood(toa, tof_map, kde) * prior_map
		elif opts.e_approx == "kde_map":
			posterior[:,1] = kde_likelihood(toa, tof_map, err_map) * prior_map
		else:
			raise ValueError, "error-approx=%s not understood"%opts.e_approx
		
		### normalize posterior
		posterior[:,1] /= sum(posterior[:,1])

                if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

		### find the posterior's mode
		if opts.plot_posteriors or opts.stats or opts.scatter:
			estpix = int(posterior[:,1].argmax())
			est_theta, est_phi = hp.pix2ang(nside, estpix)

		### record positions for scatter
		if opts.scatter:
			estangs[toa_ind] = np.array([est_theta, est_phi])
			injangs[toa_ind] = np.array([inj_theta, inj_phi])

		### plot posteriors
		if opts.plot_posteriors:
			if opts.verbose:
				print "\tplotting posterior"
				if opts.time: to=time.time()
#                        figname = "%s/posterior-%d%sINJ.png"%(opts.output_dir, toa_ind, opts.tag)
                        figname = "%s/posterior-%d%s.png"%(opts.output_dir, toa_ind, opts.tag)
			title = "$t_{LHO}-t_{LLO}=%.3f\,\mathrm{ms}$"%(toa[0]*1e3)
			unit = "probability per steradian"

			fig = plt.figure(toa_ind)
			hp.mollview(posterior[:,1]/pixarea, fig=toa_ind, title=title, unit=unit, flip="geo", min=0.0)
			ax = fig.gca()
			hp.graticule()
			est_marker = ax.projplot((est_theta, est_phi), "wo", markeredgecolor="w", markerfacecolor="none")[0]
			est_marker.set_markersize(10)
			est_marker.set_markeredgewidth(2)
			inj_marker = ax.projplot((inj_theta, inj_phi), "wx")[0]
			inj_marker.set_markersize(10)
			inj_marker.set_markeredgewidth(2)
			if opts.verbose: 
				print "\t\t", figname
			fig.savefig(figname)
			plt.close(fig)
			if opts.verbose and opts.time: print "\t\t", time.time()-to, "sec"

		### write posteriors into FITs format
		if opts.write_posteriors:
			#print "WARNING: posteriors currently written to *.npy files, not FITs format"
			if opts.verbose:
				print "\twriting posterior"
				if opts.time: to=time.time()
			filename = "%s/posterior-%d%s.npy"%(opts.output_dir, toa_ind, opts.tag)
			if opts.verbose:
				print "\t\t", filename
			np.save(filename, posterior)
			if opts.verbose and opts.time: print "\t\t", time.time()-to, "sec"

#			raise StandardError, "write code that plots posteriors and saves them into FITs format!"

		### compute basic statistics about the reconstruction
		if opts.stats:
			if opts.verbose: 
				print "\tcomputing statistics"
				if opts.time: to=time.time()
			statsfilename = "%s/stats-%d%s.txt"%(opts.output_dir, toa_ind, opts.tag)
		
			### angular offset between max of the posterior and injection
			cosDtheta = np.cos(est_theta)*np.cos(inj_theta) + np.sin(est_theta)*np.sin(inj_theta)*np.cos(est_phi - inj_phi)
			### searched area
			injpix = hp.ang2pix(nside, inj_theta, inj_phi)
			posterior = posterior[posterior[:,1].argsort()[::-1]] # sort by posterior weight
			n_sapix = 0
			cum = 0.0
			for ipix, p in posterior:
				n_sapix += 1
				cum += p
				if ipix == injpix:
					break
			else:
				raise ValueError, "could not find injpix=%d in posterior"%injpix
			searched_area = pixarea_deg*n_sapix

			statsfile = open(statsfilename, "w")
			print >> statsfile, "cos(ang_offset) = %.6f\nsearched_area = %.6f deg2\np_value = %.6f"%(cosDtheta, searched_area, cum)
			statsfile.close()
			if opts.verbose: 
				print "\t\t", statsfilename
				print "\t\tcos(ang_offset) = %.6f\n\t\tsearched_area = %.6f deg2"%(cosDtheta, searched_area)
				if opts.time: print "\t\t", time.time()-to, "sec"

	### generate scatter plot
	if opts.scatter:
		if opts.verbose: 
			print "generating population scatter plots"
			if opts.time:
				to = time.time()
		fig = plt.figure(n_toa)
		hp.mollview(prior_map/(sum(prior_map)*pixarea), fig=n_toa, title="prior", unit="probability per steradian", flip="geo", min=0.0)
                hp.graticule()
		ax = fig.gca()

		est_marker = ax.projplot(estangs[:,0], estangs[:,1], "wo", markerfacecolor="none", markeredgecolor="w")[0]
		est_marker.set_markersize(1)
		est_marker.set_markeredgewidth(1)

		inj_marker = ax.projplot(injangs[:,0], injangs[:,1], "wx")[0]
		inj_marker.set_markersize(1)
		inj_marker.set_markeredgewidth(1)

		figname = "%s/populations%s.png"%(opts.output_dir,opts.tag)
		if opts.verbose: 
			print "saving %s"%figname
		fig.savefig(figname)
		plt.close(fig)
		if opts.time and opts.verbose:
			print time.time()-to, "sec"


