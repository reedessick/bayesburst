usage = """triangulation.py [--options] detector_name1 detector_name2 detector_name3 ...
written to compute posteriors over the sky using triangulation and modulation with antenna patterns"""

import utils
np = utils.np
import healpy as hp

#=================================================
#
#                  utilities
#
#=================================================
def likelihood(toa, tof_map, err_map):
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

	return (2*np.pi*err_map[:,1:]**2)**0.5 * np.exp( -(toa-tof_map[:,1:])**2/(2*err_map[:,1:]**2) )

def prior(ap_map):
	"""
	computes the prior assigned to each point using the antenna patterns stored in ap_map
	"""
	return ap_map[:,-1] # shells in the sky


class TimingNetwork(utils.Network):
	"""
	an extension of utils.network that will include time-of-flight errors between detectors
	we assume the timing errors are gaussian so we can describe them with a single number (the standard deviation)
	"""
	errors = dict()

	def __init__(self, errs={}):
		for (detector1,detector2), err in errs.items():
			name1 = detector1.name
			name2 = detector2.name
			self.detectors[name1] = detector1
			self.detectors[name2] = detector2
			names = tuple(sorted([name1,name2]))
			self.errors[names] = err
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
		dr = self.detectors[name2].dr - self.detectors[name1].dr
		return utils.time_of_flight(theta, phi, self.detectors[name2].dr - self.detectors[name1].dr)

	def get_tof_names(self):
		"""returns an ordered list of detector combinations"""
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
	parser.add_option("-e", "--errors-cache", dest="e_cache", default="errors.cache", type="string", help="a cache file containing errors in time-of-flight measurements between detectors")

	parser.add_option("-n", "--nside", default=128, type="int", help="HEALPix NSIDE parameter for pixelization")

	parser.add_option("-w", "--write-posteriors", default=False, action="store_true", help="generate FITs files for posteriors")
	parser.add_option("-p", "--plot-posteriors", default=False, action="store_true", help="generate plots for posteriors")

	parser.add_option("-o", "--output-dir", default="./", type="string")
	parser.add_option("-t", "--tag", default="", type="string")

	parser.add_option("", "--time", default=False, action="store_true")

	opts, args = parser.parse_args()
	args = sorted(args)
	if opts.tag:
		opts.tag = "_%s"%opts.tag

	if opts.plot_posteriors:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

	if opts.time:
		import time

	#=========================================================================
	### for starters, we only consider LHO-LLO networks, so let's demand that from our input arguments
	if args != sorted(["LHO", "LLO"]):
		raise ValueError, "we only consider LHO and LLO networks for now, so please supply exactly those two detector names"
	#=========================================================================

	### get list of detectors from arguments
	if opts.verbose: 
		print "loading list of detectors"
		if opts.time: to=time.time()
	detectors = []
	for arg in args:
		if arg == "LHO":
			detectors.append( utils.LHO )
		elif arg == "LLO":
			detectors.append( utils.LLO )
		elif arg == "Virgo":
			detectors.append( utils.Virgo )
		else:
			raise ValueError, "detector=%s not understood"%arg
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	### get set of timing errors from error_cache
	### we should select only those errors which are relevant to this network
#=================================================================================================================
#
#
	### for testing purposes, we take a short cut and don't implement this right away
	if opts.verbose:
		print "loading error distributions"
		if opts.time: to = time.time()
	errs = {(utils.LHO, utils.LLO):0.0005}
#	errs = {(utils.LHO, utils.LLO):0.0030}
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"
#
#
#=================================================================================================================

	### build the network
	if opts.verbose:
		print "building TimingNetwork"
		if opts.time: to=time.time()
	network = TimingNetwork(errs)
	for (detector1,detector2),err in errs.items():
		network.set_err(detector1.name, detector2.name, err)

	ndetectors = len(network)
	ntof = ndetectors*(ndetectors-1)/2 # number of combinations of detectors

	if opts.verbose: 
		print "\t", network
		if opts.time: print "\t", time.time()-to, "sec"

	### number of pixels in the sky map
	npix = hp.nside2npix(opts.nside)
        if opts.verbose: 
		print "pixelating the sky with nside=%d ==> %d pixels"%(opts.nside,npix)
		if opts.time: to = time.time()
	pixarray = np.zeros((npix,3))
	for ipix in np.arange(npix):
		theta, phi = hp.pix2ang(opts.nside, ipix)
		pixarray[ipix,:] = np.array([ipix, theta, phi])
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

	### build sky map of expected time-of-flights and expected errors
	if opts.verbose: 
		print "computing tof_map and err_map"
		if opts.time: to=time.time()
	tof_map = np.zeros((npix,1+ntof)) # ipix, tau for each combination of detectors
	tof_map[:,0] = pixarray[:,0]
	err_map = np.zeros((npix,1+ntof)) # ipix, tau for each combination of detectors
	err_map[:,0] = pixarray[:,0]

	for ipix, theta, phi in pixarray:
		for ind, (name1,name2) in enumerate(network.get_tof_names()):
                        tof_map[ipix,1+ind] = network.get_tof(name1,name2,theta,phi)
			err_map[ipix,1+ind] = network.get_err(name1,name2,theta,phi)
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"
	
	### build sky map of antenna patterns
	if opts.verbose: 
		print "computing ap_map"
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
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

#===========================================================================================================
#
#
	### load in list of arrival times
	if opts.verbose: 
		print "loading a_cache"
		if opts.time: to=time.time()
	a_cache = []
	for dt in np.arange(-0.0195, +0.0200, 0.0005):
		a_cache.append( {"LLO":dt, "LHO":0.0000} ) # needs to match ndetectors
	if opts.verbose and opts.time: print "\t", time.time()-to, "sec"
#
#
#===========================================================================================================

	### compute posteriors for each event in a_cache
	if opts.verbose: print "computing posteriors"
	for toa_ind, toa in enumerate(a_cache):
		if opts.verbose: 
			print "toa =", toa
			if opts.time: to=time.time()

		### build observed time-of-flight vector
		toa = np.array([toa[name2]-toa[name1] for name1,name2 in network.get_tof_names()])

		### build posteriors for each point in the sky
		posterior = np.zeros((npix,2)) # ipix, p(ipix|d)
		posterior[:,0] = pixarray[:,0]
		posterior[:,1] = likelihood(toa, tof_map, err_map).flatten() * prior(ap_map) # flatten() is to make the shapes compatible
		
		### normalize posterior
		posterior[:,1] /= sum(posterior[:,1])

		### plot posteriors
		if opts.plot_posteriors:
                        figname = "posterior-%d.png"%toa_ind
			title = "$t_{LHO}-t_{LLO}=%.4f\,\mathrm{sec}$"%(toa[0])
			unit = "probability per steradian"

			fig = plt.figure(toa_ind)
			view = hp.mollview(posterior[:,1], fig=toa_ind, title=title, unit=unit)
			if opts.verbose: 
				print "\tsaving", figname
			fig.savefig(figname)
			plt.close(fig)

		### write posteriors into FITs format
		if opts.write_posteriors:
			raise StandardError, "write code that plots posteriors and saves them into FITs format!"

		if opts.verbose and opts.time: print "\t", time.time()-to, "sec"
