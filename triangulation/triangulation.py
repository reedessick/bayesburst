### written by R.Essick (ressick@mit.edu)

usage = """triangulation.py [--options] detector_name1 detector_name2 detector_name3 ...
written to compute posteriors over the sky using triangulation and modulation with antenna patterns"""

import sys
sys.path.append("../")

import utils
np = utils.np
pickle = utils.pickle
import triang
time = triang.time
plt = triang.plt
import stats
import detector_cache as det_cache

import healpy as hp

#===================================================================================================
#
#                                               MAIN
#
#===================================================================================================
if __name__ == "__main__":
	#================================================
	# parse options, arguments
	#================================================

	from optparse import OptionParser
	parser = OptionParser(usage=usage)

	parser.add_option("-v", "--verbose", dest="verbose", default=False, action="store_true")
	parser.add_option("", "--time", default=False, action="store_true")

	parser.add_option("-a", "--arrivals-cache", dest="a_cache", default=False, type="string", help="a cache file containing the times-of-arrival for the detectors in this network")
	parser.add_option("-A", "--new-arrivals-cache", default=False, type="string")

	parser.add_option("", "--deg", default=False, action="store_true", help="if True, we convert injected phi,theta to radians. only important for --arrivals-cache (for which posteriors are calculated).")

	parser.add_option("-N", "--timingNetwork", default=False, type="string", help="the filename of a pickled TimingNetwork. if --errors-cache is also supplied, the constructed TimingNetwork will be written into --timingNetwork.")

	parser.add_option("-e", "--errors-cache", dest="e_cache", default=False, type="string", help="a cache file containing errors in time-of-flight measurements between detectors")
	parser.add_option("", "--error-approx", dest="e_approx", default="gaussian", type="string", help="how the triangulation code estimates time-of-flight errors")
	parser.add_option("", "--diag", dest="diag", default=False, action="store_true", help="attempt to diagonalize the covariance matrix")

	parser.add_option("", "--hist-errors", default=False, action="store_true", help="histogram the tof errors observed in --errors-cache")
	parser.add_option("", "--scatter-errors", default=False, action="store_true", help="generate scatter plots and projected histograms for tof errors observed in --errors-cache")

	parser.add_option("", "--snr-bin-edge", default=[], action="append", type="float", help="bin edge for snr bins")
	parser.add_option("", "--freq-bin-edge", default=[], action="append", type="float", help="bin edge for freq bins")
	parser.add_option("", "--bandwidth-bin-edge", default=[], action="append", type="float", help="bin edge for bandwidth bins")

	parser.add_option("", "--default-snr", default=10, type="float", help="default to this snr value if toa event does not contain the key \"snr\"")
	parser.add_option("", "--default-freq", default=100, type="float", help="default to this freq value if toa event does not contain the key \"freq\"")
	parser.add_option("", "--default-bandwidth", default=10, type="float", help="default to this bandwidth value if toa event does not contain the key \"bandwidth\"")

	parser.add_option("", "--snr-key", default="snr", type="string")
	parser.add_option("", "--freq-key", default="freq", type="string")
	parser.add_option("", "--bandwidth-key", default="bandwidth", type="string")

	parser.add_option("-n", "--nside-exp", default=7, type="int", help="HEALPix NSIDE parameter for pixelization is 2**opts.nside_exp")

	parser.add_option("", "--prior", default="shells", type="string", help="flat, shells, spheres, or a float")

	parser.add_option("-w", "--write-posteriors", default=False, action="store_true", help="generate FITs files for posteriors")
	parser.add_option("-p", "--plot-posteriors", default=False, action="store_true", help="generate plots for posteriors")
	parser.add_option("-s", "--stats", default=False, action="store_true", help="compute basic statistics about the reconstruction.")
	parser.add_option("", "--scatter", default=False, action="store_true", help="generate a scatter plot of the entire population")

	parser.add_option("-o", "--output-dir", default="./", type="string")
	parser.add_option("-t", "--tag", default="", type="string")




	opts, args = parser.parse_args()

	if (not opts.timingNetwork) and (not len(args)) :
		raise ValueError, "supply at least 2 detector names as arguments or supply an existing TimingNetwork with --timingNetwork"

	args = sorted(args)

	nside = 2**opts.nside_exp

	if opts.tag:
		opts.tag = "_%s"%opts.tag

	no_posteriors = (not opts.a_cache)

	### binning
	opts.snr_bin_edge = sorted(opts.snr_bin_edge)
	opts.freq_bin_edge = sorted(opts.freq_bin_edge)
	opts.bandwidth_bin_edge = sorted(opts.bandwidth_bin_edge)

	### prior
	if opts.prior == "flat":
		prior_exp = 0.0
	elif opts.prior == "shells":
		prior_exp = 2.0
	elif opts.prior == "spheres":
		prior_exp = 3.0
	else:
		prior_exp = float(opts.prior)

	#==================================================================
	#
	# load detectors and instantiate network
	#
	#==================================================================
	### build the TimingNetwork from scratch
	if opts.e_cache:
		if opts.verbose: 
			print "loading list of detectors"
			if opts.time: to=time.time()
		detectors = {}
		for arg in args:
			name = arg.split("-")[0]
			if arg == "H1-design": #"LHO":
				detectors[name] = det_cache.LHO
			elif arg == "L1-design": #"LLO":
				detectors[name] = det_cache.LLO
			elif arg == "V1-design": #"Virgo":
				detectors[name] = det_cache.Virgo
			else:
				raise ValueError, "detector=%s not understood"%arg
		if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

		ndetectors = len(detectors) # number of detectors
		if not ndetectors:
			raise ValueError, "supply at least 2 detector names as arguments"

		### instantiate an empty TimingNetwork to use some functions
		network = triang.BinnedTimingNetwork(opts.e_approx, detectors=detectors.values(), snr_bin_edges=opts.snr_bin_edge, freq_bin_edges=opts.freq_bin_edge, bandwidth_bin_edges=opts.bandwidth_bin_edge)

		#================================================
		# compute error distributions relevant for this network
		#================================================
		if opts.verbose:
			print "computing error distributions from", opts.e_cache
			if opts.time: to = time.time()
		### load errors and build estimation functions
		e_cache = utils.load_toacache(opts.e_cache)
		network.toacache_to_errs(e_cache, verbose=opts.verbose, timing=opts.time, hist_errors=opts.hist_errors, scatter_errors=opts.scatter_errors, output_dir=opts.output_dir, tag=opts.tag, diag=opts.diag, dt=1e-5, snr_key=opts.snr_key, freq_key=opts.freq_key, bandwidth_key=opts.bandwidth_key)	
		if opts.verbose:
			print "built TimingNetwork\n\t", network
			if opts.time: print "\t", time.time()-to, "sec"

                #===============================================
                # write timing network to file
                #===============================================
                if opts.timingNetwork:
                        if opts.verbose:
                                print "writing TimingNetwork to:", opts.timingNetwork
                                if opts.time: to=time.time()
                        file_obj = open(opts.timingNetwork,'w')
                        pickle.dump( network, file_obj )
                        file_obj.close()
                        if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

        ### load the TimingNetwork
        elif opts.timingNetwork:
                #===============================================
                # load the timing network from pkl file
                #===============================================
                if opts.verbose:
                        print "reading TimingNetwork from", opts.timingNetwork
                        if opts.time: to=time.time()
                file_obj = open(opts.timingNetwork,'r')
                network = pickle.load(file_obj)
                file_obj.close()
                if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

        ### don't know how to build TimingNetwork
        else:
                raise ValueError, "please specify an error approximation with --error-approx or an exising TimingNetwork with --timingNetwork"

	#================================================
	# only continue if --arrivals-cache is supplied
	#================================================
	if opts.a_cache:
		### the ordered names for pairs of detectors
		tof_names = network.get_tof_names()

		#================================================
		# define sky pixelization
		#================================================
		pixarea = hp.nside2pixarea(nside)
		pixarea_deg = pixarea/utils.deg2rad**2
		npix = hp.nside2npix(nside)
	        if opts.verbose: 
			print "pixelating the sky with:\n\tnside=%d\n\tnpix=%d"%(nside,npix)
			if opts.time: to = time.time()
		pixarray = triang.__pixarray(npix, nside=nside)
		if opts.verbose:
			if opts.time: print "\t", time.time()-to, "sec"

		#===============================================
		# build Likelihood object
		#===============================================
		if opts.verbose:
			print "building Likelihood"
			if opts.time: to=time.time()
		likelihood = triang.Likelihood(network, npix, pixarray=pixarray)
		if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

		#===============================================
		# build Prior object
		#===============================================
		if opts.verbose: 
			print "building Prior", opts.prior
			if opts.time: to=time.time()
		prior = triang.Prior(network, npix, pixarray=pixarray, freq=opts.default_freq, exp=prior_exp)
		if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

		#=========================================================================
		#
		# load in list of arrival times and compute posteriors
		#
		#=========================================================================
		if opts.verbose: 
			print "loading a_cache from", opts.a_cache
			if opts.time: to=time.time()
		a_cache = utils.load_toacache(opts.a_cache)
		if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

		#================================================
		# compute posteriors
		#================================================
		if opts.verbose: print "computing posteriors"
		n_toa = len(a_cache)

		### set up holders for positions
		if opts.scatter:
			estangs = np.empty((n_toa,2),np.float)
			injangs = np.empty((n_toa,2),np.float)
			prior_map = np.zeros((npix,),np.float)

		### define title_template with tof_names
		if opts.plot_posteriors:
			title_template = ""
			for name1, name2 in tof_names:
				title_template += "$t_{%s}-t_{%s}="%(name1,name2)+"%.3f\mathrm{ms}$\n"
#			title_template = title_template[:-1]
			title_template += "snr=%.1f freq=%.1f bndwth=%.1f"

		### loop over events
		for toa_ind, toa_event in enumerate(a_cache):
        	        if opts.verbose:
                	        print "%d / %d\ntoa ="%(toa_ind+1,n_toa), toa_event
                        	print "\tcomputing posterior"
	                        if opts.time: to=time.time()

			if opts.scatter or opts.plot_posteriors or opts.stats:
				inj_theta = toa_event['theta_inj']
				inj_phi = toa_event['phi_inj']
				if opts.deg:
					inj_theta *= utils.deg2rad
					inj_phi   *= utils.deg2rad

			#======================
			# build observed time-of-flight vector
			#======================
			toa = np.array([toa_event[name1]-toa_event[name2] for name1,name2 in tof_names])
				
			if opts.verbose:
				print "tof (detectors):\t", toa

			#======================
			# signal parameters
			#======================
			if toa_event.has_key(opts.snr_key):
				snr = toa_event[opts.snr_key]
			else:
				if opts.verbose: print "\t\tusing default_snr"
				snr = opts.default_snr

			if toa_event.has_key(opts.freq_key):
				freq = toa_event[opts.freq_key]
			else:
				if opts.verbose: print "\t\tusing default_freq"
				freq = opts.default_freq

			if toa_event.has_key(opts.bandwidth_key):
				bandwidth = toa_event[opts.bandwidth_key]
			else:
				if opts.verbose: print "\t\tusing default_bandwidth"
				bandwidth = opts.default_bandwidth

			if opts.verbose:
				print "snr       : %.3f"%snr
				print "freq      : %.3f"%freq
				print "bandwidth : %.3f"%bandwidth

			#======================
			# build posteriors for each point in the sky
			#======================
			_likelihood = likelihood(toa, snr, freq, bandwidth, verbose=opts.verbose) ### compute likelihood from timing
			_prior = prior(freq, exp=prior_exp, verbose=opts.verbose) ### compute prior from antenna patterns

			posterior = _likelihood * _prior
	
			### normalize posterior
			posterior /= sum(posterior)

	                if opts.verbose and opts.time: print "\t", time.time()-to, "sec"

			if opts.scatter: ### recored the average prior for the scatter plot
				prior_map += _prior

			#======================
			# save, plot, and summarize posterior
			#======================
			### find the posterior's mode and pull out injected location
			if opts.plot_posteriors or opts.stats or opts.scatter:
				### estimated pixel
				est_theta, est_phi = stats.estang(posterior, nside=nside)

				### injected pixel
                        	inj_theta = toa_event['theta_inj']
	                        inj_phi = toa_event['phi_inj']
        	                if opts.deg:
                	                inj_theta *= utils.deg2rad
                        	        inj_phi   *= utils.deg2rad

				### record positions for scatter
				if opts.scatter:
					estangs[toa_ind] = np.array([est_theta, est_phi])
					injangs[toa_ind] = np.array([inj_theta, inj_phi])

	                ### write posteriors
        	        if opts.write_posteriors:
                	        if opts.verbose:
                        	        print "\twriting posterior"
                                	if opts.time: to=time.time()
	                        filename = "%s/posterior-%d%s.fits"%(opts.output_dir, toa_ind, opts.tag)
        	                if opts.verbose:
                	                print "\t\t", filename
                        	hp.write_map(filename, posterior)
	                        if opts.verbose and opts.time: print "\t\t", time.time()-to, "sec"

				toa_event['fits'] = filename
	
			### plot posteriors
			if opts.plot_posteriors:
				if opts.verbose:
					print "\tplotting posterior"
					if opts.time: to=time.time()
        	                figname = "%s/posterior-%d%s.png"%(opts.output_dir, toa_ind, opts.tag)
				title = title_template%tuple(list(toa*1e3)+[snr,freq,bandwidth])
				unit = "probability per steradian"

				fig_ind = toa_ind+100
				fig = plt.figure(fig_ind)
				hp.mollview(posterior/pixarea, fig=fig_ind, title=title, unit=unit, flip="geo", min=0.0)
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

			### compute basic statistics about the reconstruction
			if opts.stats:
				if opts.verbose: 
					print "\tcomputing statistics"
					if opts.time: to=time.time()
				statsfilename = "%s/stats-%d%s.txt"%(opts.output_dir, toa_ind, opts.tag)
			
				### angular offset between max of the posterior and injection
				cosDtheta = stats.cos_dtheta(est_theta, est_phi, inj_theta, inj_phi)

				### searched area
				p_value = stats.p_value(posterior, inj_theta, inj_phi, nside=nside)
				searched_area = stats.searched_area(posterior, inj_theta, inj_phi, degrees=True)

				### num_mode
				num_mode = stats.num_modes(posterior, inj_theta, inj_phi, nside=nside)

				### min{cosDtheta}
				min_cosDtheta = stats.min_cos_dtheta(posterior, inj_theta, inj_phi, nside=nside)

				### entropy
				entropy = stats.entropy(posterior)
		
				### information
				info = stats.information(posterior)

				statsfile = open(statsfilename, "w")
				print >> statsfile, "cos(ang_offset) = %.6f\nsearched_area = %.6f deg2\np_value = %.6f\nnum_mode = %d\nmin{cos(ang_offset)} = %.6f\nentropy = %.6f\ninformation = %.6f"%(cosDtheta, searched_area, p_value, num_mode, min_cosDtheta, entropy, info)
				statsfile.close()

				if opts.verbose: 
					print "\t\t", statsfilename
					print "\t\tcos(ang_offset) = %.6f\n\t\tsearched_area = %.6f deg2\np_value = %.6f\nnum_mode = %d\nmin{cos(ang_offset)} = %.6f\nentropy = %.6f\ninformation = %.6f"%(cosDtheta, searched_area, p_value, num_mode, min_cosDtheta, entropy, info)
					if opts.time: print "\t\t", time.time()-to, "sec"

		#===============================================
		# generate scatter plot
		#===============================================
		if opts.scatter:
			if opts.verbose: 
				print "generating population scatter plots"
				if opts.time:
					to = time.time()
			prior_map /= sum(prior_map)

			fig = plt.figure(n_toa)
			hp.mollview(prior_map/pixarea, fig=n_toa, title="average prior", unit="probability per steradian", flip="geo", min=0.0)
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

		#===============================================
		# recore new arrivals cache
		#===============================================
		if opts.new_arrivals_cache:
			if opts.verbose:
				print "writing data to", opts.new_arrivals_cache
				file_obj = open(opts.new_arrivals_cache, "w")
				pickle.dump(a_cache, file_obj)
				file_obj.close()

