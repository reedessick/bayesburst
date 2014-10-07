#!/usr/bin/python

usage = """pipeline.py [--options] 
a script meant to run the engine contained in posteriors.py and priors.py"""

import ConfigParser
import os
import numpy as np
import pickle

import utils
import detector_cache
import priors
import posteriors
import model_selection
import visualizations as viz

from optparse import OptionParser


print """write plotting for opts.diagnostics. 
Validate that the noise and data are on the correct scales and look reasonable.
Validate that we compute a reasonable posterior and that the bayes factors are reasonable
compute snrs using injected data

once dpf functionality is robustly tested, implement those options here

write a dag-builder that writes jobs for these scripts
proceed to process large amounts of data via cluster parallelization
"""

#=================================================
parser=OptionParser(usage=usage)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-c", "--config", default="./config.ini", type="string", help="config file")

parser.add_option("-g", "--gps", default=0, type="float", help="central time of this analysis")

parser.add_option("-x", "--xmlfilename", default=False, type="float", help="an injection xml file")
parser.add_option("-i", "--sim-id", default=False, type="int", help="the injection id in the xml file")

parser.add_option("-d", "--diagnostic", default=False, action="store_true", help="output some diagnostic data to check the pipeline's functionality")
parser.add_option("-p", "--diagnostic-plots", default=False, action="store_true", help="generat some diagnostic plots along with diagonstic output")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

parser.add_option("", "--time", default=False, action="store_true")

opts, args = parser.parse_args()

if opts.tag:
	opts.tag = "_" + opts.tag

if opts.diagnostic_plots:
	opts.diagnostic = True

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

if opts.time:
	opts.verbose=True
	import time

#=================================================
### load config file
#=================================================
if opts.verbose: 
	print "\n----------------------------------\n"
	print "loading config from ", opts.config
	if opts.time:
		to = time.time()

config = ConfigParser.SafeConfigParser()
config.read(opts.config)

max_proc=config.getint("general", "max_proc")
max_array_size=config.getint("general", "max_array_size")

n_pol=config.getint("general", "num_pol")

if opts.time:
	print "\t", time.time()-to

#=================================================
### fft parameters
#=================================================
if opts.verbose: 
	print "\n----------------------------------\n"
	print "setting up fft paramters"
	if opts.time:
		to = time.time()

seglen = config.getfloat("fft","seglen")
df = 1.0/seglen
fs = config.getfloat("fft","fs")
freqs=np.arange(0, fs/2, df)
n_freqs = len(freqs)

flow = config.getfloat("fft","flow") ### currently only used in plotting and model selection
                                     ### any speed-up associated with skipping these bins in 
                                     ### log posterior elements is currently unavailable

freq_truth = freqs >= flow

if opts.time:
	print "\t", time.time()-to

#=================================================
### build network
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "building network"
	if opts.time:
		to = time.time()

ifos = eval(config.get("network","ifos"))
n_ifo = len(ifos)
network = utils.Network([detector_cache.detectors[ifo] for ifo in ifos], freqs=freqs, Np=n_pol)

if opts.time:
        print "\t", time.time()-to

#=================================================
### estimate PSD and load it into the network 
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "generating PSD's"
	if opts.time:
		to = time.time()

if config.has_option("psd_estimation","cache"):
	psd_cache = eval(config.get("psd_estimation", "cache"))
	for ifo in ifos:
		psd_filename = psd_cache[ifo]
		if opts.verbose:
			print "\treading PSD for %s from %s"%(ifo, psd_filename)
		psd_freqs, asd = np.transpose(np.loadtxt(psd_filename)) ### read in the psd from file
		network.detectors[ifo].set_psd(asd**2, freqs=psd_freqs) ### update the detector

elif config.has_option("noise","cache"): ### estimate PSD's from noise frames
	noise_cache = eval(config.get("noise","cache"))
	noise_chans = eval(config.get("noise","channels"))

	### pull out parameters for PSD estimation
	fs = config.getfloat("psd_estimation","fs")
	seglen= config.getfloat("psd_estimation","seglen")
	overlap = config.getfloat("psd_estimation","overlap")
	if overlap >= seglen:
		raise ValueError, "overlap must be less than seglen in psd_estimation"
	num_segs = config.getint("psd_estimation","num_segs")
	
	duration = (num_segs*seglen - (num_segs-1)*overlap) ### amount of time used for PSD estimation
	dseg = 0.5*config.getfloat("fft","seglen") ### buffer for fft used to compute noise, inj, etc

	from pylal import Fr
	for ifo in ifos:

		ifo_chan = noise_chans[ifo]
		ifo_cache = noise_cache[ifo]
		if opts.verbose: 
			print "reading %s frames from %s with channel=%s"%(ifo, ifo_cache, ifo_chan)
			if opts.time:
				to = time.time()
		### find relevant files
		frames = utils.files_from_cache(ifo_cache, (opts.gps-dseg)-duration, opts.gps-dseg)

		if opts.time:
			print "\t", time.time()-to

		### read vectors from files
		### FIXME: we assume data is continuous, which may not be the case!
		### we also assume that data is sampled at a constant rate
		vecs = []
		dt = 0
		for frame, start, dur in frames:
			if opts.verbose:
				print "\t", frame
				if opts.time:
					to = time.time()
			s = max(start,(opts.gps-dseg)-duration)
			d = min(start+dur,(opts.gps-dseg))-s
			vec, gpstart, offset, dt, _, _ = Fr.frgetvect1d(frame, ifo_chan, start=s, span=d)
			if opts.time:
				print "\t\t", time.time()-to
			vecs.append( vec )
		vec = np.concatenate(vecs)
		N = len(vec)

		### downsample data to fs?
		_dt = 1.0/fs
		if opts.verbose:
                	print "downsampling time-domain data : dt=%fe-6 -> dt=%fe-6"%(dt*1e6, _dt*1e6)
			if opts.time:
				to = time.time()
		vec, dt = utils.resample(vec, dt, _dt)
		if opts.time:
			print "\t", time.time()-to
			
		### check that we have all the data we expect
		if len(vec)*dt != duration:
			raise ValueError, "len(vec)*dt = %f != %f = duration"%(len(vec)*dt, duration)

		### estimate psd
		if opts.verbose:
			print "estimating PSD with %d segments"%num_segs
			if opts.time:
				to = time.time()
		psd, psd_freqs = utils.estimate_psd(vec, num_segs=num_segs, overlap=overlap/dt, dt=dt)
		if opts.time:
			print "\t", time.time()-to

		### update network object
		network.detectors[ifo].set_psd(psd, freqs=psd_freqs)

else:
	raise StandardError, "could not estimate PSDs. Please either supply a cache of ascii files or a cache of noise frames in the configuration file."

if opts.time:
	print "\t", time.time()-to

### dump psd's to numpy arrays if requested
if opts.diagnostic:
	for ifo in ifos:
		### save psd to file
		psd_filename = "%s/%s-PSD%s_%d.pkl"%(opts.output_dir, ifo, opts.tag, int(opts.gps))
		if opts.verbose:
			print "writing %s PSD to %s"%(ifo, psd_filename)
			if opts.time:
				to = time.time()
		psd_obj = network.detectors[ifo].get_psd()
		file_obj = open(psd_filename, "w")
		for f, p in zip(psd_obj.get_freqs(), psd_obj.get_psd()):
			print >> file_obj, f, p
		file_obj.close()
		if opts.time:
			print "\t", time.time()-to

		if opts.diagnostic_plots:
			### plot psd
			psd_figname = "%spng"%psd_filename[:-3]
			if opts.verbose:
				print "plotting %s PSD : %s"%(ifo, psd_figname)
				if opts.time:
					to = time.time()
			fig, ax = viz.ascii_psd(psd_filename, label=ifo)

			ax.set_xlabel("frequency [Hz]")
			ax.set_ylabel("PSD [1/Hz]")
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_xlim(xmin=flow, xmax=np.max(psd_obj.get_freqs()))
			ax.grid(True, which="both")

			fig.savefig(psd_figname)
			viz.plt.close(fig)
			if opts.time:
				print "\t", time.time()-to

#=================================================
### load noise
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "generating noise"

if config.has_option("noise","zero"):
        if opts.verbose:
                print "zeroing noise data"
        noise = np.zeros((n_freqs, n_ifo), complex)

elif config.has_option("noise","gaussian"):
        ### simulate gaussian noise
        if opts.verbose:
                print "drawing gaussian noise"
                if opts.time:
                        to = time.time()
        noise = network.draw_noise()
        if opts.time:
                print "\t", time.time()-to

elif config.has_option("noise", "cache"):
        noise = np.empty((n_freqs, n_ifo), complex)

        noise_cache = eval(config.get("noise","cache"))
        noise_chans = eval(config.get("noise","channels"))

        ### pull out parameters for PSD estimation
        fs = config.getfloat("fft","fs")
        seglen= config.getfloat("fft","seglen")
        dseg = 0.5*seglen

        from pylal import Fr
        for ifo_ind, ifo in enumerate(ifos):

                ifo_chan = noise_chans[ifo]
                ifo_cache = noise_cache[ifo]
                if opts.verbose:
                        print "reading %s frames from %s with channel=%s"%(ifo, ifo_cache, ifo_chan)
                        if opts.time:
                                to = time.time()
                ### find relevant files
                frames = utils.files_from_cache(ifo_cache, (opts.gps-dseg), (opts.gps+dseg))

                if opts.time:
                        print "\t", time.time()-to

                ### read vectors from files
                ### FIXME: we assume data is continuous, which may not be the case!
                ### we also assume that data is sampled at a constant rate
                vecs = []
                dt = 0
                for frame, start, dur in frames:
                        if opts.verbose:
                                print "\t", frame
                                if opts.time:
                                        to = time.time()
                        s = max(start, opts.gps-dseg)
                        d = min(start+dur, opts.gps+dseg)-s
                        vec, gpstart, _, dt, _, _ = Fr.frgetvect1d(frame, ifo_chan, start=s, span=d)
                        if opts.time:
                                print "\t\t", time.time()-to
                        vecs.append( vec )
                vec = np.concatenate(vecs)
                N = len(vec)

                ### downsample data to fs?
                _dt = 1.0/fs
                if opts.verbose:
                        print "downsampling time-domain data : dt=%fe-6 -> dt=%fe-6"%(dt*1e6, _dt*1e6)
			if opts.time:
				to = time.time()
                vec, dt = utils.resample(vec, dt, _dt)
                if opts.time:
                        print "\t", time.time()-to

                ### check that we have all the data we expect
                if len(vec)*dt != seglen:
                        raise ValueError, "len(vec)*dt = %f != %f = seglen"%(len(vec)*dt, seglen)

                ### store noise
                dft_vec, dft_freqs = utils.dft(vec, dt=dt)
                if np.any(dft_freqs != freqs):
                        raise ValueError, "frequencies from utils.dft do not agree with freqs defined by hand"
                noise[:,ifo_ind] = dft_vec * np.exp(2j*np.pi*opts.gps*freqs) ### add phase shift for start time

else:
        if opts.verbose:
                print "no noise generation specified. zero-ing noise"
        noise = np.zeros((n_freqs, n_ifo), complex)

### dump noise to file
if opts.diagnostic:
	### save noise to file
        noise_filename = "%s/noise%s_%d.pkl"%(opts.output_dir, opts.tag, int(opts.gps))
        if opts.verbose:
                print "writing noise to %s"%noise_filename
                if opts.time:
                        to = time.time()
        file_obj = open(noise_filename, "w")
        pickle.dump(noise, file_obj)
        file_obj.close()
        if opts.time:
                print "\t", time.time()-to

	if opts.diagnostic_plots:
		### plot noise
		noise_figname = "%spng"%noise_filename[:-3]
		if opts.verbose:
			print "plotting noise : %s"%noise_figname
			if opts.time:
				to = time.time()
		fig, axs = viz.data(freqs, noise, ifos, units="$1/\sqrt{\mathrm{Hz}}$")

		xmax = np.max(freqs)
		for ax in axs:
			ax.grid(True, which="both")
			ax.set_xlim(xmax=xmax)
			ax.legend(loc="best")
		ax.set_xlabel("frequency [Hz]")

		fig.savefig(noise_figname)
		viz.plt.close(fig)
		if opts.time:
			print "\t", time.time()-to

#=================================================
### load injection
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "generating injections"

if config.has_option("injection","zero"): ### inject zeros. In place for convenience
        if opts.verbose:
                print "zeroing injected data"
        inj = np.zeros((n_freqs, n_ifo), complex)

elif config.has_option("injection","dummy"): ### inject a dummy signal that's consistently placed
        import injections

        fo = 200
        t = opts.gps + seglen/2.0
        phio = 0.0
        tau = 0.010
        hrss = 6e-23
        alpha = np.pi/2

        theta = np.pi/4
        phi = 3*np.pi/4
        psi = 0.0
        if opts.verbose:
                print """injecting SineGaussian with parameters
        fo = %f
        to = %f
        phio = %f
        tau = %f
        hrss = %fe-23
        alpha = %f
        
        theta = %f
        phi = %f
        psi = %f"""%(fo, t, phio, tau, hrss*1e23, alpha, theta, phi, psi)

                if opts.time:
                        to = time.time()
        h = injections.sinegaussian_f(freqs, to=t, phio=phio, fo=fo, tau=tau, hrss=hrss, alpha=alpha)
        inj = injections.inject( network, h, theta, phi, psi)

        if opts.time:
                print "\t", time.time()-to

elif config.has_option("injection","cache"): ### read injection frame
        inj = np.empty((n_freqs, n_ifo), complex)

        inj_cache = eval(config.get("injection","cache"))
        inj_chans = eval(config.get("injection","channels"))

        ### pull out parameters for PSD estimation
        fs = config.getfloat("fft","fs")
        seglen= config.getfloat("fft","seglen")
        dseg = 0.5*seglen

        from pylal import Fr
        for ifo_ind, ifo in enumerate(ifos):

                ifo_chan = inj_chans[ifo]
                ifo_cache = inj_cache[ifo]
                if opts.verbose:
                        print "reading %s frames from %s with channel=%s"%(ifo, ifo_cache, ifo_chan)
                        if opts.time:
                                to = time.time()
                ### find relevant files
                frames = utils.files_from_cache(ifo_cache, opts.gps-dseg, opts.gps+dseg)

                if opts.time:
                        print "\t", time.time()-to

                ### read vectors from files
                ### FIXME: we assume data is continuous, which may not be the case!
                ### we also assume that data is sampled at a constant rate
                vecs = []
                dt = 0
                for frame, start, dur in frames:
                        if opts.verbose:
                                print "\t", frame
                                if opts.time:
                                        to = time.time()
                        s = max(start, opts.gps-dseg)
                        d = min(start+dur, opts.gps+dseg)-s
                        vec, gpstart, _, dt, _, _ = Fr.frgetvect1d(frame, ifo_chan, start=s, span=d)
                        if opts.time:
                                print "\t\t", time.time()-to
                        vecs.append( vec )
                vec = np.concatenate(vecs)
                N = len(vec)

                ### downsample data to fs?
                _dt = 1.0/fs
                if opts.verbose:
                        print "downsampling time-domain data : dt=%fe-6 -> dt=%fe-6"%(dt*1e6, _dt*1e6)
			if opts.time:
				to = time.time()
                vec, dt = utils.resample(vec, dt, _dt)
                if opts.time:
                        print "\t", time.time()-to

                ### check that we have all the data we expect
                if len(vec)*dt != seglen:
                        raise ValueError, "len(vec)*dt = %f != %f = seglen"%(len(vec)*dt, seglen)

                ### store injection
                dft_vec, dft_freqs = utils.dft(vec, dt=dt)
                if np.any(dft_freqs != freqs):
                        raise ValueError, "frequencies from utils.dft do not agree with freqs defined by hand"
                inj[:,ifo_ind] = dft_vec * np.exp(2j*np.pi*opts.gps*freqs) ### add phase shift for start time

elif opts.xmlfilename: ### read injection from xmlfile
        if opts.sim_id==None:
                raise StandardError, "must supply --event-id with --xmlfilename"

        if opts.verbose:
                print "reading simulation_id=%d from %s"%(opts.sim_id, opts.xmlfilename)
                if opts.time:
                        to = time.time()

        from glue.ligolw import lsctables
        from glue.ligolw import utils as ligolw_utils

        ### we specialize to sim_burst tables for now...
        table_name = "sim_burst"

        ### load xmlfile and find row corresponding to the specified entry
        xmldoc = utils.load_filename(opts.xmlfilename)
        tbl = lsctables.table.get_table(xmldoc, table_name)
        for row in tbl:
                if row.simulation_id == opts.sim_id:
                        break
        else:
                raise ValueError, "could not find sim_id=%d in %s"%(opts.sim_id, opts.xmlfilename)

        import injections

        if row.waveform == "Gaussian":
                t = row.time_geocent_gps + row.time_geocent_gps_ns*1e-9
                tau = row.duration
                hrss = row.hrss
                print "warning! only injecting alpha=np.pi/2 for now"
                alpha = np.pi/2

                ra = row.ra
                dec = row.dec
                psi = row.psi

                gmst = row.time_geocent_gmst
                theta = np.pi/2 - dec
                phi = (ra-gmst)%(2*np.pi)

                wavefunc = injections.gaussian_f
                waveargs = {"to":t, "tau":tau, "alpha":alpha, "hrss":hrss}

        elif row.waveform == "SineGaussian":
                t = row.time_geocent_gps + row.time_geocent_gps_ns*1e-9
                tau = row.duration
                fo = row.frequency
                hrss = row.hrss
                print "warning! only injecting alpha=np.pi/2 for now"
                alpha = np.pi/2
                print "warning! only injecting phio=0 for now"
                phio = 0

                ra = row.ra
                dec = row.dec
                psi = row.psi

                gmst = row.time_geocent_gmst
                theta = np.pi/2 - dec
                phi = (ra-gmst)%(2*np.pi)

                wavefunc = injections.sinegaussian_f
                waveargs = {"to":t, "phio":phio, "fo":fo, "tau":tau, "alpha":alpha, "hrss":hrss}

        else:
                raise ValueError, "row.waveform=%s not recognized"%row.waveform

        inj = injections.inject( network, wavefunc(freqs, **waveargs), theta, phi, psi=psi)

        if opts.time:
                print "\t", time.time()-to

else: ### no injection data specified
        if opts.verbose:
                print "no injection specified, so zeroing injection data"
        inj = np.zeros((n_freqs, n_ifo), complex)

if config.has_option("injection","factor"): ### mdc factor for injection
        factor = config.getfloat("injection","factor")
        if opts.verbose:
                print "applying mdc factor =", factor
        inj *= factor

### dump injection to file
if opts.diagnostic:
	### write injection to file
        inj_filename = "%s/injections%s_%d.pkl"%(opts.output_dir, opts.tag, int(opts.gps))
        if opts.verbose:
                print "writing injections to %s"%inj_filename
                if opts.time:
                        to = time.time()
        file_obj = open(inj_filename, "w")
        pickle.dump(inj, file_obj)
        file_obj.close()
        if opts.time:
                print "\t", time.time()-to

	if opts.diagnostic_plots:
	        ### plot injection
        	inj_figname = "%spng"%inj_filename[:-3]
	        if opts.verbose:
        	        print "plotting injections : %s"%inj_figname
                	if opts.time:
                        	to = time.time()
	        fig, axs = viz.data(freqs, inj, ifos, units="$1/\sqrt{\mathrm{Hz}}$")
        
		xmax = np.max(freqs)
        	for ax in axs:
                	ax.grid(True, which="both")
			ax.set_xlim(xmax=xmax)
			ax.legend(loc="best")
	        ax.set_xlabel("frequency [Hz]")

	        fig.savefig(inj_figname)
        	viz.plt.close(fig)
	        if opts.time:
        	        print "\t", time.time()-to

#=================================================
### build angprior
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "building angPrior"
	if opts.time:
		to = time.time()

nside_exp=config.getint("angPrior","nside_exp")
prior_type=config.get("angPrior","prior_type")
angPrior_kwargs={}
if prior_type=="uniform":
	pass
elif prior_type=="antenna_pattern":
	angPrior_kwargs["frequency"]=config.getfloat("angPrior","frequency")
	angPrior_kwargs["exp"]=config.getfloat("angPrior","exp")
else:
	raise ValueError, "prior_type=%s not understood"%prior_type

angprior = priors.angPrior(nside_exp, prior_type=prior_type, network=network, **angPrior_kwargs)

if opts.time:
	print "\t", time.time()-to

#=================================================
### build hprior
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "building hPrior"
	if opts.time:
		to = time.time()

pareto_a=config.getfloat("hPrior","pareto_a")
n_gaus_per_dec=config.getfloat("hPrior","n_gaus_per_dec")
log10_min=np.log10(config.getfloat("hPrior","min"))
log10_max=np.log10(config.getfloat("hPrior","max"))
n_gaus = max(1, int(round((log10_max-log10_min)*n_gaus_per_dec,0)))

variances=np.logspace(log10_min, log10_max, n_gaus)**2

pareto_means, pareto_covariance, pareto_amps = priors.pareto(pareto_a, n_freqs, n_pol, variances)

hprior = priors.hPrior(freqs, pareto_means, pareto_covariance, amplitudes=pareto_amps, n_gaus=n_gaus, n_pol=n_pol)

if opts.time:
	print "\t", time.time()-to

#=================================================
### build posterior_obj
#=================================================
if opts.verbose: 
	print "\n----------------------------------\n"
	print "building posterior"

posterior = posteriors.Posterior(network=network, hPrior=hprior, angPrior=angprior, seglen=seglen)

if config.has_option("posterior","num_proc"):
	posterior_num_proc=config.getint("posterior","num_proc")
else:
	posterior_num_proc = 1

byhand = config.has_option("posterior","byhand")

### set things
if opts.verbose:
	print "set_theta_phi"
	if opts.time:
		to=time.time()
posterior.set_theta_phi()
if opts.time:
	print "\t", time.time()-to

if opts.verbose:
	print "set_AB_mp"
	if opts.time:
		to = time.time()
posterior.set_AB_mp(num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=byhand)
if opts.time:
	print "\t", time.time()-to

if opts.verbose:
	print "set_P_mp"
	if opts.time:
		to = time.time()
posterior.set_P_mp(num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size, byhand=byhand)
if opts.time:
	print "\t", time.time()-to

if opts.verbose:
        print "setting data"
        if opts.time:
                to=time.time()
posterior.set_data( noise+inj )
posterior.set_dataB()
if opts.time:
        print "\t", time.time()-to

#=================================================
### compute log_posterior_elements
#=================================================
if opts.verbose:
	print "\n----------------------------------\n"
	print "computing log_posterior_elements"
	if opts.time:
		to = time.time()

log_posterior_elements, n_pol_eff = posterior.log_posterior_elements_mp(posterior.theta, posterior.phi, psi=0.0, invP_dataB=(posterior.invP, posterior.detinvP, posterior.dataB, posterior.dataB_conj), A_invA=None, diagnostic=False, num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size)

n_pol_eff = n_pol_eff[0] ### we only accept integer n_pol_eff for now...

if opts.time:
	print "\t", time.time()-to

#=================================================
### model_selection
#=================================================
selection=config.get("model_selection","selection")

if config.has_option("model_selection", "num_proc"):
	model_selection_num_proc = config.getint("model_selection","num_proc")
else:
	model_selection_num_proc = 1

if opts.verbose:
	print "\n----------------------------------\n"
	print "model selection with ", selection
	if opts.time:
		to = time.time()
###
if selection=="waterfill":
	model, log_bayes = model_selection.waterfill(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, freq_truth)

elif selection=="log_bayes_cut":
	if not config.has_option("model_selection","log_bayes_thr"):
		raise ValueError, "must supply \"log_bayes_thr\" in config file with \"selection=log_bayes_cut\""
	log_bayes_thr = config.getfloat("model_selection","log_bayes_thr")

	model, log_bayes = model_selection_mp.log_bayes_cut(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, freq_truth, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

###
elif selection=="fixed_bandwidth":
	if not config.has_option("model_selection","n_bins"):
		raise ValueError, "must supply \"n_bins\" in config file with \"selection=fixed_bandwidth\""
	n_bins = config.getint("model_selectin","n_bins")

	model, log_bayes = model_selection.fixed_bandwidth_mp(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, freq_truth, n_bins=n_bins, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

###
elif selection=="variable_bandwidth":
	if not config.has_option("model_selection","min_n_bins"):
		raise ValueError, "must supply \"min_n_bins\" in config file with \"selection=variable_bandwidth\""
	min_n_bins = config.getint("model_selection","min_n_bins")

	if not config.has_option("model_selection","max_n_bins"):
		raise ValueError, "must supply \"max_n_bins\" in config file with \"selection=variable_bandwidth\""
	max_n_bins = config.getint("model_selection","max_n_bins")

	if not config.has_option("model_selection","dn_bins"):
		raise ValueError, "must supply \"dn_bins\" in config file with \"selection=variable_bandwidth\""
	dn_bins = config.getint("model_selection","dn_bins")

	model, log_bayes = model_selection.variable_bandwidth_mp(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

###
else:
	raise ValueError, "selection=%s not understood"%selection

if opts.time:
	print "\t", time.time()-to

#=================================================
# compute final posterior
#=================================================

if opts.verbose: 
	print "\n----------------------------------\n"
	print "computing log_posterior"
	if opts.time:
		to = time.time()

log_posterior = posterior.log_posterior_mp(posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, model, normalize=True, num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size)

if opts.time:
	print "\t", time.time()-to

#=================================================
### save output
#=================================================
pklname = "%s/model%s_%d.pkl"%(opts.output_dir, opts.tag, int(opts.gps))
if opts.verbose:
	print "\n----------------------------------\n"
	print "saving model to %s"%pklname
	if opts.time:
		to = time.time()

pkl_obj = open(pklname, "w")
#pickle.dump(posterior, pkl_obj)
#pickle.dump(log_posterior_elements, pkl_obj)
pickle.dump(model, pkl_obj)
pickle.dump(log_bayes, pkl_obj)
pickle.dump(log_posterior, pkl_obj)
pkl_obj.close()
if opts.time:
	print "\t", time.time()-to

fitsname = "%s/posterior%s_%d.fits"%(opts.output_dir, opts.tag, int(opts.gps))
if opts.verbose:
	print "saving posterior to %s"%fitsname
	if opts.time:
		to=time.time()
	utils.hp.write_map(fitsname, np.exp(log_posterior))
if opts.time:
	print "\t", time.time()-to

### dump data to file if requested
if opts.diagnostic:
        ### compute mle_strain at MAP point of the sky
        mappix = np.argmax(log_posterior)
        map_theta = posterior.theta[mappix]
        map_phi = posterior.phi[mappix]

        ### save mle_strain to file
        if opts.verbose:
                print "computing mle_strain at MAP position"
                if opts.time:
                        to = time.time()
        mle_strain = posterior.mle_strain(map_theta, map_phi, invA_dataB=(posterior.invA[mappix], posterior.dataB[mappix]))
        if opts.time:
                print "\t", time.time()-to

        mle_filename = "%s/mle-strain%s_%d.pkl"%(opts.output_dir, opts.tag, int(opts.gps))
        if opts.verbose:
                print "writing mle_strain to %s"%mle_filename
                if opts.time:
                        to = time.time()
        file_obj = open(mle_filename, "w")
        pickle.dump((map_theta, map_phi), file_obj)
        pickle.dump(mle_strain, file_obj)
        file.close()
        if opts.time:
                print "\t", time.time()-to

        if opts.diagnostic_plots:
                ### project mle strain onto detectors and plot
                mle_figname = "%spng"%mle_filename[:-3]
                if opts.verbose:
                        print "projecting mle_strain and plotting : %s"%mle_figname
                        if opts.time:
                                to = time.time()
                fig, axs = viz.project(posterior.network, freqs, mle_strain, map_theta, map_phi, posterior.psi, posterior.data, units="$1/\sqrt{\mathrm{Hz}}$")

                xmax = np.max(freqs)
                for ax in axs:
                        ax.grid(True, which="both")
                        ax.set_xlim(xmax=xmax)
                        ax.legend(loc="best")
                ax.set_xlabel("frequency [Hz]")

                fig.savefig(mle_figname)
                viz.plt.close(fig)
                if opts.time:
                        print "\t", time.time()

