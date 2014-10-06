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

from optparse import OptionParser

#=================================================
parser=OptionParser(usage=usage)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-c", "--config", default="./config.ini", type="string", help="config file")

parser.add_option("-g", "--gps", default=0, type="float", help="central time of this analysis")

parser.add_option("-x", "--xmlfilename", default=False, type="float", help="an injection xml file")
parser.add_option("-i", "--sim-id", default=False, type="int", help="the injection id in the xml file")

parser.add_option("-d", "--diagnostics", default=False, action="store_true", help="output some diagnostic data to check the pipeline's functionality")

parser.add_option("-o", "--output-dir", default="./", type="string")
parser.add_option("-t", "--tag", default="", type="string")

parser.add_option("", "--time", default=False, action="store_true")

opts, args = parser.parse_args()

if opts.tag:
	opts.tag = "_" + opts.tag

if not os.path.exists(opts.output_dir):
	os.makedirs(opts.output_dir)

if opts.time:
	opts.verbose=True
	import time

#=================================================
### load config file
#=================================================
if opts.verbose: 
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
	print "setting up fft paramters"
	if opts.time:
		to = time.time()

seglen = config.getfloat("fft","seglen")
df = 1.0/seglen
fs = config.getfloat("fft","fs")
freqs=np.arange(0, fs/2, df)
n_freqs = len(freqs)

if opts.time:
	print "\t", time.time()-to

#=================================================
### build network
#=================================================
if opts.verbose:
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
	fs = config.getloat("psd_estimation","fs")
	seglen= config.getfloat("psd_estimation","win")
	overlap = config.getfloat("psd_estimation","overlap")
	if overlap >= seglen:
		raise ValueError, "overlap must be less than seglen in psd_estimation"
	num_segs = config.getint("psd_estimation","num_segs")
	
	dur = (num_segs*seglen - (num_segs-1)*overlap) ### amount of time used for PSD estimation

	from pylal import Fr
	for ifo in ifos:

		ifo_chan = noise_chans[ifo]
		ifo_cache = noise_cache[ifo]
		if opts.verbose: 
			print "reading %s frames from %s with channel=%s"%(ifo, ifo_cache, ifo_chan)
			if opts.time:
				to = time.time()
		### find relevant files
		frames = utils.files_from_cache(ifo_cache, opts.gps-dur, opts.gps)

		if opts.time:
			print "\t", time.time()-to

		### read vectors from files
		### FIXME: we assume data is continuous, which may not be the case!
		### we also assume that data is sampled at a constant rate
		vecs = []
		dt = 0
		for frame in frames:
			if opts.verbose:
				print "\t", frame
				if opts.time:
					to = time.time()
			vec, gpstart, _, dt, _, _ = Fr.getvect1d(frame, ifo_chan, start=opts.gps-dur, stop=opts.gps)
			if opts.time:
				print "\t\t", time.time()-to
			vecs.append( vec )
		vec = np.concatenate(vecs)
		N = len(vec)

		### downsample data to fs?
		_dt = 1.0/fs
		if dt > _dt:
			raise ValueError, "cannot upsample data to higher sampling frequency: dt=%f -> dt=%f"%(dt, _dt)
		elif dt < _dt:
			if opts.verbose:
				print "downsampling time-domain data via interpolation : dt=%f -> dt=%f"%(dt, _dt)
				if opts.time:
					to = time.time()
			vec = np.interp( np.arange(0, N*dt, _dt), np.arange(0, N*dt, dt), vec)
			dt = _dt
			if opts.time:
				print "\t", time.time()-to
			
		### check that we have all the data we expect
		if len(vec)*dt != dur:
			raise ValueError, "len(vec)*dt = %f != %f = duration"%(len(vec)*dt, dur)

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

freq_truth = np.ones_like(freqs, bool) ### put this here in case we want to throw out lines, etc

### dump psd's to numpy arrays if requested
if opts.diagnostics:
	for ifo in ifos:
		psd_filename = "%s/%s-PSD%s_%d.pkl"%(opts.output_dir, ifo, opts.tag, int(opts.gps))
		if opts.verbose:
			print "writing %s PSD to %s"%(ifo, psd_filename)
			if opts.time:
				to = time.time()
		psd_obj = network.detectors[ifo].get_psd()
		file_obj = open(psd_filename, "w")
		pickle.dump(psd_obj.get_freqs(), file_obj)
		pickle.dump(psd_obj.get_psd(), file_obj)
		file_obj.close()
		if opts.time:
			print "\t", time.time()-to

#=================================================
### build angprior
#=================================================
if opts.verbose:
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

#=================================================
### load noise
#=================================================
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
        fs = config.getloat("fft","fs")
        seglen= config.getfloat("fft","win")

        from pylal import Fr
        for ifo_ind, ifo in enumerate(ifos):

                ifo_chan = noise_chans[ifo]
                ifo_cache = noise_cache[ifo]
                if opts.verbose:
                        print "reading %s frames from %s with channel=%s"%(ifo, ifo_cache, ifo_chan)
                        if opts.time:
                                to = time.time()
                ### find relevant files
                frames = utils.files_from_cache(ifo_cache, opts.gps-dur, opts.gps)

                if opts.time:
                        print "\t", time.time()-to

                ### read vectors from files
                ### FIXME: we assume data is continuous, which may not be the case!
                ### we also assume that data is sampled at a constant rate
                vecs = []
                dt = 0
                for frame in frames:
                        if opts.verbose:
                                print "\t", frame
                                if opts.time:
                                        to = time.time()
                        vec, gpstart, _, dt, _, _ = Fr.getvect1d(frame, ifo_chan, start=opts.gps, stop=opts.gps+seglen)
                        if opts.time:
                                print "\t\t", time.time()-to
                        vecs.append( vec )
                vec = np.concatenate(vecs)
                N = len(vec)

                ### downsample data to fs?
                _dt = 1.0/fs
                if dt > _dt:
                        raise ValueError, "cannot upsample data to higher sampling frequency: dt=%f -> dt=%f"%(dt, _dt)
                elif dt < _dt:
                        if opts.verbose:
                                print "downsampling time-domain data via interpolation : dt=%f -> dt=%f"%(dt, _dt)
                                if opts.time:
                                        to = time.time()
                        vec = np.interp( np.arange(0, N*dt, _dt), np.arange(0, N*dt, dt), vec)
                        dt = _dt
                        if opts.time:
                                print "\t", time.time()-to

                ### check that we have all the data we expect
                if len(vec)*dt != seglen:
                        raise ValueError, "len(vec)*dt = %f != %f = duration"%(len(vec)*dt, seglen)

		### store noise
		dft_vec, dft_freqs = utils.dft(vec, dt=dt)
		if dft_freqs != freqs:
			raise ValueError, "frequencies from utils.dft do not agree with freqs defined by hand"
		noise[:,ifo_ind] = dft_vec * np.exp(2j*np.pi*opts.gps*freqs) ### add phase shift for start time

else:
	if opts.verbose:
		print "no noise generation specified. zero-ing noise"
	noise = np.zeros((n_freqs, n_ifo), complex)

### dump noise to file
if opts.diagnostics:
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

#=================================================
### load injection
#=================================================
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

        inj_cache = eval(config.get("noise","cache"))
        inj_chans = eval(config.get("noise","channels"))

        ### pull out parameters for PSD estimation
        fs = config.getloat("fft","fs")
        seglen= config.getfloat("fft","win")

        from pylal import Fr
        for ifo_ind, ifo in enumerate(ifos):

                ifo_chan = inj_chans[ifo]
                ifo_cache = inj_cache[ifo]
                if opts.verbose:
                        print "reading %s frames from %s with channel=%s"%(ifo, ifo_cache, ifo_chan)
                        if opts.time:
                                to = time.time()
                ### find relevant files
                frames = utils.files_from_cache(ifo_cache, opts.gps-dur, opts.gps)

                if opts.time:
                        print "\t", time.time()-to

                ### read vectors from files
                ### FIXME: we assume data is continuous, which may not be the case!
                ### we also assume that data is sampled at a constant rate
                vecs = []
                dt = 0
                for frame in frames:
                        if opts.verbose:
                                print "\t", frame
                                if opts.time:
                                        to = time.time()
                        vec, gpstart, _, dt, _, _ = Fr.getvect1d(frame, ifo_chan, start=opts.gps, stop=opts.gps+seglen)
                        if opts.time:
                                print "\t\t", time.time()-to
                        vecs.append( vec )
                vec = np.concatenate(vecs)
                N = len(vec)

                ### downsample data to fs?
                _dt = 1.0/fs
                if dt > _dt:
                        raise ValueError, "cannot upsample data to higher sampling frequency: dt=%f -> dt=%f"%(dt, _dt)
                elif dt < _dt:
                        if opts.verbose:
                                print "downsampling time-domain data via interpolation : dt=%f -> dt=%f"%(dt, _dt)
                                if opts.time:
                                        to = time.time()
                        vec = np.interp( np.arange(0, N*dt, _dt), np.arange(0, N*dt, dt), vec)
                        dt = _dt
                        if opts.time:
                                print "\t", time.time()-to

                ### check that we have all the data we expect
                if len(vec)*dt != seglen:
                        raise ValueError, "len(vec)*dt = %f != %f = duration"%(len(vec)*dt, seglen)

                ### store noise
                dft_vec, dft_freqs = utils.dft(vec, dt=dt)
                if dft_freqs != freqs:
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
if opts.diagnostics:
	inj_filename = "%s/injections%s_%d.pkl"%(opts.output_dir, opts.tag, int(opts.gps))
        if opts.verbose:
                print "writing injections to %s"%inj_filename
                if opts.time:
                        to = time.time()
        file_obj = open(noise_filename, "w")
        pickle.dump(inj, file_obj)
        file_obj.close()
        if opts.time:
                print "\t", time.time()-to

#=================================================
### set data, dataB
#=================================================
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
	print "computing log_posterior"
	if opts.time:
		to = time.time()

log_posterior = posterior.log_posterior_mp(posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, model, normalize=True, num_proc=posterior_num_proc, max_proc=max_proc, max_array_size=max_array_size)

if opts.time:
	print "\t", time.time()-to

#=================================================
### save output
#=================================================
pklname = "%s/pipeline%s.pkl"%(opts.output_dir, opts.tag)
if opts.verbose:
	print "saving data to ", pklname
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
