#!/usr/bin/python

usage = """pipeline.py [--options] 
a script meant to run the engine contained in posteriors.py and priors.py"""

import ConfigParser
import os
import numpy as np
import pickle

import utils
import dft
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

parser.add_option("-x", "--xmlfilename", default=False, type="string", help="an injection xml file")
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

opts.output_dir += "/%d/"%int(opts.gps)

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
	print "loading config from :", opts.config
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

resample_method=config.get("fft","resample_method") ### method for downsampling

seglen = config.getfloat("fft","seglen")
fs = config.getfloat("fft","fs")
padding = config.getfloat("fft","padding")

df = 1.0/seglen
freqs=np.arange(0, fs/2, df)
n_freqs = len(freqs)

### low frequency range for the analysis
if config.has_option("fft","flow"):
	flow = config.getfloat("fft","flow") ### currently only used in plotting and model selection
        	                             ### any speed-up associated with skipping these bins in 
                	                     ### log posterior elements is currently unavailable

	if flow < 1.0/seglen:
		if opts.verbose:
			print "WARNING: seglen is too short to reach flow=%f. changing flow->1/seglen=%f"%(flow,1.0/seglen)
		flow=1.0/seglen
else:
	flow = 1.0/seglen

### high frequency range for the analysis
if config.has_option("fft","fhigh"):
	fhigh = config.getfloat("fft","fhigh")
	if fhigh > 0.5*fs:
		if opts.verbose:
			print "WARNING: fs is too low to reach fhigh=%f. changing fhigh->fs/2=%f"%(fhigh, 0.5*fs)
		fhigh = 0.5*fs
else:
	fhigh = 0.5*fs

### window of frequencies we want to analyze
freq_truth = (freqs >= flow)*(freqs <= fhigh) 

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
network = utils.Network([detector_cache.detectors[ifo] for ifo in ifos], freqs=freqs, Np=n_pol) ### we use freqs for now, but will update this based on the data

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
	psd_fs = config.getfloat("psd_estimation","fs")
	psd_seglen= config.getfloat("psd_estimation","seglen")
	psd_padding = config.getfloat("psd_estimation","padding")
	overlap = config.getfloat("psd_estimation","overlap")
	if overlap >= psd_seglen:
		raise ValueError, "overlap must be less than seglen in psd_estimation"
	num_segs = config.getint("psd_estimation","num_segs")
	
	duration = (num_segs*psd_seglen - (num_segs-1)*overlap) ### amount of time used for PSD estimation
	dseg = 0.5*config.getfloat("fft","seglen") ### buffer for fft used to compute noise, inj, etc

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
		
		vec, dt = utils.vec_from_frames(frames, opts.gps-dseg-duration, opts.gps-dseg, verbose=opts.verbose)
		N = len(vec)

		### downsample data to fs?
		### IF WE WANT TO DO THIS, WE SHOULD DO THIS WITHIN THE PSD ESTIMATION?
		_dt = 1.0/psd_fs
		if opts.verbose:
                	print "downsampling time-domain data : dt=%fe-6 -> dt=%fe-6"%(dt*1e6, _dt*1e6)
			if opts.time:
				to = time.time()
		vec, dt = dft.resample(vec, dt, _dt, method=resample_method)
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
		psd, psd_freqs = dft.estimate_psd(vec, num_segs=num_segs, overlap=overlap/dt, dt=dt)
		if opts.time:
			print "\t", time.time()-to

		if config.has_option("psd_estimation","smooth"): ### smooth psd estimate by averaging neighboring bins
			N = len(psd)
			smooth = config.getint("psd_estimation","smooth")

			psd = np.sum(np.reshape(psd, (N/smooth, smooth)), axis=1)/smooth
			psd_freqs = np.sum(np.reshape(psd_freqs, (N/smooth,smooth)), axis=1)/smooth

		### update network object
		network.detectors[ifo].set_psd(psd, freqs=psd_freqs)

else:
	raise StandardError, "could not estimate PSDs. Please either supply a cache of ascii files or a cache of noise frames in the configuration file."

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
			ax.set_xlim(xmin=flow, xmax=fhigh)
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
        dseg = 0.5*seglen

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

		vec, dt = utils.vec_from_frames(frames, opts.gps-dseg, opts.gps+dseg, vebose=opts.verbose)
		N = len(vec)

                ### downsample data to fs?
                _dt = 1.0/fs
                if opts.verbose:
                        print "downsampling time-domain data : dt=%fe-6 -> dt=%fe-6"%(dt*1e6, _dt*1e6)
			if opts.time:
				to = time.time()
                vec, dt = dft.resample(vec, dt, _dt, method=resample_method)
                if opts.time:
                        print "\t", time.time()-to

                ### check that we have all the data we expect
                if len(vec)*dt != seglen:
                        raise ValueError, "len(vec)*dt = %f != %f = seglen"%(len(vec)*dt, seglen)

		### compute windowing function
		win = dft.window(vec, kind="tukey", alpha=2*padding/seglen)

                ### store noise
                dft_vec, dft_freqs = dft.dft(vec*win, dt=dt)
                if np.any(dft_freqs != freqs):
                        raise ValueError, "frequencies from utils.dft do not agree with freqs defined by hand"
                noise[:,ifo_ind] = dft_vec

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
		fig, axs = viz.data(freqs[freq_truth], noise[freq_truth], ifos, units="$1/\sqrt{\mathrm{Hz}}$")

		for ax in axs:
			ax.grid(True, which="both")
			ax.set_xlim(xmin=flow, xmax=fhigh)
			ax.legend(loc="best")
		ax.set_xlabel("frequency [Hz]")

		fig.savefig(noise_figname)
		viz.plt.close(fig)
		if opts.time:
			print "\t", time.time()-to

		### log axes
		if np.any(noise.real[freq_truth]**2+noise.imag[freq_truth]**2 > 0):
	                noise_figname = "%s/noise-log%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
			if opts.verbose:
	                        print "plotting noise : %s"%noise_figname
        	                if opts.time:
                	                to = time.time()
	                fig, axs = viz.data(freqs[freq_truth], noise.real[freq_truth]**2+noise.imag[freq_truth]**2, ifos, units="$1/\mathrm{Hz}$")

        	        for ax in axs:
                	        ax.grid(True, which="both")
				ax.set_xscale('log')
				ax.set_yscale('log')
        	                ax.set_xlim(xmin=flow, xmax=fhigh)
#               	         ax.legend(loc="best")
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
        dseg = 0.5*seglen

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

		vec, dt = utils.vec_from_frames(frames, opts.gsp-dseg, opts.gps+dseg, verbose=opts.verbose)
                N = len(vec)

                ### downsample data to fs?
                _dt = 1.0/fs
                if opts.verbose:
                        print "downsampling time-domain data : dt=%fe-6 -> dt=%fe-6"%(dt*1e6, _dt*1e6)
			if opts.time:
				to = time.time()
                vec, dt = dft.resample(vec, dt, _dt, method=resample_method)
                if opts.time:
                        print "\t", time.time()-to

                ### check that we have all the data we expect
                if len(vec)*dt != seglen:
                        raise ValueError, "len(vec)*dt = %f != %f = seglen"%(len(vec)*dt, seglen)

                ### compute windowing function
                win = dft.window(vec, kind="tukey", alpha=2*padding/seglen)

                ### store injection
                dft_vec, dft_freqs = dft.dft(vec*win, dt=dt)
                if np.any(dft_freqs != freqs):
                        raise ValueError, "frequencies from utils.dft do not agree with freqs defined by hand"
                inj[:,ifo_ind] = dft_vec 

elif opts.xmlfilename: ### read injection from xmlfile
        if opts.sim_id==None:
                raise StandardError, "must supply --event-id with --xmlfilename"

        if opts.verbose:
                print "reading simulation_id=%d from %s"%(opts.sim_id, opts.xmlfilename)
                if opts.time:
                        to = time.time()

	from glue.ligolw import ligolw
        from glue.ligolw import lsctables
        from glue.ligolw import utils as ligolw_utils

	lsctables.use_in(ligolw.LIGOLWContentHandler)

        ### we specialize to sim_burst tables for now...
        table_name = lsctables.SimBurstTable.tableName

        ### load xmlfile and find row corresponding to the specified entry
        xmldoc = ligolw_utils.load_filename(opts.xmlfilename, contenthandler=ligolw.LIGOLWContentHandler)
        tbl = lsctables.table.get_table(xmldoc, table_name)
        for row in tbl:
                if str(row.simulation_id) == "sim_burst:simulation_id:%d"%opts.sim_id: ### FRAGILE AND NOT PRETTY
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

        elif (row.waveform == "SineGaussian") or (row.waveform == "SineGaussianF"): 
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

		if tau != tau: ### tau == nan (i.e.: not set) Compute using q
			q = row.q
			tau = q / (2**0.5 * np.pi * fo)

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

	### snr
	snr_filename = "%s/snr%s_%d.txt"%(opts.output_dir, opts.tag, int(opts.gps))
	if opts.verbose:
		print "writing snrs to %s"%snr_filename
		if opts.time:
			to = time.time()
	snrs = network.snrs(inj)
	snr_net = np.sum(snrs**2)**0.5
	snr_string =  " ".join([str(snr_net)]+[str(s) for s in snrs])
	file_obj = open(snr_filename, "w")
	print >> file_obj, " ".join(["network"]+network.detector_names_list())
	print >> file_obj, snr_string
	file_obj.close()
	if opts.verbose:
		print "\t%s"%snr_string
		if opts.time:
			print "\t", time.time()-to

	if opts.diagnostic_plots:
	        ### plot injection
        	inj_figname = "%spng"%inj_filename[:-3]
	        if opts.verbose:
        	        print "plotting injections : %s"%inj_figname
                	if opts.time:
                        	to = time.time()
	        fig, axs = viz.data(freqs[freq_truth], inj[freq_truth], ifos, units="$1/\sqrt{\mathrm{Hz}}$")
        
        	for ax in axs:
                	ax.grid(True, which="both")
			ax.set_xlim(xmin=flow, xmax=fhigh)
			ax.legend(loc="best")
	        ax.set_xlabel("frequency [Hz]")

	        fig.savefig(inj_figname)
        	viz.plt.close(fig)
	        if opts.time:
        	        print "\t", time.time()-to

		### log axis
		if np.any(inj.real[freq_truth]**2+inj.imag[freq_truth]**2 > 0):
			inj_figname = "%s/injections-log%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
			if opts.verbose:
                	        print "plotting injections : %s"%inj_figname
                        	if opts.time:
                                	to = time.time()
	                fig, axs = viz.data(freqs[freq_truth], inj.real[freq_truth]**2+inj.imag[freq_truth]**2, ifos, units="$1/\mathrm{Hz}$")

        	        for ax in axs:
                	        ax.grid(True, which="both")
				ax.set_xscale('log')
				ax.set_yscale('log')
        	                ax.set_xlim(xmin=flow, xmax=fhigh)
#				ax.legend(loc="best")
	                ax.set_xlabel("frequency [Hz]")
	
        	        fig.savefig(inj_figname)
                	viz.plt.close(fig)
	                if opts.time:
        	                print "\t", time.time()-to

#=================================================
# some diagnostic plotting
#=================================================
if opts.diagnostic_plots:
	data_figname = "%s/data%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
	if opts.verbose:
		print "\n----------------------------------\n"
		print "plotting data : %s"%inj_figname
		if opts.time:
                	to = time.time()
	fig, axs = viz.data(freqs[freq_truth], noise[freq_truth]+inj[freq_truth], ifos, units="$1/\sqrt{\mathrm{Hz}}$")

        for ax in axs:
        	ax.grid(True, which="both")
                ax.set_xlim(xmin=flow, xmax=fhigh)
                ax.legend(loc="best")
	ax.set_xlabel("frequency [Hz]")

	fig.savefig(data_figname)
        viz.plt.close(fig)
        if opts.time:
        	print "\t", time.time()-to

	### log axis
	if np.any(inj.real[freq_truth]**2+inj.imag[freq_truth]**2 + noise[freq_truth].real**2+noise[freq_truth].imag**2 > 0):
	        data_figname = "%s/data-log%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
	        if opts.verbose:
        		print "plotting injections : %s"%inj_figname
                	if opts.time:
                		to = time.time()
		fig, axs = viz.data(freqs[freq_truth], inj.real[freq_truth]**2+inj.imag[freq_truth]**2 + noise[freq_truth].real**2+noise[freq_truth].imag**2, ifos, units="$1/\mathrm{Hz}$")

        	for ax in axs:
	        	ax.grid(True, which="both")
        	        ax.set_xscale('log')
                	ax.set_yscale('log')
	                ax.set_xlim(xmin=flow, xmax=fhigh)
#			ax.legend(loc="best")
		ax.set_xlabel("frequency [Hz]")

		fig.savefig(data_figname)
		viz.plt.close(fig)
		if opts.time:
			print "\t", time.time()-to


##### WRITE a whitened version that references PSD's!

#=================================================
# set frequencies for the analysis
#=================================================
### SELECT OFF ONLY THOSE FREQUENCIES THAT BELONG TO freq_truth AND USE THOSE THROUGHOUT THE REST OF THIS.
### network object is instantiated with freqs=freqs, so we'll need to update that
### We can choose which frequencies we search based on a simple SNR calculation (a-la cWB?)
### This is primarily a computational concern, but it can have important impacts for parameter estimation if we aren't careful.
if opts.verbose:
	print "\n----------------------------------\n"
	print "selecting only those frequencies within [flow, fhigh]"
	if opts.time:
		to = time.time()

analysis_freqs = freqs[freq_truth]
analysis_inj = inj[freq_truth]
analysis_noise = noise[freq_truth]

analysis_freq_truth = np.ones_like(analysis_freqs, bool)
analysis_n_freqs = len(analysis_freq_truth)

if opts.verbose:
	print "\tn_freqs : %d -> %d"%(n_freqs, analysis_n_freqs)
	if opts.time:
		print "\t", time.time()-to

### update network object!
network.freqs=analysis_freqs






#=============================================================================================================
#
# CLEAN UP VARIABLES THAT ARE NO LONGER NEEDED
# eg: data from outside the analysis_frequency range
# intermediate data products, etc.
# THIS WILL HELP WITH MEMORY REQUIREMENTS
#
#=============================================================================================================








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

pareto_means, pareto_covariance, pareto_amps = priors.pareto(pareto_a, analysis_n_freqs, n_pol, variances)

hprior = priors.hPrior(analysis_freqs, pareto_means, pareto_covariance, amplitudes=pareto_amps, n_gaus=n_gaus, n_pol=n_pol)

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
posterior.set_data( analysis_noise+analysis_inj )
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
	model, log_bayes = model_selection.waterfill_mp(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, analysis_freq_truth, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

elif selection=="log_bayes_cut":
	if not config.has_option("model_selection","log_bayes_thr"):
		raise ValueError, "must supply \"log_bayes_thr\" in config file with \"selection=log_bayes_cut\""
	log_bayes_thr = config.getfloat("model_selection","log_bayes_thr")
	if opts.verbose:
		print "\tlog_bayes_thr =", log_bayes_thr
	model, log_bayes = model_selection.log_bayes_cut_mp(log_bayes_thr, posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, analysis_freq_truth, joint_log_bayes=True, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

###
elif selection=="fixed_bandwidth":
	if not config.has_option("model_selection","n_bins"):
		raise ValueError, "must supply \"n_bins\" in config file with \"selection=fixed_bandwidth\""
	n_bins = config.getint("model_selection","n_bins")

	model, log_bayes = model_selection.fixed_bandwidth_mp(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, analysis_freq_truth, n_bins=n_bins, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

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

	model, log_bayes = model_selection.variable_bandwidth_mp(posterior, posterior.theta, posterior.phi, log_posterior_elements, n_pol_eff, analysis_freq_truth, min_n_bins=min_n_bins, max_n_bins=max_n_bins, dn_bins=dn_bins, num_proc=model_selection_num_proc, max_proc=max_proc, max_array_size=max_array_size)

###
else:
	raise ValueError, "selection=%s not understood"%selection

if opts.verbose:
	print "n_bins : %d / %d"%(np.sum(model), len(model))
	print "logB   : %f"%log_bayes
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

### generate plots
if opts.diagnostic_plots:
	### log-posterior
	log_figname = "%s/posterior-log%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
	if opts.verbose:
		print "plotting : %s"%log_figname
		if opts.time:
			to = time.time()
	fig, ax = viz.hp_mollweide(log_posterior, unit="log(prob/pix)")
	fig.savefig(log_figname)
	viz.plt.close(fig)
	if opts.time:
		print "\t", time.time()-to

	### posterior
	pst_figname = "%s/posterior%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
	if opts.verbose:
		print "plotting : %s"%pst_figname
		if opts.time:
			to = time.time()
	fig, ax = viz.hp_mollweide(np.exp(log_posterior), unit="prob/pix")
	fig.savefig(pst_figname)
	viz.plt.close(fig)
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
        pkl_obj = open(mle_filename, "w")
        pickle.dump((map_theta, map_phi), pkl_obj)
        pickle.dump(mle_strain, pkl_obj)
        pkl_obj.close()
        if opts.time:
                print "\t", time.time()-to

        if opts.diagnostic_plots:
                ### project mle strain onto detectors and plot
                mle_figname = "%spng"%mle_filename[:-3]
                if opts.verbose:
                        print "projecting mle_strain and plotting : %s"%mle_figname
                        if opts.time:
                                to = time.time()

                fig, axs = viz.project(posterior.network, analysis_freqs, mle_strain, map_theta, map_phi, posterior.psi, posterior.data, units="$1/\sqrt{\mathrm{Hz}}$")

                for ax in axs:
                        ax.grid(True, which="both")
                        ax.set_xlim(xmin=flow, xmax=fhigh)
                        ax.legend(loc="best")
                ax.set_xlabel("frequency [Hz]")

                fig.savefig(mle_figname)
                viz.plt.close(fig)
                if opts.time:
                        print "\t", time.time()-to

		### log
		if np.any(mle_strain.real**2+mle_strain.imag**2 > 0):
			mle_figname = "%s/mle-strain-log%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
	                if opts.verbose:
        	                print "projecting mle_strain and plotting : %s"%mle_figname
                	        if opts.time:
                        	        to = time.time()

	                fig, axs = viz.project(posterior.network, analysis_freqs, mle_strain.real**2+mle_strain.imag**2, map_theta, map_phi, posterior.psi, posterior.data, units="$1/\sqrt{\mathrm{Hz}}$")

        	        for ax in axs:
                	        ax.grid(True, which="both")
				ax.set_yscale('log')
				ax.set_xscale('log')
        	                ax.set_xlim(xmin=flow, xmax=fhigh)
                	        ax.legend(loc="best")
	                ax.set_xlabel("frequency [Hz]")

        	        fig.savefig(mle_figname)
                	viz.plt.close(fig)
	                if opts.time:
        	                print "\t", time.time()-to

		### model
		model_figname = "%s/model-strain%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
		if opts.verbose:
			print "projecting mle_strain and plotting : %s"%mle_figname
                        if opts.time:
                                to = time.time()

                h = np.zeros_like(mle_strain, complex)
                h[model,:] = mle_strain[model,:] ### only include those bins selected by the model!

                fig, axs = viz.project(posterior.network, analysis_freqs, h, map_theta, map_phi, posterior.psi, posterior.data, units="$1/\sqrt{\mathrm{Hz}}$")

                for ax in axs:
                        ax.grid(True, which="both")
                        ax.set_xlim(xmin=flow, xmax=fhigh)
                        ax.legend(loc="best")
                ax.set_xlabel("frequency [Hz]")

                fig.savefig(model_figname)
                viz.plt.close(fig)
                if opts.time:
                        print "\t", time.time()-to

		### model log
		if np.any(h.real**2+h.imag**2 > 0):
	                model_figname = "%s/model-strain-log%s_%d.png"%(opts.output_dir, opts.tag, int(opts.gps))
        	        if opts.verbose:
                	        print "projecting mle_strain and plotting : %s"%mle_figname
                        	if opts.time:
                                	to = time.time()

	                fig, axs = viz.project(posterior.network, analysis_freqs, h.real**2+h.imag**2, map_theta, map_phi, posterior.psi, posterior.data, units="$1/\sqrt{\mathrm{Hz}}$")
	
        	        for ax in axs:
                	        ax.grid(True, which="both")
				ax.set_yscale('log')
				ax.set_xscale('log')
        	                ax.set_xlim(xmin=flow, xmax=fhigh)
                	        ax.legend(loc="best")
	                ax.set_xlabel("frequency [Hz]")

        	        fig.savefig(model_figname)
                	viz.plt.close(fig)
	                if opts.time:
        	                print "\t", time.time()-to


