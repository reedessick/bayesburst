usage = "python compare_maps.py [--options] label1,fits1 label2,fits2 ..."
description = "computes several comparison statistics for the FITs files provided. Computes comparison statistics for all possible pairings (downsampling the finner resolution map if needed)"
author = "R. Essick (reed.essick@ligo.org)"

#==========================================================

import numpy as np
import healpy as hp

import stats

from optparse import OptionParser

#==========================================================
parser = OptionParser(usage=usage, description=description)

parser.add_option("-v", "--verbose", default=False, action="store_true")

parser.add_option("-d", "--degrees", default=False, action="store_true")

parser.add_option("", "--fidelity", default=False, action="store_true", help="compute the fidelity between maps")
parser.add_option("", "--symKL", default=False, action="store_true", help="compute symmetric KLdivergence between maps")
parser.add_option("", "--mse", default=False, action="store_true", help="compute the mean square error between maps")
parser.add_option("", "--peak-snr", default=False, action="store_true", help="compute peak snr between maps")
parser.add_option("", "--structural-similarity", default=False, action="store_true", help="compute structural similarity between maps")
parser.add_option("", "--pearson", default=False, action="store_true", help="compute pearson correlation coefficient between maps")
parser.add_option("", "--dot", default=False, action="store_true", help="compute normalized dot product between maps")

parser.add_option("-c", "--credible-interval", default=[], type='float', action='append', help='compute the overlap and intersection of the credible intervals reported in the maps')

parser.add_option("", "--graceid", default=[], type="string", action="append", help="will upload annotations to GraceDB events. if used, there must be one graceid per argment. DO NOT USE UNLESS YOU HAVE LALSuite AND PERMISSION TO ANNOTATE GraceDB!")

parser.add_option('', '--gdb-url', default='https://gracedb.ligo.org/api', type='string')

opts, args = parser.parse_args()

if opts.graceid:
	raise StandardError("WRITE CONNECTION TO GRACEDB. NEED TO ANNOTATE THE EVENTS. POST COMPARISON TO EACH graceid INVOLVED, but do not duplicate if they are the same graceid.")
        from ligo.gracedb.rest import GraceDb
        gracedb = gracedb = GraceDb(opts.gdb_url)

if opts.graceid and len(opts.gracedb)!=len(args):
        raise ValueError("when supplying --graceid, you must supply the same number of graceid entries and fits files")

opts.credible_interval = sorted(opts.credible_interval)

maps = {}
if opts.graceid:
	for gid, arg in args:
		label, fits = arg.split(",")
		maps[label] = {"fits":fits, "graceid":gid}
else:
	for arg in args:
		label, fits = arg.split(",")
		maps[label] = {"fits":fits}

labels = sorted(maps.keys())

#==========================================================

if opts.degrees:
        unit = "deg"
        areaunit = "deg2"
        angle_conversion = 180/np.pi
else:
        unit = "radians"
        areaunit = "stradians"
        angle_conversion = 1.0

#==========================================================
### load posteriors from fits files

for label in labels:
        d = maps[label]
        fits = d['fits']
        if opts.verbose:
                print "reading map from", fits
        post = hp.read_map(fits)
        npix = len(post)
        nside = hp.npix2nside(npix)
        if opts.verbose:
                print "\tnside=%d"%nside

        d['post'] = post
        d['npix'] = npix
        d['nside'] = nside

#=================================================
### iterate through pairs and compute statistics

for ind, label1 in enumerate(labels):
	d1 = maps[label1]
	post1 = d1['post']
	nside1 = d1['nside']
	
	for label2 in labels[ind+1:]:

		d2 = maps[label2]
		post2 = d2['post']
		nside2 = d2['nside']

		print "%s vs %s"%(label1, label2)
		
		### resample if needed
		nside = min(nside1, nside)
		pixarea = hp.nside2pixarea(nside, degrees=opts.degrees)

		if nside2 > nside:
			if opts.verbose:
				print "resampling %s : %d -> %d"%(label2, nside2, nside1)
			post1 = stats.resample(post2, nside1, nest=False)
		elif nside1 > nside:
			if opts.verbose:
				print "resampling %s : %d -> %d"%(label1, nside1, nside2)
			post2 = stats.resample(post1, nside2, nest=False)
		
		### compute statistics
		if opts.fidelity:
			print "\t fidelity : %.5f"%(stats.fidelity(post1, post2))

		if opts.symKL:
			print "\t symmetric KL divergence : %.5f"%stats.symmetric_KLdivergence(post1, post2)

		if opts.mse:
			print "\t mean square error : %.5e"%stats.mse(post1, post2)

		if opts.peak_snr:
			print "\t peak SNR : (%.5f, %.5f)"%(stats.peak_snr(post1, post2))

		if opts.structural_similarity:
			print "\t structural similarity : %.5f"%stats.structural_similarity(post1, post2)

		if opts.pearson:
			print "\t pearson : %.5f"%stats.pearson(post1, post2)

		if opts.dot:
			print "\t dot : %.5f"%stats.dot(post1, post2)

		
		for conf, pix1, pix2 in zip(opts.credible_interval, stats.credible_region(post1, opts.credible_interval), stats.credible_region(post2, opts.credible_interval) ):
			conf100 = 100*conf
			print "\t %.3f %s CR : %s = %.3f %s"%(conf100, "%", label1, pixarea*len(pix1) , areaunit)
			print "\t %.3f %s CR : %s = %.3f %s"%(conf100, "%", label2, pixarea*len(pix2) , areaunit)
			i, u = stats.geometric_overlap(pix1, pix2, nside=nside, degrees=opts.degrees)
			print "\t %.3f %s CR : intersection = %.3f %s"%(conf100, "%", i, areaunit)
			print "\t %.3f %s CR : union = %.3f %s"%(conf100, "%", u, areaunit)


