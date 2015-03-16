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

parser.add_option("", "--fidelity", default=False, action="store_true", help="compute the fidelity between maps")
parser.add_option("", "--symKL", default=False, action="store_true", help="compute symmetric KLdivergence between maps")
parser.add_option("", "--mse", default=False, action="store_true", help="compute the mean square error between maps")
parser.add_option("", "--peak-snr", default=False, action="store_true", help="compute peak snr between maps")
parser.add_option("", "--structural-similarity", default=False, action="store_true", help="compute structural similarity between maps")
parser.add_option("", "--pearson", default=False, action="store_true", help="compute pearson correlation coefficient between maps")
parser.add_option("", "--dot", default=False, action="store_true", help="compute normalized dot product between maps")

parser.add_option("-c", "--credible-interval", default=[], type='float', action='append', help='compute the overlap and intersection of the credible intervals reported in the maps')

opts, args = parser.parse_args()

maps = dict( (label, {'fits':fits}) for arg in args for label, fits in arg.split(',') )
label = sorted(maps.keys())

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
	post1 = d1['fits']
	nside1 = d1['nside']
	
	for label2 in enumerate(labels[ind+1:]):

		d2 = maps[label2]
		post2 = d2['fits']
		nside2 = d2['nside']

		print "%s vs %s"%(label1, label2)
		
		### resample if needed
		nside = min(nside1, nside)
		pixarea = hp.nside2pixarea(nside, degrees=opts.degrees)

		if nside2 > nside:
			if opts.verbose:
				print "resampling %s : %d -> %d"%(label2, nside2, nside1)
			post1 = stats.resample(post2, nside1, nest=False)
		elif nsid1 > nside:
			if opts.verbose:
				print "resampling %s : %d -> %d"%(label1, nside1, nside2)
			post2 = stats.resample(post1, nside2, nest=False)
		
		### compute statistics
		if opts.fidelity:
			print "\t fidelity : %.5f"%(stats.fidelity(post1, post2))

		if opts.symKLdivergence:
			print "\t symmetric KL divergence : %.5f"%stats.symmetric_KLdivergence(post1, post2)

		if opts.mse:
			print "\t mean square error : %.5f"%stats.mse(post1, post2)

		if opts.peak_snr:
			print "\t peak SNR : %.5f"%stats.peak_snr(post1, post2)

		if opts.structural-similarity:
			print "\t structural_similarity : %.5f"%stats.structural_similarity(post1, post2)

		if opts.pearson:
			print "\t pearson : %.5f"%stats.pearson(post1, post2)

		if opts.dot:
			print "\t dot : %.5f"%stats.dot(post1, post2)

		
		for conf, pix1, pix2 in zip(opts.credible_interval, stats.credible_region(post1, opts.credible_interval), stats.credible_region(post2, opts.credible_interval) ):
			conf100 = 100*conf
			print "\t %.3f\% CR : %s = %.3f %s"%(conf100, label1, pixarea*len(pix1) , areaunit)
			print "\t %.3f\% CR : %s = %.3f %s"%(conf100, label2, pixarea*len(pix2) , areaunit)
			i, u = stats.geometric_overlap(pix1, pix2, nside=nside, degrees=opts.degrees)
			print "\t %.3f\% CR : intersection = %.3f %s"%(conf100, i, areaunit)
			print "\t %.3f\% CR : union = %.3f %s"%(conf100, u, areaunit)


