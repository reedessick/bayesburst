usage = "python analyze_maps.py [--options] label1,fits1 label2,fits2 ..."
description = "computes several descriptive statistics about the maps provided"
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

parser.add_option("", "--pvalue", default=False, type="string", help="\"theta, phi\" in the coordinate system of these maps for which we compute the pvalue (the confidence associated with the minimum credible region that marginally includes this location)")

parser.add_option("-H", "--entropy", default=False, action="store_true", help='computes the entropy of the map')

parser.add_option("-c", "--credible-interval", default=[], type='float', action='append', help='computes the size, max(dtheta), and num/size of disjoint regions for each confidence region. This argument can be supplied multiple times to analyze multiple confidence regions.')
parser.add_option("", "--no-credible-interval-dtheta", default=False, action="store_true", help='does not compute max(dtheta) for each confidence region. This may be desired if computational speed is an issue, because the max(dtheta) algorithm scales as Npix^2.')
parser.add_option("", "--no-disjoint-regions", default=False, action="store_true", help="does not compute number,size of disjoint regions for each confidence region.")

opts, args = parser.parse_args()

if opts.pvalue:
	theta, phi = [float(l) for l in opts.pvalue.split(",")]

#==========================================================
if opts.degrees:
	unit = "deg"
	areaunit = "deg2"
	angle_conversion = 180/np.pi
	theta /= angle_conversion
	phi /= angle_conversion
else:
	unit = "radians"
	areaunit = "stradians"
	angle_conversion = 1.0

#==========================================================

for arg in args:
	label, fits = arg.split(',')

	print label

	if opts.verbose:
		print "\treading map from %s"%(fits)
	post = hp.read_map( fits )
	npix = len(post)
	nside = hp.npix2nside(npix)

	print "\tnside=%d"%(nside)

	pixarea = hp.nside2pixarea(nside, degrees=opts.degrees)

	### compute statistics and report them
	if opts.pvalue:
		print "\t cdf(%s) = %.3f\%"%(opts.pvalue, stats.p_value(post, theta, phi, nside=nside)*100)
	# entropy -> size
	if opts.entropy:
		print "\t entropy = %.3f %s"%(pixarea*np.exp(stats.entropy(post, nside)), areaunit)

	# CR -> size, max(dtheta)
	for CR, conf in zip(stats.credible_region(posterior, opts.credible_interval), opts.credible_interval):
		print "\t %.3f\% CR :\t size= %.3f %s"%(conf*100, pixarea*len(CR), areaunit)
		if not opts.no_credible_interval_dtheta:
			print "\t \t max(dtheta) = %.3f %s"%(angle_conversion*np.arccos(stats.min_all_cos_dtheta(CR, nside, nest=False)), unit)
		if not opts.no_disjoint_regions:
			sizes = sorted([len(_)*pixarea for _ in stats.__into_modes(nside, CR)])
			print "\t \t disjoint regions : (%s) %s"%(", ".join(["%.3f"%x for x in sizes]), areaunit )
