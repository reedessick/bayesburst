usage = """ compute pixels that should be colored for contours """

import healpy as hp
import numpy as np

#=================================================
def contour_pix(map, vals, all_neighbours=True):
	"""
	given a healpix map (map) and a set of values, we find and return lists of pixels that constitute the boarder of those sets
	"""
	npix = len(map)
	nside = hp.npix2nside(npix)

	### separate into pixel sets based in p_values
	pix = np.arange(npix)
	boarders = []
	for val in vals:
		_pix = pix[map>=val] ### pull out included pixels

		truth = np.zeros((npix,), bool)
		truth[_pix] = True

		boarder = np.zeros((npix,),bool) ### defines which modes are in the boarder
		for ipix in _pix:
			if all_neighbours:
				boarder[ipix] = not truth[[n for n in hp.get_all_neighbours(nside, ipix) if n != -1]].all()
			else:
				boarder[ipix] = not truth[[n for n in hp.get_neighbours(nside, ipix)[0] if n != -1]].all()

		boarders.append( pix[boarder] )

	return boarders

###
def projplot_contour_pix(map, vals, ax, color="w", markersize=0.001, marker=".", linestyle="none", linewidth=1, alpha=0.25, verbose=False):
	"""
	computes the countour pixels and colors them with projplot
	"""
	npix = len(map)
	nside = hp.npix2nside(npix)

	if verbose: 
		print "finding boarder pixels"
		i=0
	for boarder in contour_pix(map, vals):
		if verbose: 
			print "%d / %d : %f"%(i+1, len(vals), vals[i])
			i+=1
		for pos in boarder_to_lines(boarder, nside, verbose=verbose):
			brdr = ax.projplot(pos, color=color, alpha=alpha, linestyle=linestyle, marker=marker, linewidth=linewidth)[0]
			brdr.set_markersize(markersize)

###
def boarder_to_lines(boarder, nside, verbose=False):
	"""
	takes a list of pixels (boarder) and converts it to a list of positions represnting a line tracing the boarder
	"""
	#==============================================================================================
	### dot every point along the ring
#	theta, phi = hp.pix2ang(nside, boarder)
#	pos = [list(theta), list(phi)]
#	return [pos]

	#==============================================================================================
	### walk around the rings
	npix = hp.nside2npix(nside)
	pix = np.arange(npix)

	boarder_truth = np.zeros((npix,),bool) ### original boarder
	boarder_truth[boarder] = True

	truth = np.zeros((npix,),bool) ### we change this to denote which pixels have not been visited
	truth[boarder] = True

	visit = np.zeros((npix,),bool) ### pixels we have visited

	ipix = pix[truth][0] ### pull out the first pixel
	truth[ipix] = False ### turn that pixel off
	visit[ipix] = True
	line = [ipix] ### start of this line

	lines = []
	### iterate over boarder
	while truth.any():
		if verbose:
			print "%d remaining pixels"%np.sum(truth)
			print "ipix : %d"%ipix
		for n in hp.get_all_neighbours(nside, ipix):
			if n == -1: ### neighbour doesn't exist
				pass
			elif boarder_truth[n]: ### neighbour is in boarder
				if verbose:
					print "\t%d in boarder"%n
				if truth[n]: ### pixel has not been visited
					if verbose:
						print "\t\thas not been visited"
					line.append( n )
					ipix = n
					truth[n] = False
					visit[n] = True
					break
				else: ### pixel has been visited. End line and start another
					if verbose:
						print "\t\thas been visited"
					line.append( n )
					lines.append( line )
					truth[n] = False
					visit[n] = True
					### find a new starting point!
					for _ipix in pix[visit]: ### all pixels we've visited
						for _n in hp.get_all_neighbours(nside, _ipix):
							if truth[_n]: ### neighbours a pixel we haven't seen
								if verbose:
									print "\t\t\tnew spur at %d"%_n
								line = [_ipix, _n]
								truth[_n] = False
								visit[_n] = True
								ipix = _n
								break
						else: ### didn't find any new spurs starting at _ipix, continue
							continue
						break ### we did find a new spur!
					else: ### didn't find any new spurs from any pixel we have visited
						if verbose:
							print "\t\t\tno new spur found"
						if  truth.any(): ### there are still pixels to be found
							ipix = pix[truth][0]
							if verbose:
								print "\t\t\tnew ring at %d"%ipix
							truth[ipix] = False
							visit[ipix] = True
							line = [ipix]			
					break
			else: ### neighbour is not in boarder
				if verbose:
					print "\t%d not in boarder"%n
		else: ### no neighbours are in boarder_truth. How did we get to this pixel?
			raise StandardError, "no neighbours aroudn %d found in boarder? How did we get to this pixel?"%ipix
			### just end the line and start another

	### check that we've visited all pixels
	if (visit != boarder_truth).any():
		raise StandardError, "visit != boarder_truth. Somehow we missed pixels?"

	### close the remaining line
	for n in hp.get_all_neighbours(nside, ipix):
		if n == -1:
			pass
		elif boarder_truth[n]:
			line.append( n ) ### close the line
			break
	else:
		raise StandardError, "hanging contour line ending at %d. How did we get to this pixel?"%ipix
	lines.append( line )

	### transform pixel numbers into coords for plotting
	pos = []
	for line in lines:
		theta, phi = hp.pix2ang(nside, line)
		pos.append( (list(theta), list(phi)) )


	### remove spurious lines? 
	### these may be caused by cutting a corner and then coming back and going the other way around.
	### a characterisitc would be that there are 3 points in the line, and all 3 points are neighbours of one another
	return pos			



