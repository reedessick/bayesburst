#!/usr/bin/python
"""
author: R. Essick (ressick@mit.edu) 
date: 20 July 2012

 this module written to help automate the process of estimating pdfs and cdfs from a set of observations.
 we implement several simple algorithms, all based on the concept of kernal estimation. They are:
  k-th nearest neighbor volume estimation
    takes the volume of a sphere with radius equal to the distance from an observed point to the k-th nearest observed point as a measure of the probability density : p ~ k / V
  fixed bandwidth gaussian kernel density estimation
  variabl bandwidth gaussian kernel density estimation
    ballooning 
      bandwidth depends on the evaluation point and is uniform across all elements of the observation set
    point-wise 
      a different bandwidth is assigned to each element of the observed set and is used for all evaluation points
    k-th nearest neighbor 
      a different bandwidth is assigned to each element of observed, and is the distance to the k-th nearest neighbor in the observed set. These are used for all evaluation points
  numerical estimation of cdf (step-like), smoothed with a spline intpolation, then differentiated to obtain a pdf

#######################################
Notes about using these algorithms. 
#######################################

k_th_neighbor(eval, observ, k)
  This algorithm appears to be subject to a lot of noise, even when k is a substantial fraction of the sample set. We recommend using a large k (> 50% sample set) and using caution.

k_th_neighbor_gaussian_kde(eval, observ, k)
  Like k_th_neighbor(), this algorithm can be subject to noise when k is small. However, the gaussian kernal tends to smooth out some of the sharper fluctuations.

fixed_bandwidth_gaussian_kde(eval, observ, v=)
  This method appears to be relatively robust, although a poor choice of varience (v) can lead to a poor kde. Currently, we don't know if there is an optimal choice for v, so the user may have to run a parameter search.

ballooning_gaussian_kde(eval, observ)
  This algorithm seems computes an estimated bandwidth for the gaussian_kde based on the points in eval and a pilot distribution. The performance depends strongly on the pilot distribution, so we suggest iteratively running this method, feeding results back in as the pilot distribution, and using a fine sampling of the pilot distribution. In particular, when pilot_pdf vanishes, the algorithm seems to seriously under-perform.

point_wise_gaussian_kde(eval, observ, smooth=)
  This algorithm computes a bandwidth for each point in observ that is used for every evaluation point. It is also sensitive to the pilot distribution used, although it appears to be more robust than ballooning_gaussian_kde() in 1-dimension. We suggest iteration and fine sampling as well. Furthermore, this algorithm retains a free parameter (smooth), which sets the over-all scale of the gaussian kernel's used. Essentially, it scales each point_wise bandwidth estimate. Currently, we don't know if there is an optimal choice for this scale, so users may have to investigate what works best for them.

num_smoothed_cdf_to_pdf(eval, observ, smooth=)
  This algorithm appears to be extremely unstable if (smooth) is set too small. Essentially, in the limit of small (smooth), the pdf will resemble a sum of delta functions. We therefore suggest smooth != 0. Currently, we don't know if there is an optimal choice for smooth, so the user will have to investigate.
  This algorithm also has a problem with it's interpolation that causes it to return 'nan' for many of the requested points. It generally performs much worse than the other algorithms, so this has been left as is with the hope that not many people will rely on it.

There are also some problems with interpolating curves between a set of sample points. In particular, most of these methods use a spline interpolation so that there is a well defined derivative (and second derivative) at each point along the curve. However, there can be problems with this interpolation method, most often relating to negative values for pdf's. For the most part, the pdf's are explicitly forced to be non-negative, but the problem persists.

Similarly, the normalization of these pdf's is only approximate. If you desire a pdf that integrates to unity exactly, you may want to re-normalize these estimates.

########################################
Notes on the use of this module
########################################
 Specific instructions and algorithmic descriptions are included in the comments at the head of each method. As a rule of thumb, each algorithm requires two sets of points, eval and observ. The method will return a list corresponding to the pdf (estimated from the points in observ) evaluated at the points in eval. Importantly, even if eval is an int or float, the method will return a list.

"""

import numpy as np
import scipy.interpolate
import scipy.special

def __Nd_gaussian_kernel(x1, x2, v):
  # the basic gaussian kernel. We define this for an arbitrary number of dimensions and an arbitrary covarience matrix (v)
  # for one dimensional kernels, v is equivalent to the varience of the distribution
  # x1, x2 can be ints, floats or array-like structures of ints, floats
  # s can be an int, float, or a matrix-like structure corresponding to a general covarience matrix
  # returns exp( - (1/2) * transpose(x1-x2) * inv(s) * (x1-x2) ) (generalized mulitvariate gaussian

  # determine the type of input data
  if isinstance(x1, (int, float)) or isinstance(x2, (int, float)):
    # 1-D kernel
    if isinstance(x1, (int,float)) and isinstance(x2, (int, float)) and isinstance(v, (int, float)):
      if v == 0:
        raise ValueError, "cannot have a vanishing standard deviation"
      else:
        return np.exp(-(x1-x2)**2/(2.*v))
    else:
      raise ValueError, "dimensions of x1, x2, v do not agree"
  else:
    # multi-dimensional kernel
    if len(x1) != len(x2):
      raise ValueError, "dimensions of x1, x2 do not agree"
    elif not isinstance(v, (int, float)) and (len(x1) != len(v) or len(x1) != len(v[0])):
      raise ValueError, "dimensions of s are not compatible with the dimensions of x1, x2"
    # uniform standard deviation across all dimensions
    elif isinstance(v, (int, float)):
      return np.exp(-sum([(x1[i] - x2[i])**2. for i in range(len(x1))]) / (2.*v))
    else:
      v = np.matrix(v)
      try:
        inverse_v = v.getI()
        x1=np.matrix(x1)
        x2=np.matrix(x2)
        z = x1 - x2 # difference between vectors x1 and x2
        zT = z.transpose() # transpose of z
        return np.exp(- (1/2.)*zT*inverse_v*z)
      except:
        raise ValueError, "v is not an invertible matrix"    

def fixed_bandwidth_gaussian_kde(eval, observ, v=0.01):
  # computes the fixed bandwidth gaussian kde for each point in array-like eval (evaluation points), summing over all points in array-like observ (observed points from sample set). 
  # Applies a fixed standard deviation/covariance matrix to all kernel estimations.
  # eval is an int, float or a list of ints, floats corresponding to the points at which the kde will be evaluated
  # observ is a list of ints, floats corresponding to the observed data from which the gaussian kde is estimated
  # v is taken to be the covarience matrix of the gaussian kde. This is equivalent to the varience in a Univariate gaussian

  # set up eval, observ, n_dims
  if isinstance(eval, (int, float)):
    # only one evaluation point
    eval = [eval]
    n_dims = 1
  elif isinstance(eval[0], (int, float)):
    n_dims = 1
  else:
    n_dims = len(eval[0]) # the number of dimensions of each element of eval
  
  if isinstance(observ, (int, float)):
    observ = [observ]

  # build kde estimates at each point in eval
  kde = []
  # we normalize each kernal to unity, and normalize the kde so it's integral is unity
  for e in eval:
    kde += [sum([__Nd_gaussian_kernel(e, o, v) for o in observ]) / (len(observ) * (2.*np.pi)**(n_dims/2.) * np.linalg.det(np.matrix(v))**(1/2.) )]
  
  return np.array(kde)


def ballooning_gaussian_kde(eval, observ, pilot_interp='NONE', pilot_x='NONE', pilot_y='NONE', s=0.1):
  # computes the variable bandwidth gaussian kde for each element of eval based on the elements of observ
  # the algorithm we use estimates the optimal bandwidth (s) based on the evaluation point (eval). 
  # This is computed from a 'pilot estimate' of the pdf, which can be supplied in the following ways (ordered by preference):
  #  1) pilot_interp : must be a scipy.interp1d callable object
  #  2) pilot_x and pilot_y : must be equal length vectors corresponding to a sampling of the pilot pdf
  #  3) s : a float, which will be used in the fixed_bandwidth_gaussian_kde(~,~,s**2) to estimpate the pilot pdf
  #         s acts as the standard deviation of a univariate gaussian kernel
  # eval is an int, float or a list of ints, floats
  # observ is a list of ints, floats
  # pilot_interp is an instance of scipy.interpolate.interp1d or scipy.interpolate.InterpolatedUnivariateSpline
  # pilot_x is a list of ints, floats
  # pilot_y is a list of ints, floats
  # s is an int, float

  # IMPORTANTLY
  #  this in only implemented for 1-D

  # set up the pilot pdf
  if pilot_interp == 'NONE': # pilot_interp not supplied
    if (pilot_x == 'NONE') and (pilot_y == 'NONE'): # neither pilot_x nor pilot_y supplied
      pilot_x = np.linspace(min(observ), max(observ), np.ceil(10*(max(observ) - min(observ))/s))
      pilot_y = fixed_bandwidth_gaussian_kde(pilot_x, observ, v=s**2)
    elif not (pilot_x != 'NONE' and pilot_y != 'NONE'): # pilot_x or pilot_y supplied, but not both
      raise ValueError, "one of pilot_x and pilot_y supplied. please supply both"
    # both pilot_x and pilot_y are supplied, or have now been defined
    if len(pilot_x) != len(pilot_y):
      raise ValueError, "pilot_x and pilot_y are not the same length"
    else:
        pilot_interp = scipy.interpolate.InterpolatedUnivariateSpline(pilot_x, pilot_y, k=3) # we automatically look for a cubic spline
  else: # pilot_interp is supplied
        # we require pilot_interp to be an intsance of scipy.interpolate.InterpolatedUnivariateSpline or scipy.interpolate.interp1d
    if not isinstance(pilot_interp, scipy.interpolate.InterpolatedUnivariateSpline):
      if isinstance(pilot_interp, scipy.interpolate.interp1d):
        pilot_interp = scipy.interpolate.InterpolatedUnivariateSpline(pilot_interp.__dict__['x'], pilot_interp.__dict__['y'], k=3) # if the object supplied is an instance of scipy.interpolate.interp1d, we convert it
      else:
        raise ValueError, "data structure 'pilot_interp' not understood"

  if isinstance(eval, (int, float)):
    eval = [eval]

  kde = []
  for e in eval:
    s_optimal = __ballooning_optimal_s(e, len(observ), pilot_interp) # computes the optimal standard deviation for a univariate gaussian kernel
    if np.isnan(s_optimal):
      s_optimal == 5*(max(observ) - min(observ))
      kde += [sum([__Nd_gaussian_kernel(e, o, s_optimal**2) for o in observ]) / ((2.*np.pi)**0.5 * len(observ) * s_optimal) ]
    elif s_optimal == 0:
      kde += [0]
    else:    
      kde += [sum([__Nd_gaussian_kernel(e, o, s_optimal**2) for o in observ]) / ((2.*np.pi)**0.5 * len(observ) * s_optimal) ]

  return np.array(kde)


def __ballooning_optimal_s(eval, n_samples, pdf):
  # computes the optimal choice of standard deviation for a ballooning gaussian kde based on a pilot distribution (pdf), the number of sample points (n_samples), and an evaluation point (eval)
  # the assymptotically optimal choice for s(eval) is:
  #  s(eval) = ( ((2!)**2 * pdf(eval) * \int{ K(x)**2 dx}) / (2*2*n_samples * (d^2 pdf(eval) /dx^2)^2) )^(1/5)
  # we have to evaluate the second derivative of the pilot distribution, so it should be finely sampled
  # eval is an int, float
  # n_samples is an int
  # pdf is an instance of scipy.interpolate.InterpolatedUnivariateSpline

  if pdf(eval, nu=2)[0] != 0:
    return ((16*np.sqrt(np.pi)*max([pdf(eval)[0],0])) / (4*n_samples*pdf(eval, nu=2)[0]**2.))**(1/5.) # the maximum statement forces pdf(eval) to be non-negative
  else:
    raise ValueError, "second derivative of the pilot_interp pdf vanishes. Optimal bandwidth is not defined"


def point_wise_gaussian_kde(eval, observ, scale=0.1, pilot_interp='NONE', pilot_x='NONE', pilot_y='NONE', s=0.1):
  # computes the variable bandwidth gaussian kde for every point in eval based on the points in observ
  # this algorithm computes an appropriate bandwidth for each point in observ
  # This is computed from a 'pilot estimate' of the pdf, which can be supplied in the following ways:
  #  1) pilot_interp : must be a scipy.interp1d callable object
  #  2) pilot_x and pilot_y : must be equal length vectors corresponding to a sampling of the pilot pdf
  #  3) s : a float, which will be used in the fixed_bandwidth_gaussian_kde(~,~,s**2) to estimpate the pilot pdf
  #         s acts as the standard deviation of a univariate gaussian kernel
  # eval is an int, float or a list of ints, floats
  # observ is a list of ints, floats
  # scale is an int, float and is the universal scaling constant for __point_wise_optimal_s
  # pilot_interp is an instance of scipy.interpolate.interp1d or scipy.interpolate.InterpolatedUnivariateSpline
  # pilot_x is a list of ints, floats
  # pilot_y is a list of ints, floats
  # s is an int, float

  # IMPORTANTLY
  #  this in only implemented for 1-D

  # set up the pilot pdf
  if pilot_interp == 'NONE': # pilot_interp not supplied
    if (pilot_x == 'NONE') and (pilot_y == 'NONE'): # neither pilot_x nor pilot_y supplied
      pilot_x = np.linspace(min(observ), max(observ), np.ceil(10*(max(observ) - min(observ))/s)+4)
      pilot_y = fixed_bandwidth_gaussian_kde(pilot_x, observ, v=s**2)
    elif not (pilot_x != 'NONE' and pilot_y != 'NONE'): # pilot_x or pilot_y supplied, but not both
      raise ValueError, "one of pilot_x and pilot_y supplied. please supply both"
    # both pilot_x and pilot_y are supplied, or have now been defined
    if len(pilot_x) != len(pilot_y):
      raise ValueError, "pilot_x and pilot_y are not the same length"
    else:
        pilot_interp = scipy.interpolate.InterpolatedUnivariateSpline(pilot_x, pilot_y, k=3) # we automatically look for a cubic spline
  else: # pilot_interp is supplied
        # we require pilot_interp to be an intsance of scipy.interpolate.InterpolatedUnivariateSpline or scipy.interpolate.interp1d
    if not isinstance(pilot_interp, scipy.interpolate.InterpolatedUnivariateSpline):
      if isinstance(pilot_interp, scipy.interpolate.interp1d):
        pilot_interp = scipy.interpolate.InterpolatedUnivariateSpline(pilot_interp.__dict__['x'], pilot_interp.__dict__['y'], k=3) # if the object supplied is an instance of scipy.interpolate.interp1d, we convert it
      else:
        raise ValueError, "data structure 'pilot_interp' not understood"

  if isinstance(eval, (int, float)):
    eval = [eval]

  kde = []
  s_optimal = [__point_wise_optimal_s(o, scale, pilot_interp) for o in observ] # we base the optimal bandwidth off the observed observations
  len_observ = len(observ)
  for e in eval:
    kde.append( sum([__Nd_gaussian_kernel(e, o, s_optimal[ind]**2) / ((2.*np.pi)**0.5 * len_observ * s_optimal[ind]) for ind, o in enumerate(observ)]) )

  return np.array(kde)

def __point_wise_optimal_s(eval, scale, pdf):
 # computes the optimal choice of standard deviation for a point-wise gaussian kde based on a pilot distribution (pdf), an evaluation point (eval), and a universal scale parameter (scale)
  # the assymptotically optimal choice for s(eval) is:
  #  s(eval) = scale*pdf(eval)**(-1/2.)
  # eval is an int, float
  # n_samples is an int
  # pdf is an instance of scipy.interpolate.InterpolatedUnivariateSpline

  ### support for different version of scipy?
  pdf_eval = pdf(eval)
  try:
    pdf_eval = pdf_eval[0]
  except TypeError:
    pass

  if pdf_eval > 0:
#    print pdf(eval)
    return scale*(pdf_eval)**(-0.5)
#    return scale*(max([pdf(eval)[0], 0]))**(-0.5)
  else:
    return scale
    raise ValueError, "pilot_interp pdf vanishes. Optimal bandwidth is not defined."
#    return False


def k_th_neighbor_gaussian_kde(eval, observ, k=1):
  # computes the k_th nearest neighbor variable bandwidth gaussian kde for each point in eval based on the points in observ
  # a different bandwidth is computed for each point in observ, which corresponds to the distance to the k-th nearet neighbor in observ
  # eval is an int, float or a list of ints, floats
  # observ is a list of ints, floats  
  # k is an int

  if len(observ)-1 < k:
    raise ValueError, "k > len(observ)-1, we cannot define the k-th nearest neighbor"

  if isinstance(eval, (int, float)):
    eval = [eval]

  kde = []
  for e in eval:
    kernel_sum = 0.0
    for o in observ:
      s = __distance_to_kth_neighbor(e, observ, k+1) # we look for the k+1-th neighbor, because o is included in observ
      kernel_sum += __Nd_gaussian_kernel(e, o, s**2) / ((2.*np.pi)**0.5 * s)
    kde += [kernel_sum / len(observ)]
  
  return kde

def __distance_to_kth_neighbor(o, observ, k):
  # coputes the distance to o's k-th nearest neighbor in observ
  # o is an int, float or a vector of ints, floats
  # observ is a list of int,s floats or a list of vectors of ints, floats
  # k is an int

  if isinstance(o, (int, float)) and isinstance(observ[0], (int, float)):
    observ = np.array(sorted(observ))

    # pull out a subset of glitches that are close to o
    close_to_o = []
    less_than_o = list(observ[np.nonzero(observ < o)[0],:])
    if len(less_than_o) < k:
      close_to_o += less_than_o
    else:
      close_to_o += less_than_o[-k:]
    greater_than_o = list(observ[np.nonzero(observ > o)[0],:])
    if len(greater_than_o) < k:
      close_to_o += greater_than_o
    else:
      close_to_o += greater_than_o[:k]

    return sorted([np.abs(o - obs) for obs in close_to_o])[k-1]

  elif len(o) == len(observ[0]):
    return sorted([np.sqrt(sum([(o[ind] - obs[ind])**2. for ind in range(len(o))])) for obs in observ])[k-1]

  else:
    raise ValueError, "length(o) != length(observ[0]). bad input data"
    return False

def k_th_neighbor(eval, observ, k):
  # computes an estimate for the pdf by assuming p(x) ~ k / V, where V is the volume of a (hyper-)sphere with radius equal to the distance to eval's k-th nearest neighbor in observ
  # eval is an int, float or a list of ints, floats
  # observ is a list of ints, floats
  # k is an int

  if len(observ)-1 < k:
    raise ValueError, "k > len(observ)-1, we cannot define the k-th nearest neighbor"

  if isinstance(eval, (int, float)):
    eval = [eval]
    n_dims = 1
  elif isinstance(eval[0], (int, float)):
    n_dims = 1
  else:
    n_dims = len(eval[0])

  kde = []
  for e in eval:
    kde += [(1.*k/len(observ))/(__Unit_sphere_volume(n_dims)*__distance_to_kth_neighbor(e, observ, k)**n_dims)]
  
  return kde

def __Unit_sphere_volume(dim):
  # returns the volume of a unit sphere in (dim) dimensions
  return np.pi**(dim/2.)/scipy.special.gamma(dim/2. + 1)

def num_smoothed_cdf_to_pdf(eval, observ, k=3, smooth=False, min_observ='NONE', max_observ='NONE'):
  # computes a numerical estimate of the cdf from observ, then fits a spline interpolant of order (k) to the cdf
  # we differentiate the spline to obtain an estimation of the pdf
  # eval is an int, float or a list of ints, floats corresponding to the points at which we will evaluate the pdf
  # observ is a list of ints, floats
  # k is the order of the spline we use to interpolate the cdf, default is 3
  # smooth is the smoothing parameter to be used in the spline fit (0 corresponds to strict interpolation)
  #
  # this method returns a list of pdf estimates corresonding to the points in eval, and also returns the spline interpolation of the cdf

  # IMPORTANTLY:
  #  this is only implemented for 1D

  # set up eval
  if isinstance(eval, (int, float)):
    eval = [eval]

  cdf_x = sorted(observ)

  # set up min_observ and max_observ
  if min_observ != 'NONE':
    if min_observ > min(observ):
      print "Bad min_observ value, we found a data point smaller than that"
      return False
    else:
      cdf_x = [min_observ] + cdf_x
  if max_observ != 'NONE':
    if max_observ < max(observ):
      print "Bad max_observ value, we found a data point larger than that"
      return False
    else:
      cdf_x = cdf_x + [max_observ]

  num_cdf = [float(i)/(len(cdf_x)-1) for i in range(len(cdf_x))] # a simple counting proceedure, and every point in this list corresponds to a point in observ
  spline_cdf = scipy.interpolate.InterpolatedUnivariateSpline(cdf_x, num_cdf, k=k)
  if smooth:
    spline_cdf.set_smoothing_factor(smooth)
    
  pdf = []
  for e in eval:
    pdf += [max(spline_cdf(e, nu=1)[0], 0)] # the first derivative of the interpolated cdf is an estimate of the pdf, and we demand that it is non-negative (which can  make us fudge our fit)

  return pdf, spline_cdf
