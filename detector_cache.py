### written by R.Essick (ressick@mit.edu)

usage = """ a holder for default/known detectors """

import utils
np = utils.np

#=================================================
# path to psd cache
#=================================================
path = __file__.strip(__name__+".py")

#=================================================
#
# DEFINE KNOWN DETECTORS
#
#=================================================
detectors = {}

default_psd = utils.PSD(np.array([0]), np.array([1]), kind="linear")

### design psd's
ligo_design_psd_file = path+'PSDs/aLIGO_design.txt'
ligo_design_psd_dat = np.genfromtxt(ligo_design_psd_file)
ligo_design_psd = utils.PSD(ligo_design_psd_dat[:,0], ligo_design_psd_dat[:,1]**2, kind="linear")

virgo_design_psd_file = path+'PSDs/aVirgo_design.txt'
virgo_design_psd_dat = np.genfromtxt(virgo_design_psd_file)
virgo_design_psd = utils.PSD(virgo_design_psd_dat[:,0], virgo_design_psd_dat[:,1]**2, kind="linear")

### Detector locations and orientations taken from Anderson, et all PhysRevD 63(04) 2003

__H_dr__ = np.array((-2.161415, -3.834695, +4.600350))*1e6/utils.c # sec
__H_nx__ = np.array((-0.2239, +0.7998, +0.5569))
__H_ny__ = np.array((-0.9140, +0.0261, -0.4049))
#LHO = Detector("LHO", __H_dr__, __H_nx__, __H_ny__, default_psd)
LHO = utils.Detector("H1", __H_dr__, __H_nx__, __H_ny__, ligo_design_psd)
detectors["H1"] = LHO

__L_dr__ = np.array((-0.074276, -5.496284, +3.224257))*1e6/utils.c # sec
__L_nx__ = np.array((-0.9546, -0.1416, -0.2622))
__L_ny__ = np.array((+0.2977, -0.4879, -0.8205))
#LLO = Detector("LLO", __L_dr__, __L_nx__, __L_ny__, default_psd)
LLO = utils.Detector("L1", __L_dr__, __L_nx__, __L_ny__, ligo_design_psd)
detectors["L1"] = LLO

__V_dr__ = np.array((+4.546374, +0.842990, +4.378577))*1e6/utils.c # sec
__V_nx__ = np.array((-0.7005, +0.2085, +0.6826))
__V_ny__ = np.array((-0.0538, -0.9691, +0.2408))
#Virgo = Detector("Virgo", __V_dr__, __V_nx__, __V_ny__, default_psd)
Virgo = utils.Detector("V1", __V_dr__, __V_nx__, __V_ny__, virgo_design_psd)
detectors["V1"] = Virgo

