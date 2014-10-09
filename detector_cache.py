### written by R.Essick (ressick@mit.edu)

usage = """ a holder for default/known detectors """

import copy
import utils
np = utils.np

#=================================================
# path to psd cache
#=================================================
path = __file__.strip(__name__+".py")

#=================================================
# known PSDs
#=================================================
psds = {}


default_psd = utils.PSD(np.array([0]), np.array([1]), kind="linear")
psds["default"] = default_psd

### design psd's
aligo_design_psd_file = path+'PSDs/aLIGO_design.txt'
aligo_design_psd_dat = np.genfromtxt(aligo_design_psd_file)
aligo_design_psd = utils.PSD(aligo_design_psd_dat[:,0], aligo_design_psd_dat[:,1]**2, kind="linear")
psds["aligo_design"] = aligo_design_psd

avirgo_design_psd_file = path+'PSDs/aVirgo_design.txt'
avirgo_design_psd_dat = np.genfromtxt(avirgo_design_psd_file)
avirgo_design_psd = utils.PSD(avirgo_design_psd_dat[:,0], avirgo_design_psd_dat[:,1]**2, kind="linear")
psds["avirgo_design"] = avirgo_design_psd

#=================================================
# known detectors
#=================================================
### Detector locations and orientations taken from Anderson, et all PhysRevD 63(04) 2003
detectors = {}

__H_dr__ = np.array((-2.161415, -3.834695, +4.600350))*1e6/utils.c # sec
__H_nx__ = np.array((-0.2239, +0.7998, +0.5569))
__H_ny__ = np.array((-0.9140, +0.0261, -0.4049))
LHO = utils.Detector("H1", __H_dr__, __H_nx__, __H_ny__, copy.deepcopy(aligo_design_psd))
detectors["H1"] = LHO

__L_dr__ = np.array((-0.074276, -5.496284, +3.224257))*1e6/utils.c # sec
__L_nx__ = np.array((-0.9546, -0.1416, -0.2622))
__L_ny__ = np.array((+0.2977, -0.4879, -0.8205))
LLO = utils.Detector("L1", __L_dr__, __L_nx__, __L_ny__, copy.deepcopy(aligo_design_psd))
detectors["L1"] = LLO

__V_dr__ = np.array((+4.546374, +0.842990, +4.378577))*1e6/utils.c # sec
__V_nx__ = np.array((-0.7005, +0.2085, +0.6826))
__V_ny__ = np.array((-0.0538, -0.9691, +0.2408))
Virgo = utils.Detector("V1", __V_dr__, __V_nx__, __V_ny__, copy.deepcopy(avirgo_design_psd))
detectors["V1"] = Virgo

