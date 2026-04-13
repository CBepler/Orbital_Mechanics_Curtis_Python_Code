import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gef_to_pf import get_transformation_matrix
from coe_from_sv import coe_from_sv


def get_geo_to_cw_dcm(RA_deg, incl_deg, w_deg, TA_deg):
    """
    Build the 3x3 direction cosine matrix from the geocentric equatorial (GEO)
    frame to the Clohessy-Wiltshire / LVLH frame.

    The LVLH frame is defined at a point on an elliptical orbit:
        x: radial outward (along position vector)
        y: along-track (perpendicular to x in the orbital plane)
        z: orbit normal (along angular momentum vector)

    Args:
        RA_deg   (float): Right ascension of ascending node (degrees)
        incl_deg (float): Inclination (degrees)
        w_deg    (float): Argument of perigee (degrees)
        TA_deg   (float): True anomaly (degrees)

    Returns:
        np.array: 3x3 DCM such that v_cw = DCM @ v_geo
    """
    # Perifocal-to-GEO transformation matrix
    T = get_transformation_matrix(RA_deg, incl_deg, w_deg)

    # GEO-to-perifocal is the transpose
    Q_geo_to_pf = T.T

    # Perifocal-to-LVLH is a z-axis rotation by true anomaly
    theta = np.radians(TA_deg)
    R3 = np.array([[ np.cos(theta),  np.sin(theta), 0],
                   [-np.sin(theta),  np.cos(theta), 0],
                   [     0,               0,         1]])

    return R3 @ Q_geo_to_pf


def geo_to_cw(vec_geo, RA_deg, incl_deg, w_deg, TA_deg):
    """
    Transform a vector from the geocentric equatorial (GEO) frame
    to the Clohessy-Wiltshire / LVLH frame for an elliptical orbit.

    Args:
        vec_geo  (np.array): 3-vector in GEO coordinates
        RA_deg   (float):    Right ascension of ascending node (degrees)
        incl_deg (float):    Inclination (degrees)
        w_deg    (float):    Argument of perigee (degrees)
        TA_deg   (float):    True anomaly (degrees)

    Returns:
        np.array: 3-vector in CW/LVLH coordinates
    """
    dcm = get_geo_to_cw_dcm(RA_deg, incl_deg, w_deg, TA_deg)
    return dcm @ vec_geo


def geo_to_cw_from_sv(vec_geo, R, V):
    """
    Transform a vector from GEO to CW/LVLH frame, deriving the frame
    orientation from the target spacecraft's state vector.

    Args:
        vec_geo (np.array): 3-vector in GEO coordinates
        R (np.array):       Position vector of target in GEO (km)
        V (np.array):       Velocity vector of target in GEO (km/s)

    Returns:
        np.array: 3-vector in CW/LVLH coordinates
    """
    coe = coe_from_sv(R, V)
    _, _, RA, incl, w, TA, _ = coe

    return geo_to_cw(vec_geo,
                     np.degrees(RA),
                     np.degrees(incl),
                     np.degrees(w),
                     np.degrees(TA))


if __name__ == "__main__":
    R = np.array([-4430.4, 3669.8, 3267.3])
    V = np.array([-5.1969, -5.7334, -0.60712])

    R_cw = geo_to_cw_from_sv([0.0806, -0.0679, -0.06262], R, V)
    print(R_cw)

