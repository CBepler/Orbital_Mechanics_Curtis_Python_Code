"""
Astronomical data tables from Curtis, Orbital Mechanics for Engineering Students.

Table 8.1  — Planetary orbital elements and centennial rates (Standish et al. 1992)
Table A.1  — Physical and orbital data for the sun, planets, and moon
Table A.2  — Gravitational parameter (mu) and sphere of influence (SOI) radius

Usage:
    from tables import TABLE_8_1, TABLE_8_1_RATES, TABLE_A1, TABLE_A2
    from tables import PLANET_NAMES, MU_SUN, AU
"""

from collections import namedtuple
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MU_SUN = 1.327124e11   # km^3/s^2
AU     = 149_597_871.0  # km per astronomical unit

PLANET_NAMES = {
    1: "Mercury",
    2: "Venus",
    3: "Earth",
    4: "Mars",
    5: "Jupiter",
    6: "Saturn",
    7: "Uranus",
    8: "Neptune",
    9: "Pluto",
}

# ─────────────────────────────────────────────────────────────────────────────
# Table 8.1  —  J2000 orbital elements (Standish et al. 1992)
# Valid for years 1800–2050
#
# Columns: [a (AU), e, i (deg), Omega (deg), omega_hat (deg), L (deg)]
# Rows: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto
#
# Omega    = right ascension of ascending node
# omega_hat = longitude of perihelion (= Omega + omega)
# L        = mean longitude (= omega_hat + M)
# ─────────────────────────────────────────────────────────────────────────────
TABLE_8_1 = np.array([
    # a (AU)        e            i (deg)   Omega (deg)  omega_hat(deg)  L (deg)
    [0.38709893,  0.20563069,  7.00487,   48.33167,   77.45645,  252.25084],  # Mercury
    [0.72333199,  0.00677323,  3.39471,   76.68069,  131.53298,  181.97973],  # Venus
    [1.00000011,  0.01671022,  0.00005,  -11.26064,  102.94719,  100.46435],  # Earth
    [1.52366231,  0.09341233,  1.85061,   49.57854,  336.04084,  355.45332],  # Mars
    [5.20336301,  0.04839266,  1.30530,  100.55615,   14.75385,   34.40438],  # Jupiter
    [9.53707032,  0.05415060,  2.48446,  113.71504,   92.43194,   49.94432],  # Saturn
    [19.19126393, 0.04716771,  0.76986,   74.22988,  170.96424,  313.23218],  # Uranus
    [30.06896348, 0.00858587,  1.76917,  131.72169,   44.97135,  304.88003],  # Neptune
    [39.48168677, 0.24880766, 17.14175,  110.30347,  224.06676,  238.92881],  # Pluto
])

# Centennial rates
# Columns: [a_dot (AU/Cy), e_dot (1/Cy), i_dot ("/Cy), Omega_dot ("/Cy),
#           omega_hat_dot ("/Cy), L_dot ("/Cy)]
TABLE_8_1_RATES = np.array([
    # a_dot        e_dot       i_dot ("/Cy) Omega_dot   ohat_dot     L_dot
    [ 0.00000066,  0.00002527,  -23.51,   -446.30,    573.57,  538101628.29],  # Mercury
    [ 0.00000092, -0.00004938,   -2.86,   -996.89,   -108.80,  210664136.06],  # Venus
    [-0.00000005, -0.00003804,  -46.94, -18228.25,   1198.28,  129597740.63],  # Earth
    [-0.00007221,  0.00011902,  -25.47,  -1020.19,   1560.78,   68905103.78],  # Mars
    [ 0.00060737, -0.00012880,   -4.15,   1217.17,    839.93,   10925078.35],  # Jupiter
    [-0.00301530, -0.00036762,    6.11,  -1591.05,  -1948.89,    4401052.95],  # Saturn
    [ 0.00152025, -0.00019150,   -2.09,  -1681.40,   1312.56,    1542547.79],  # Uranus
    [-0.00125196,  0.00002514,   -3.64,   -151.25,   -844.43,     786449.21],  # Neptune
    [-0.00076912,  0.00006465,   11.07,    -37.33,   -132.25,     522747.90],  # Pluto
])

# ─────────────────────────────────────────────────────────────────────────────
# Table A.1  —  Astronomical data for the sun, planets, and moon
#
# rotation_period_days : sidereal rotation period in days
#                        negative value indicates retrograde rotation
# sma_km               : semimajor axis of orbit (km); None for Sun
# orbit_period_days    : sidereal orbital period (days); None for Sun
# ─────────────────────────────────────────────────────────────────────────────
_BodyData = namedtuple('BodyData', [
    'radius_km',
    'mass_kg',
    'rotation_period_days',
    'equator_inclination_deg',
    'sma_km',
    'eccentricity',
    'orbit_inclination_deg',
    'orbit_period_days',
])

TABLE_A1 = {
    'Sun':     _BodyData(
        radius_km=696_000,
        mass_kg=1.989e30,
        rotation_period_days=25.38,
        equator_inclination_deg=7.25,
        sma_km=None,
        eccentricity=None,
        orbit_inclination_deg=None,
        orbit_period_days=None,
    ),
    'Mercury': _BodyData(
        radius_km=2_440,
        mass_kg=3.30e23,
        rotation_period_days=58.65,
        equator_inclination_deg=0.01,
        sma_km=57.91e6,
        eccentricity=0.2056,
        orbit_inclination_deg=7.00,
        orbit_period_days=87.97,
    ),
    'Venus':   _BodyData(
        radius_km=6_052,
        mass_kg=4.869e24,
        rotation_period_days=-243.0,   # retrograde
        equator_inclination_deg=177.4,
        sma_km=108.2e6,
        eccentricity=0.0067,
        orbit_inclination_deg=3.39,
        orbit_period_days=224.7,
    ),
    'Earth':   _BodyData(
        radius_km=6_378,
        mass_kg=5.974e24,
        rotation_period_days=23.9345 / 24.0,
        equator_inclination_deg=23.45,
        sma_km=149.6e6,
        eccentricity=0.0167,
        orbit_inclination_deg=0.0,
        orbit_period_days=365.256,
    ),
    'Moon':    _BodyData(
        radius_km=1_737,
        mass_kg=7.348e22,
        rotation_period_days=27.32,
        equator_inclination_deg=6.68,
        sma_km=384.4e3,
        eccentricity=0.0549,
        orbit_inclination_deg=5.145,
        orbit_period_days=27.322,
    ),
    'Mars':    _BodyData(
        radius_km=3_396,
        mass_kg=6.419e23,
        rotation_period_days=24.07 / 24.0,
        equator_inclination_deg=25.19,
        sma_km=227.9e6,
        eccentricity=0.0934,
        orbit_inclination_deg=1.850,
        orbit_period_days=1.881 * 365.25,
    ),
    'Jupiter': _BodyData(
        radius_km=71_490,
        mass_kg=1.899e27,
        rotation_period_days=9.925 / 24.0,
        equator_inclination_deg=3.13,
        sma_km=778.6e6,
        eccentricity=0.0484,
        orbit_inclination_deg=1.304,
        orbit_period_days=11.86 * 365.25,
    ),
    'Saturn':  _BodyData(
        radius_km=60_270,
        mass_kg=5.685e26,
        rotation_period_days=10.66 / 24.0,
        equator_inclination_deg=26.73,
        sma_km=1.433e9,
        eccentricity=0.0565,
        orbit_inclination_deg=2.485,
        orbit_period_days=29.46 * 365.25,
    ),
    'Uranus':  _BodyData(
        radius_km=25_560,
        mass_kg=8.683e25,
        rotation_period_days=17.24 / 24.0,
        equator_inclination_deg=97.77,
        sma_km=2.872e9,
        eccentricity=0.0460,
        orbit_inclination_deg=0.772,
        orbit_period_days=84.01 * 365.25,
    ),
    'Neptune': _BodyData(
        radius_km=24_760,
        mass_kg=1.024e26,
        rotation_period_days=16.11 / 24.0,
        equator_inclination_deg=28.32,
        sma_km=4.495e9,
        eccentricity=0.0113,
        orbit_inclination_deg=1.769,
        orbit_period_days=164.8 * 365.25,
    ),
    'Pluto':   _BodyData(
        radius_km=1_195,
        mass_kg=1.25e22,
        rotation_period_days=-6.387,   # retrograde
        equator_inclination_deg=122.5,
        sma_km=5.870e9,
        eccentricity=0.2444,
        orbit_inclination_deg=17.16,
        orbit_period_days=247.7 * 365.25,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Table A.2  —  Gravitational parameter (mu) and sphere of influence (SOI)
#
# mu_km3_s2 : gravitational parameter mu (km^3/s^2)
# soi_km    : sphere of influence radius (km); None for Sun (no SOI)
# ─────────────────────────────────────────────────────────────────────────────
_GravData = namedtuple('GravData', ['mu_km3_s2', 'soi_km'])

TABLE_A2 = {
    'Sun':        _GravData(mu_km3_s2=132_712_000_000, soi_km=None),
    'Mercury':    _GravData(mu_km3_s2=22_030,          soi_km=112_000),
    'Venus':      _GravData(mu_km3_s2=324_900,         soi_km=616_000),
    'Earth':      _GravData(mu_km3_s2=398_600,         soi_km=925_000),
    'Moon':       _GravData(mu_km3_s2=4_903,           soi_km=66_200),
    'Mars':       _GravData(mu_km3_s2=42_828,          soi_km=577_000),
    'Jupiter':    _GravData(mu_km3_s2=126_686_000,     soi_km=48_200_000),
    'Saturn':     _GravData(mu_km3_s2=37_931_000,      soi_km=54_800_000),
    'Uranus':     _GravData(mu_km3_s2=5_794_000,       soi_km=51_800_000),
    'Neptune':    _GravData(mu_km3_s2=6_835_100,       soi_km=86_600_000),
    'Pluto':      _GravData(mu_km3_s2=830,             soi_km=3_080_000),
}
