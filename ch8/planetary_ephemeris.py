"""
Algorithm 8.1 (Curtis, Orbital Mechanics for Engineering Students)
Calculation of the state vector of a planet at a given epoch.

Table 8.1 data from Standish et al. (1992).
Valid for years 1800 to 2050.

Units:
    a       - semimajor axis         (km)
    e       - eccentricity           (-)
    incl    - inclination            (deg)
    RA      - right ascension of ascending node (deg)
    w_hat   - longitude of perihelion (deg)
    L       - mean longitude        (deg)
    h       - specific angular momentum (km^2/s)
    r       - heliocentric position vector (km)
    v       - heliocentric velocity vector (km/s)
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tables import MU_SUN, AU, PLANET_NAMES, TABLE_8_1, TABLE_8_1_RATES

# ─────────────────────────────────────────────────────────────────────────────
# Julian day numbers for the valid date range of Table 8.1 (1800–2050)
# ─────────────────────────────────────────────────────────────────────────────
JD_1800 = 2378496.5
JD_2050 = 2469807.5

J2000_ELEMENTS = TABLE_8_1
CENT_RATES     = TABLE_8_1_RATES


# ─────────────────────────────────────────────────────────────────────────────
# Helper: reduce angle to [0, 360) degrees
# ─────────────────────────────────────────────────────────────────────────────
def zero_to_360(x: float) -> float:
    """Reduce angle x (degrees) to the range [0, 360)."""
    x = x % 360.0
    if x < 0:
        x += 360.0
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Julian day number at 0h UT  (Eq. 5.48, Curtis)
# ─────────────────────────────────────────────────────────────────────────────
def J0(year: int, month: int, day: int) -> float:
    """
    Julian day number at 0 hr UT for a given calendar date.
    Valid for years 1901–2099 (Eq. 5.48).
    """
    return (367 * year
            - int(7 * (year + int((month + 9) / 12)) / 4)
            + int(275 * month / 9)
            + day
            + 1721013.5)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: solve Kepler's equation  E - e*sin(E) = M  (radians)
# ─────────────────────────────────────────────────────────────────────────────
def kepler_E(e: float, M: float, tol: float = 1e-10) -> float:
    """
    Solve Kepler's equation for eccentric anomaly E (radians).

    Parameters
    ----------
    e : eccentricity
    M : mean anomaly (radians)
    tol : convergence tolerance

    Returns
    -------
    E : eccentric anomaly (radians)
    """
    # Initial guess
    E = M + e * math.sin(M) * (1.0 + e * math.cos(M))

    for _ in range(300):
        dE = (M - E + e * math.sin(E)) / (1.0 - e * math.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


# ─────────────────────────────────────────────────────────────────────────────
# Helper: state vector from classical orbital elements  (Algorithm 4.2)
# ─────────────────────────────────────────────────────────────────────────────
def sv_from_coe(h: float, e: float, RA: float, incl: float,
                w: float, TA: float) -> tuple:
    """
    Compute heliocentric state vector from classical orbital elements.
    All angles in radians.

    Parameters
    ----------
    h    : specific angular momentum (km^2/s)
    e    : eccentricity
    RA   : right ascension of ascending node (rad)
    incl : inclination (rad)
    w    : argument of perihelion (rad)
    TA   : true anomaly (rad)

    Returns
    -------
    r_vec : position vector in heliocentric equatorial frame (km)
    v_vec : velocity vector in heliocentric equatorial frame (km/s)
    """
    mu = MU_SUN

    # Position and velocity in perifocal frame
    r_p = (h**2 / mu) * (1.0 / (1.0 + e * math.cos(TA))) * np.array([
        math.cos(TA),
        math.sin(TA),
        0.0
    ])
    v_p = (mu / h) * np.array([
        -math.sin(TA),
        e + math.cos(TA),
        0.0
    ])

    # Rotation matrix: perifocal → geocentric equatorial
    # Q = R3(-RA) * R1(-incl) * R3(-w)
    cRA, sRA = math.cos(RA), math.sin(RA)
    ci,  si  = math.cos(incl), math.sin(incl)
    cw,  sw  = math.cos(w), math.sin(w)

    Q = np.array([
        [-sRA*ci*sw + cRA*cw,  -sRA*ci*cw - cRA*sw,  sRA*si],
        [ cRA*ci*sw + sRA*cw,   cRA*ci*cw - sRA*sw, -cRA*si],
        [          si*sw,                si*cw,        ci   ],
    ])

    r_vec = Q @ r_p
    v_vec = Q @ v_p
    return r_vec, v_vec


# ─────────────────────────────────────────────────────────────────────────────
# Subfunctions: planetary elements from Table 8.1
# ─────────────────────────────────────────────────────────────────────────────
def planetary_elements(planet_id: int):
    """
    Return J2000 orbital elements and centennial rates for a planet,
    with units converted to km and degrees.

    Parameters
    ----------
    planet_id : int, 1=Mercury … 9=Pluto

    Returns
    -------
    J2000_coe : np.ndarray, shape (6,)
        [a(km), e, i(deg), Omega(deg), omega_hat(deg), L(deg)]
    rates : np.ndarray, shape (6,)
        [a_dot(km/Cy), e_dot(1/Cy), i_dot(deg/Cy), Omega_dot(deg/Cy),
         omega_hat_dot(deg/Cy), L_dot(deg/Cy)]
    """
    idx = planet_id - 1
    J2000_coe = J2000_ELEMENTS[idx, :].copy()
    rates     = CENT_RATES[idx, :].copy()

    # Convert semimajor axis from AU to km
    J2000_coe[0] *= AU
    rates[0]     *= AU

    # Convert angular rates from arcseconds/Cy to degrees/Cy
    rates[2:6] /= 3600.0

    return J2000_coe, rates


# ─────────────────────────────────────────────────────────────────────────────
# Main function: Algorithm 8.1
# ─────────────────────────────────────────────────────────────────────────────
def planet_elements_and_sv(planet_id: int,
                            year: int, month: int, day: int,
                            hour: int = 0, minute: int = 0, second: float = 0.0):
    """
    Compute the heliocentric orbital elements and state vector of a planet
    at a given date and universal time (Algorithm 8.1, Curtis).

    Table 8.1 is valid for years 1800–2050. An error is raised if the
    requested date falls outside this range.

    Parameters
    ----------
    planet_id : int
        1=Mercury, 2=Venus, 3=Earth, 4=Mars, 5=Jupiter,
        6=Saturn, 7=Uranus, 8=Neptune, 9=Pluto
    year, month, day : int
        Calendar date (year 1901–2099 for the J0 formula).
    hour, minute : int
        Universal time.
    second : float
        Seconds of UT.

    Returns
    -------
    coe : dict
        Orbital elements at the requested epoch:
            h        – specific angular momentum (km^2/s)
            e        – eccentricity
            RA       – right ascension of ascending node (deg)
            incl     – inclination (deg)
            w        – argument of perihelion (deg)
            TA       – true anomaly (deg)
            a        – semimajor axis (km)
            w_hat    – longitude of perihelion (deg)
            L        – mean longitude (deg)
            M        – mean anomaly (deg)
            E        – eccentric anomaly (deg)
    r : np.ndarray, shape (3,)
        Heliocentric position vector (km).
    v : np.ndarray, shape (3,)
        Heliocentric velocity vector (km/s).
    jd : float
        Julian day number of the requested date/time.
    """
    if planet_id not in PLANET_NAMES:
        raise ValueError(f"planet_id must be 1–9, got {planet_id}.")

    # ── Step 1: Julian day number (Eqs. 5.47–5.48) ──────────────────────────
    j0 = J0(year, month, day)
    ut = (hour + minute / 60.0 + second / 3600.0) / 24.0
    jd = j0 + ut

    # ── Validity check: Table 8.1 is accurate for 1800–2050 ─────────────────
    if jd < JD_1800 or jd > JD_2050:
        raise ValueError(
            f"Requested date (JD={jd:.3f}) is outside the valid range of "
            f"Table 8.1 (1 Jan 1800 – 31 Dec 2050, "
            f"JD {JD_1800:.1f} – {JD_2050:.1f})."
        )

    # ── Step 2: Julian centuries since J2000 (Eq. 8.104a) ───────────────────
    t0 = (jd - 2451545.0) / 36525.0

    # ── Step 3: Orbital elements at JD (Eq. 8.104b) ─────────────────────────
    J2000_coe, rates = planetary_elements(planet_id)
    elements = J2000_coe + rates * t0

    a     = elements[0]
    e     = elements[1]
    incl  = elements[2]                          # deg (not angle-wrapped)
    RA    = zero_to_360(elements[3])             # deg
    w_hat = zero_to_360(elements[4])             # deg
    L     = zero_to_360(elements[5])             # deg
    w     = zero_to_360(w_hat - RA)              # argument of perihelion, deg
    M     = zero_to_360(L - w_hat)              # mean anomaly, deg

    # ── Specific angular momentum (Eq. 2.61) ─────────────────────────────────
    h = math.sqrt(MU_SUN * a * (1.0 - e**2))

    # ── Eccentric anomaly via Kepler's equation (Algorithm 3.1) ──────────────
    E_rad = kepler_E(e, math.radians(M))        # radians
    E_deg = math.degrees(E_rad)

    # ── True anomaly (Eq. 3.10) ──────────────────────────────────────────────
    TA_rad = 2.0 * math.atan2(
        math.sqrt(1.0 + e) * math.sin(E_rad / 2.0),
        math.sqrt(1.0 - e) * math.cos(E_rad / 2.0)
    )
    TA_deg = zero_to_360(math.degrees(TA_rad))

    # ── State vector via Algorithm 4.2 ───────────────────────────────────────
    r, v = sv_from_coe(
        h,
        e,
        math.radians(RA),
        math.radians(incl),
        math.radians(w),
        TA_rad,
    )

    coe = {
        "h":     h,
        "e":     e,
        "RA":    RA,
        "incl":  incl,
        "w":     w,
        "TA":    TA_deg,
        "a":     a,
        "w_hat": w_hat,
        "L":     L,
        "M":     M,
        "E":     E_deg,
    }

    return coe, r, v, jd


# ─────────────────────────────────────────────────────────────────────────────
# Example  (replicates Example 8.7 from the textbook: Earth on 27 Aug 2003)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    planet_id = 3          # Earth
    year, month, day = 2003, 8, 27
    hour, minute, second = 12, 0, 0

    coe, r, v, jd = planet_elements_and_sv(
        planet_id, year, month, day, hour, minute, second
    )

    planet_name = PLANET_NAMES[planet_id]

    print("=" * 52)
    print(" Algorithm 8.1 – Planetary State Vector")
    print("=" * 52)
    print(f"\n Input data:")
    print(f"   Planet : {planet_name}")
    print(f"   Date   : {year}-{month:02d}-{day:02d}  {hour:02d}:{minute:02d}:{second:05.2f} UT")
    print(f"\n   Julian day : {jd:.3f}")

    print(f"\n Orbital elements:")
    print(f"   Angular momentum  h     (km^2/s) = {coe['h']:.4e}")
    print(f"   Eccentricity      e              = {coe['e']:.7f}")
    print(f"   RAAN              Ω       (deg)  = {coe['RA']:.3f}")
    print(f"   Inclination       i       (deg)  = {coe['incl']:.6f}")
    print(f"   Arg. perihelion   ω       (deg)  = {coe['w']:.3f}")
    print(f"   True anomaly      ν       (deg)  = {coe['TA']:.3f}")
    print(f"   Semimajor axis    a        (km)  = {coe['a']:.5e}")
    print(f"   Long. perihelion  ω̃       (deg)  = {coe['w_hat']:.3f}")
    print(f"   Mean longitude    L       (deg)  = {coe['L']:.3f}")
    print(f"   Mean anomaly      M       (deg)  = {coe['M']:.3f}")
    print(f"   Eccentric anomaly E       (deg)  = {coe['E']:.3f}")

    print(f"\n State vector:")
    print(f"   Position (km)  = [{r[0]:.5e}  {r[1]:.5e}  {r[2]:.5e}]")
    print(f"   |r|       (km) = {np.linalg.norm(r):.5e}")
    print(f"   Velocity (km/s)= [{v[0]:.5f}  {v[1]:.5f}  {v[2]:.8f}]")
    print(f"   |v|     (km/s) = {np.linalg.norm(v):.4f}")
    print("=" * 52)