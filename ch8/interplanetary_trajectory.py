"""
Algorithm 8.2 (Curtis, Orbital Mechanics for Engineering Students)
Heliocentric spacecraft trajectory from planet 1 to planet 2.

Given departure and arrival epochs, returns:
  - planet 1 state vector at departure
  - planet 2 state vector at arrival
  - spacecraft heliocentric velocities at departure and arrival (from Lambert)
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ch5'))

from ch8.planetary_ephemeris import planet_elements_and_sv, PLANET_NAMES, MU_SUN
from ch5.lamberts import lambert
from coe_from_sv import coe_from_sv


def interplanetary(depart, arrive):
    """
    Parameters
    ----------
    depart, arrive : 7-tuples (planet_id, year, month, day, hour, minute, second)

    Returns
    -------
    planet1    : (R1, V1, jd1)     planet 1 heliocentric state at departure
    planet2    : (R2, V2, jd2)     planet 2 heliocentric state at arrival
    trajectory : (V1_sc, V2_sc)    spacecraft velocities at departure/arrival
    """
    _, R1, V1, jd1 = planet_elements_and_sv(*depart)
    _, R2, V2, jd2 = planet_elements_and_sv(*arrive)

    tof = (jd2 - jd1) * 86400.0
    V1_sc, V2_sc = lambert(R1, R2, tof, mu=MU_SUN)

    return (R1, V1, jd1), (R2, V2, jd2), (V1_sc, V2_sc)


# ─────────────────────────────────────────────────────────────────────────────
# Example 8.8: Earth → Mars, depart 1996-11-07, arrive 1997-09-12
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    depart = (3, 1996, 11,  7, 0, 0, 0)   # Earth
    arrive = (4, 1997,  9, 12, 0, 0, 0)   # Mars

    (R1, V1, jd1), (R2, V2, jd2), (V1_sc, V2_sc) = interplanetary(depart, arrive)

    coe_dep = coe_from_sv(R1, V1_sc, mu=MU_SUN)
    coe_arr = coe_from_sv(R2, V2_sc, mu=MU_SUN)
    h, e, RA, incl, w, TA_dep, a = coe_dep
    TA_arr = coe_arr[5]

    vinf_dep = V1_sc - V1
    vinf_arr = V2_sc - V2
    tof_days = jd2 - jd1
    period_days = 2 * math.pi / math.sqrt(MU_SUN) * a**1.5 / 86400.0

    deg = 180.0 / math.pi

    print("-" * 60)
    print(" Example 8.8  (Curtis, Algorithm 8.2)")
    print("-" * 60)
    print("\n Departure:")
    print(f"   Planet     : {PLANET_NAMES[depart[0]]}")
    print(f"   Date (UT)  : {depart[1]}-{depart[2]:02d}-{depart[3]:02d}  "
          f"{depart[4]:02d}:{depart[5]:02d}:{depart[6]:02d}")
    print(f"   Julian day : {jd1:.3f}")
    print(f"   Planet position (km)  = [{R1[0]:.5e} {R1[1]:.5e} {R1[2]:.5e}]")
    print(f"     |R1|    (km)        = {np.linalg.norm(R1):.5e}")
    print(f"   Planet velocity (km/s)= [{V1[0]:.4f} {V1[1]:.4f} {V1[2]:.4f}]")
    print(f"     |V1|    (km/s)      = {np.linalg.norm(V1):.4f}")
    print(f"   Spacecraft velocity   = [{V1_sc[0]:.4f} {V1_sc[1]:.4f} {V1_sc[2]:.6f}]")
    print(f"     |V1_sc| (km/s)      = {np.linalg.norm(V1_sc):.4f}")
    print(f"   v-infinity            = [{vinf_dep[0]:.4f} {vinf_dep[1]:.4f} {vinf_dep[2]:.6f}]")
    print(f"     |vinf|  (km/s)      = {np.linalg.norm(vinf_dep):.5f}")

    print(f"\n Time of flight = {tof_days:.0f} days")

    print("\n Arrival:")
    print(f"   Planet     : {PLANET_NAMES[arrive[0]]}")
    print(f"   Date (UT)  : {arrive[1]}-{arrive[2]:02d}-{arrive[3]:02d}  "
          f"{arrive[4]:02d}:{arrive[5]:02d}:{arrive[6]:02d}")
    print(f"   Julian day : {jd2:.3f}")
    print(f"   Planet position (km)  = [{R2[0]:.5e} {R2[1]:.5e} {R2[2]:.5e}]")
    print(f"     |R2|    (km)        = {np.linalg.norm(R2):.5e}")
    print(f"   Planet velocity (km/s)= [{V2[0]:.4f} {V2[1]:.4f} {V2[2]:.4f}]")
    print(f"     |V2|    (km/s)      = {np.linalg.norm(V2):.4f}")
    print(f"   Spacecraft velocity   = [{V2_sc[0]:.4f} {V2_sc[1]:.5f} {V2_sc[2]:.6f}]")
    print(f"     |V2_sc| (km/s)      = {np.linalg.norm(V2_sc):.4f}")
    print(f"   v-infinity            = [{vinf_arr[0]:.5f} {vinf_arr[1]:.6f} {vinf_arr[2]:.6f}]")
    print(f"     |vinf|  (km/s)      = {np.linalg.norm(vinf_arr):.5f}")

    print("\n Orbital elements of flight trajectory:")
    print(f"   Angular momentum h  (km^2/s) = {h:.5e}")
    print(f"   Eccentricity      e          = {e:.6f}")
    print(f"   RAAN              (deg)      = {RA*deg:.4f}")
    print(f"   Inclination       (deg)      = {incl*deg:.4f}")
    print(f"   Arg. perihelion   (deg)      = {w*deg:.4f}")
    print(f"   True anomaly @ departure     = {TA_dep*deg:.3f} deg")
    print(f"   True anomaly @ arrival       = {TA_arr*deg:.3f} deg")
    print(f"   Semimajor axis    a   (km)   = {a:.5e}")
    print(f"   Period            (days)     = {period_days:.3f}")
    print("-" * 60)
