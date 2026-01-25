import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ch5.helpers import (
    get_observer_position,
    degrees_to_radians,
    get_direction_cosines_horizon,
    topocentric_horizon_to_geocentric_equatorial,
)
from ch5.gibbs_method import gibbs_method

if __name__ == "__main__":
    R1 = get_observer_position(degrees_to_radians(-20), degrees_to_radians(60), 0.5)
    R2 = get_observer_position(
        degrees_to_radians(-20), degrees_to_radians(60.5014), 0.5
    )
    R3 = get_observer_position(
        degrees_to_radians(-20), degrees_to_radians(61.0027), 0.5
    )
    rho_1 = (
        topocentric_horizon_to_geocentric_equatorial(
            get_direction_cosines_horizon(
                degrees_to_radians(165.932), degrees_to_radians(8.81952)
            ),
            degrees_to_radians(60),
            degrees_to_radians(-20),
        )
        * 1212.48
    )
    rho_2 = (
        topocentric_horizon_to_geocentric_equatorial(
            get_direction_cosines_horizon(
                degrees_to_radians(145.970), degrees_to_radians(44.2734)
            ),
            degrees_to_radians(60.5014),
            degrees_to_radians(-20),
        )
        * 410.596
    )
    rho_3 = (
        topocentric_horizon_to_geocentric_equatorial(
            get_direction_cosines_horizon(
                degrees_to_radians(2.40973), degrees_to_radians(20.7594)
            ),
            degrees_to_radians(61.0027),
            degrees_to_radians(-20),
        )
        * 726.464
    )
    r1 = R1 + rho_1
    r2 = R2 + rho_2
    r3 = R3 + rho_3
    v2 = gibbs_method(r1, r2, r3)
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")
