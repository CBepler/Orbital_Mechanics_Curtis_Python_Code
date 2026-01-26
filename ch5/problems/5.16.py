import numpy as np
import sys
from pathlib import Path

# Add parent directories to path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ch5.helpers import (
    degrees_to_radians,
    get_observer_position,
    get_direction_cosines_topocentric,
)
from ch5.gauss import gauss
from ch5.gauss_iterative import gauss_iterative

if __name__ == "__main__":
    # 5.16
    altitude = 0
    latitude = degrees_to_radians(29)
    times = [0, 60, 120]
    local_sidereal_time = map(degrees_to_radians, [0, 0.250684, 0.501369])
    topocentric_ra = map(degrees_to_radians, [0, 65.9279, 79.8500])
    topocentric_dec = map(degrees_to_radians, [51.5110, 27.9911, 14.6609])
    observer_position = [
        get_observer_position(latitude, lst, altitude) for lst in local_sidereal_time
    ]
    direction_cosines = list(
        map(get_direction_cosines_topocentric, topocentric_ra, topocentric_dec)
    )
    r2, v2 = gauss(direction_cosines, observer_position, times)
    print("========== 5.16 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.18
    r2, v2 = gauss_iterative(direction_cosines, observer_position, times)
    print("========== 5.18 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.19
    altitude = 0
    latitude = degrees_to_radians(29)
    times = [0, 60, 120]
    local_sidereal_time = map(degrees_to_radians, [90, 90.2507, 90.5014])
    topocentric_ra = map(degrees_to_radians, [15.0394, 25.7539, 48.6055])
    topocentric_dec = map(degrees_to_radians, [20.7487, 30.1410, 43.8910])
    observer_position = [
        get_observer_position(latitude, lst, altitude) for lst in local_sidereal_time
    ]
    direction_cosines = list(
        map(get_direction_cosines_topocentric, topocentric_ra, topocentric_dec)
    )
    r2, v2 = gauss(direction_cosines, observer_position, times)
    print("========== 5.19 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.20
    r2, v2 = gauss_iterative(direction_cosines, observer_position, times)
    print("========== 5.20 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")
