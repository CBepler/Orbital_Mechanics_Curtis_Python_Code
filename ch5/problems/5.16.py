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

if __name__ == "__main__":
    altitude = 0
    latitude = degrees_to_radians(29)
    times = [0, 60, 120]
    local_sidereal_time = map(degrees_to_radians, [0, 0.250684, 0.501369])
    topocentric_ra = map(degrees_to_radians, [0, 65.9279, 79.8500])
    topocentric_dec = map(degrees_to_radians, [51.5110, 27.9911, 14.6609])
    observer_position = list(
        map(get_observer_position, latitude, local_sidereal_time, altitude)
    )
    direction_cosines = list(
        map(get_direction_cosines_topocentric, topocentric_ra, topocentric_dec)
    )
    r2, v2 = gauss(direction_cosines, observer_position, times)
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")
