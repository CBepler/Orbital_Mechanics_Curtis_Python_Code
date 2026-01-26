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

    # 5.22
    times = [0, 60, 120]
    observer_position = np.array(
        [
            [-1825.96, 3583.66, 4933.54],
            [-1841.63, 3575.63, 4933.54],
            [-1857.25, 3567.54, 4933.54],
        ]
    )
    direction_cosines = np.array(
        [
            [-0.301687, 0.200673, 0.932049],
            [-0.793090, -0.210324, 0.571640],
            [-0.873085, -0.362969, 0.325539],
        ]
    )
    r2, v2 = gauss(direction_cosines, observer_position, times)
    print("========== 5.22 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.23
    r2, v2 = gauss_iterative(direction_cosines, observer_position, times)
    print("========== 5.23 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.25
    altitude = 0.5
    latitude = degrees_to_radians(60)
    times = [0, 300, 600]
    local_sidereal_time = map(degrees_to_radians, [150, 151.253, 152.507])
    topocentric_ra = map(degrees_to_radians, [157.783, 159.221, 160.526])
    topocentric_dec = map(degrees_to_radians, [24.2403, 27.2993, 29.8982])
    observer_position = [
        get_observer_position(latitude, lst, altitude) for lst in local_sidereal_time
    ]
    direction_cosines = list(
        map(get_direction_cosines_topocentric, topocentric_ra, topocentric_dec)
    )
    r2, v2 = gauss(direction_cosines, observer_position, times)
    print("========== 5.25 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.26
    r2, v2 = gauss_iterative(direction_cosines, observer_position, times)
    print("========== 5.26 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.28
    times = [0, 300, 600]
    observer_position = np.array(
        [
            [5582.84, 0, 3073.90],
            [5581.50, 122.122, 3073.90],
            [5577.50, 244.186, 3073.90],
        ]
    )
    direction_cosines = np.array(
        [
            [0.846428, 0, 0.532504],
            [0.749290, 0.463023, 0.473470],
            [0.529447, 0.777163, 0.340152],
        ]
    )
    r2, v2 = gauss(direction_cosines, observer_position, times)
    print("========== 5.28 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")

    # 5.29
    r2, v2 = gauss_iterative(direction_cosines, observer_position, times)
    print("========== 5.29 ==========")
    print(f"r2 = {r2}")
    print(f"v2 = {v2}")
    print(f"r2_mag = {np.linalg.norm(r2)}")
    print(f"v2_mag = {np.linalg.norm(v2)}")
