import numpy as np

R_E = 6378
FLATTENING = 1 / 298.257223563
ANGULAR_VELOCITY_EARTH = 7.292115e-5


def get_observer_position(latitude, local_sidereal_time, altitude):
    factor_1 = (
        R_E
        / (np.sqrt(1 - ((2 * FLATTENING) - (FLATTENING**2)) * np.sin(latitude) ** 2))
        + altitude
    ) * np.cos(latitude)
    factor_2 = (R_E * (1 - FLATTENING) ** 2) / (
        np.sqrt(1 - ((2 * FLATTENING) - (FLATTENING**2)) * np.sin(latitude) ** 2)
    ) + altitude
    R = np.array(
        [
            factor_1 * np.cos(local_sidereal_time),
            factor_1 * np.sin(local_sidereal_time),
            factor_2 * np.sin(latitude),
        ]
    )
    return R


def degrees_to_radians(degrees):
    return degrees * np.pi / 180


def get_direction_cosines_horizon(azimuth, elevation):
    return np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            np.sin(elevation),
        ]
    )


def topocentric_horizon_to_geocentric_equatorial(vec, local_sidereal_time, latitude):
    transformation_matrix = np.array(
        [
            [
                -np.sin(local_sidereal_time),
                -np.sin(latitude) * np.cos(local_sidereal_time),
                np.cos(latitude) * np.cos(local_sidereal_time),
            ],
            [
                np.cos(local_sidereal_time),
                -np.sin(latitude) * np.sin(local_sidereal_time),
                np.cos(latitude) * np.sin(local_sidereal_time),
            ],
            [0, np.cos(latitude), np.sin(latitude)],
        ]
    )
    return transformation_matrix @ vec
