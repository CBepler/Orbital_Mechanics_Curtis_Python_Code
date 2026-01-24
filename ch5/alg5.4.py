import numpy as np

R_E = 6378
FLATTENING = 1/298.257223563
ANGULAR_VELOCITY_EARTH = 7.292115e-5

def alg5_4(range, range_rate, azimuth, azimuth_rate, elevation, elevation_rate, local_sidereal_time, latitude, altitude):
    #Produces state vector from angles and range measurements
    #Calculate Geocentric position vector of the observer
    factor_1 = (R_E / (np.sqrt(1 - ((2 * FLATTENING) - (FLATTENING ** 2)) * np.sin(latitude) ** 2)) + altitude) * np.cos(latitude)
    factor_2 = ((R_E * (1 - FLATTENING)**2) / (np.sqrt(1 - ((2 * FLATTENING) - (FLATTENING ** 2)) * np.sin(latitude) ** 2)) + altitude)
    R = np.array([factor_1 * np.cos(local_sidereal_time), factor_1 * np.sin(local_sidereal_time), factor_2 * np.sin(latitude)])
    #Calculate topocentric declination
    declination = np.arcsin(np.cos(latitude) * np.cos(azimuth) * np.cos(elevation) + np.sin(latitude) * np.sin(elevation))
    #Calculate topocentric right ascension
    hour_angle = np.arccos((np.cos(latitude) * np.sin(elevation) - np.sin(latitude) * np.cos(elevation) * np.cos(azimuth))/(np.cos(declination)))
    if azimuth < np.pi:
        hour_angle = 2 * np.pi - hour_angle
    right_ascension = local_sidereal_time - hour_angle
    #Calculate the direction cosines unit vector
    direction_cosines = np.array([np.cos(right_ascension) * np.cos(declination), np.sin(right_ascension) * np.cos(declination), np.sin(declination)])
    #Calculate the geocentric position vector of the target
    r = R + range * direction_cosines
    #Calculate the inertial velocity of the site
    constant_angular_velocity_R = np.array([0, 0, ANGULAR_VELOCITY_EARTH])
    R_dot = np.cross(constant_angular_velocity_R, R)
    #Calculate the declination rate
    declination_rate = (1 / np.cos(declination)) * (-azimuth_rate * np.cos(latitude) * np.sin(azimuth) * np.cos(elevation) + elevation_rate * (np.sin(latitude) * np.cos(elevation) - np.cos(latitude) * np.cos(azimuth) * np.sin(elevation)))
    #Calculate the right ascension rate
    right_ascension_rate = ANGULAR_VELOCITY_EARTH + (azimuth_rate * np.cos(azimuth) * np.cos(elevation) - elevation_rate * np.sin(azimuth) * np.sin(elevation) + declination_rate * np.sin(azimuth) * np.cos(elevation) * np.tan(declination))/(np.cos(latitude) * np.sin(elevation) - np.sin(latitude) * np.cos(azimuth) * np.cos(elevation))
    #Calculate the direction cosines rates vector
    direction_cosines_rate = np.array([(-right_ascension_rate * np.sin(right_ascension) * np.cos(declination)) - (declination_rate * np.cos(right_ascension) * np.sin(declination)), (right_ascension_rate * np.cos(right_ascension) * np.cos(declination)) - (declination_rate * np.sin(right_ascension) * np.sin(declination)), declination_rate * np.cos(declination)])    
    #Calculate the geocentric velocity vector
    v = R_dot + range_rate * direction_cosines + range * direction_cosines_rate
    return r, v

if __name__ == '__main__':
    degrees_to_radians = lambda x: x * np.pi / 180
    r, v = alg5_4(988, 4.86, degrees_to_radians(36), degrees_to_radians(0.590), degrees_to_radians(36.6), degrees_to_radians(-0.263), degrees_to_radians(40), degrees_to_radians(35), 0)
    print(f"r = {r}")
    print(f"mag_r = {np.linalg.norm(r)}")
    print(f"v = {v}")
    print(f"mag_v = {np.linalg.norm(v)}")