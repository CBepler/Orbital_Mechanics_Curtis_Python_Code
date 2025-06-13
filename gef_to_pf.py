import numpy as np

from coe_from_sv import coe_from_sv

def get_transformation_matrix(RA, incl, w):
    """
    Generate the transformation matrix from PF to GEF coordinates.

    Parameters:
    RA - Right Ascension of Ascending Node (degrees)
    incl - Inclination (degrees)
    w - Argument of Perigee (degrees)

    Returns:
    T - Transformation matrix
    """
    RA_rad = np.radians(RA)
    incl_rad = np.radians(incl)
    w_rad = np.radians(w)

    T = np.array([
        [np.cos(RA_rad) * np.cos(w_rad) - np.sin(RA_rad) * np.sin(w_rad) * np.cos(incl_rad),
         -np.cos(RA_rad) * np.sin(w_rad) - np.sin(RA_rad) * np.cos(w_rad) * np.cos(incl_rad),
         np.sin(RA_rad) * np.sin(incl_rad)],
        [np.sin(RA_rad) * np.cos(w_rad) + np.cos(RA_rad) * np.sin(w_rad) * np.cos(incl_rad),
         -np.sin(RA_rad) * np.sin(w_rad) + np.cos(RA_rad) * np.cos(w_rad) * np.cos(incl_rad),
         -np.cos(RA_rad) * np.sin(incl_rad)],
        [np.sin(w_rad) * np.sin(incl_rad), 
         np.cos(w_rad) * np.sin(incl_rad), 
         np.cos(incl_rad)]
    ])

    return T

def gef_to_pf(R, V):
    """
    Convert Geocentric Earth Fixed (GEF) coordinates to Perifocal (PF) coordinates.

    Parameters:
    R - Position vector in GEF coordinates (numpy array)
    V - Velocity vector in GEF coordinates (numpy array)

    Returns:
    R_pf - Position vector in PF coordinates
    V_pf - Velocity vector in PF coordinates
    """

    # Compute classical orbital elements from state vectors
    coe = coe_from_sv(R, V)

    # Extract classical orbital elements
    h, e, RA, incl, w, TA, a = coe

    RA_deg = np.degrees(RA)
    incl_deg = np.degrees(incl)
    w_deg = np.degrees(w)

    # Get the transformation matrix from PF to GEF coordinates
    T = get_transformation_matrix(RA_deg, incl_deg, w_deg)

    # Convert position and velocity vectors to PF coordinates
    R_pf = np.dot(T.T, R)
    V_pf = np.dot(T.T, V)

    return R_pf, V_pf

def pf_to_gef(R_pf, V_pf, RA, incl, w):
    """
    Convert Perifocal (PF) coordinates to Geocentric Earth Fixed (GEF) coordinates.

    Parameters:
    R_pf - Position vector in PF coordinates (numpy array)
    V_pf - Velocity vector in PF coordinates (numpy array)
    RA - Right Ascension of Ascending Node (degrees)
    incl - Inclination (degrees)
    w - Argument of Perigee (degrees)

    Returns:
    R_gef - Position vector in GEF coordinates
    V_gef - Velocity vector in GEF coordinates
    """

    # Get the transformation matrix from PF to GEF coordinates
    T = get_transformation_matrix(RA, incl, w)

    # Convert position and velocity vectors to GEF coordinates
    R_gef = np.dot(T, R_pf)
    V_gef = np.dot(T, V_pf)

    return R_gef, V_gef

if __name__ == "__main__":
    # Example usage
    R = np.array([-5102, -8228, -2105])  # Example position vector in GEF coordinates (km)
    V = np.array([-4.348, 3.478, -2.846])    # Example velocity vector in GEF coordinates (km/s)

    R_pf, V_pf = gef_to_pf(R, V)
    print("Position in PF coordinates:", R_pf)
    print("Velocity in PF coordinates:", V_pf)

    RA = 80.0944  # Example RA in degrees
    incl = 149.7570  # Example inclination in degrees
    w = 11.0904  # Example argument of perigee in degrees

    R_p = [5664, -7581, 0]
    V_p = [5.154, 4.038, 0]

    R_gef, V_gef = pf_to_gef(R_p, V_p, RA, incl, w)
    print("Position in GEF coordinates:", R_gef)
    print("Velocity in GEF coordinates:", V_gef)