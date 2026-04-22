def local_sidereal_time(year, month, day, hour, minute, second, east_long):
    # Calculates the local sidereal time at a given date and time
    # Valid for dates from 1901 to 2099
    j_0 = 367 * year - int((7 * (year + int((month + 9) / 12))) / 4) + int(275 * month / 9) + day + 1721013.5
    t_0 = (j_0 - 2451545) / 36525
    theta_g_0 = 100.4606184 + 36000.77004 * t_0 + 0.000387933 * t_0**2 - 2.583 * 10**-8 * t_0**3
    theta_g_0 = theta_g_0 % 360
    if theta_g_0 < 0:
        theta_g_0 += 360
    ut = hour + minute / 60 + second / 3600
    theta_g = theta_g_0 + 360.98564724 * (ut / 24)
    theta = theta_g + east_long
    theta = theta % 360
    if theta < 0:
        theta += 360
    return theta

def julian_date(year, month, day, hour, minute, second):
    # Calculates the Julian Date (JD) at a given date and time
    # Valid for dates from 1901 to 2099
    j_0 = 367 * year - int((7 * (year + int((month + 9) / 12))) / 4) + int(275 * month / 9) + day + 1721013.5
    ut = hour + minute / 60 + second / 3600
    j_d = j_0 + ut / 24
    return j_d

def julian_century(julian_day):
    return (julian_day - 2451545) / 36525

if __name__ == '__main__':
    #print(local_sidereal_time(2026, 1, 23, 12, 0, 0, 75.47))
    #print(julian_date(2005, 12, 31, 23, 59, 59))
    print(julian_century(2453736.5))
