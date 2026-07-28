"""
rocket_altitude.py

Single-stage vertical (gamma = 90 deg) rocket trajectory: constant
propellant mass flow rate, no drag, constant g0. Based on Curtis,
"Orbital Mechanics for Engineering Students", Example 11.1, generalized
to allow a nonzero initial velocity v0 (and initial altitude h0), so it
can also represent a stage that ignites already moving -- e.g. an upper
stage, or a rocket launched from a moving platform.

Governing equations (Curtis Eqs. c and d, with v0/h0 added):

  Burn phase (0 <= t <= t_bo, t_bo = (m0 - mf)/mdot):

    m(t) = m0 - mdot*t
    v(t) = v0 + c*ln(m0/m(t)) - g0*t
    h(t) = h0 + v0*t + (c/mdot)*[m(t)*ln(m0/m(t)) + mdot*t] - 0.5*g0*t**2

    where c = Isp*g0 is the effective exhaust velocity.

  Coast phase (t > t_bo, free flight under gravity alone):

    v(t) = v_bo - g0*(t - t_bo)
    h(t) = h_bo + v_bo*(t - t_bo) - 0.5*g0*(t - t_bo)**2

Usage
-----
    rocket = Rocket(m0=249.5, mf=170.1, mdot=10.61, Isp=235, v0=0, h0=0)
    v, h = rocket.state(t)      # velocity [m/s], altitude [m] at time t [s]
    t_apogee, h_max = rocket.max_altitude()
"""

from dataclasses import dataclass
import math


@dataclass
class Rocket:
    m0: float  # initial mass, kg
    mf: float  # burnout mass, kg
    mdot: float  # constant propellant mass flow rate, kg/s
    Isp: float  # specific impulse, s
    v0: float = 0.0  # initial velocity at t=0, m/s
    h0: float = 0.0  # initial altitude at t=0, m
    g0: float = 9.81  # gravitational acceleration, m/s^2 (assumed constant)

    @property
    def c(self) -> float:
        """Effective exhaust velocity Isp*g0."""
        return self.Isp * self.g0

    @property
    def t_bo(self) -> float:
        """Burnout time."""
        return (self.m0 - self.mf) / self.mdot

    def state(self, t: float):
        """Return (velocity [m/s], altitude [m]) at time t [s] since t=0."""
        if t < 0:
            raise ValueError("t must be >= 0")

        g0, c, mdot, m0, v0, h0 = self.g0, self.c, self.mdot, self.m0, self.v0, self.h0
        t_bo = self.t_bo

        if t <= t_bo:
            m_t = m0 - mdot * t
            v = v0 + c * math.log(m0 / m_t) - g0 * t
            h = (
                h0
                + v0 * t
                + (c / mdot) * (m_t * math.log(m_t / m0) + mdot * t)
                - 0.5 * g0 * t**2
            )
            return v, h
        else:
            v_bo, h_bo = self.state(t_bo)
            dt = t - t_bo
            v = v_bo - g0 * dt
            h = h_bo + v_bo * dt - 0.5 * g0 * dt**2
            return v, h

    def altitude(self, t: float) -> float:
        return self.state(t)[1]

    def velocity(self, t: float) -> float:
        return self.state(t)[0]

    def max_altitude(self):
        """
        Apogee (v = 0). If the rocket is still under thrust and never
        stops climbing before burnout, apogee occurs during the coast
        phase, found in closed form from v_bo and h_bo.
        """
        v_bo, h_bo = self.state(self.t_bo)
        if v_bo <= 0:
            raise RuntimeError(
                "Velocity is already <= 0 at burnout; "
                "apogee occurs during the burn -- refine "
                "with a root find if this case applies to you."
            )
        t_apogee = self.t_bo + v_bo / self.g0
        h_apogee = h_bo + v_bo**2 / (2 * self.g0)
        return t_apogee, h_apogee


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Curtis Problem 11.2 -- two-stage sounding rocket.
    # Note: stage 1 Isp = 300 s (not 235 s -- the problem statement's listed
    # value of 235 s for stage 1 does not reproduce the solutions manual;
    # 300 s does, matching every intermediate value below).
    g0 = 9.81

    rocket1 = Rocket(m0=249.5, mf=170.1, mdot=10.61, Isp=300, v0=0, h0=0)
    v_bo1, h_bo1 = rocket1.state(rocket1.t_bo)
    print(
        f"Stage 1 burnout: v = {v_bo1:.1f} m/s, h = {h_bo1:.1f} m   (manual: 1054 m/s, 3673 m)"
    )

    # 3 s coast between stage 1 burnout and stage 2 ignition
    coast = 3.0
    v0 = v_bo1 - g0 * coast
    h0 = h_bo1 + v_bo1 * coast - 0.5 * g0 * coast**2
    print(
        f"After coast:     v0 = {v0:.1f} m/s, h0 = {h0:.1f} m   (manual: 1024 m/s, 6790 m)"
    )

    rocket2 = Rocket(m0=113.4, mf=58.97, mdot=4.053, Isp=235, v0=v0, h0=h0)
    v_bo2, h_bo2 = rocket2.state(rocket2.t_bo)
    print(
        f"Stage 2 burnout: v = {v_bo2:.1f} m/s, h = {h_bo2:.1f} m   (manual: 2400 m/s, 28690 m)"
    )

    t_max, h_max = rocket2.max_altitude()
    print(
        f"Apogee:          t = {t_max:.1f} s (from liftoff), h_max = {h_max / 1000:.1f} km"
        f"   (manual: 244.7 s from 2nd-stage burnout, 322.3 km)"
    )
