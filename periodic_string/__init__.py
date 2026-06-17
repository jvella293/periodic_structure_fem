"""Periodic taut string with moving load.

This package assembles a periodic string on elastic supports and
integrates its response to a moving harmonic point load using
Newmark time integration.
"""

from periodic_string.solver import solve_moving_load

__all__ = ["solve_moving_load"]
