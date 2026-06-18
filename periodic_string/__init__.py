"""Periodic taut string with moving oscillator coupled to 
the string by a contact spring.

This package assembles a periodic string on elastic supports and
integrates its response to a moving oscillator using
Newmark time integration.
"""

from periodic_string.solver import solve_moving_load

__all__ = ["solve_moving_load"]
