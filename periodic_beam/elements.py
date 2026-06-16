from __future__ import annotations

import numpy as np


def beam_stiffness(ei: float, length: float) -> np.ndarray:
    """Local stiffness matrix for a 2-node Euler-Bernoulli beam element.

    Parameters
    ----------
    ei : float
        Bending rigidity (E * I) of the beam.
    length : float
        Element length.

    Returns
    -------
    numpy.ndarray
        4x4 stiffness matrix with DOF ordering
        ``[v1, theta1, v2, theta2]``.
    """
    l = length
    l2 = l * l
    l3 = l2 * l
    return ei * np.array(
        [
            [12.0 / l3, 6.0 / l2, -12.0 / l3, 6.0 / l2],
            [6.0 / l2, 4.0 / l, -6.0 / l2, 2.0 / l],
            [-12.0 / l3, -6.0 / l2, 12.0 / l3, -6.0 / l2],
            [6.0 / l2, 2.0 / l, -6.0 / l2, 4.0 / l],
        ]
    )


def beam_mass(mass_per_length: float, length: float) -> np.ndarray:
    """Consistent mass matrix for a 2-node Euler-Bernoulli beam element.

    Parameters
    ----------
    mass_per_length : float
        Mass per unit length of the beam.
    length : float
        Element length.

    Returns
    -------
    numpy.ndarray
        4x4 consistent mass matrix with DOF ordering
        ``[v1, theta1, v2, theta2]``.
    """
    l = length
    l2 = l * l
    return mass_per_length * l / 420.0 * np.array(
        [
            [156.0, 22.0 * l, 54.0, -13.0 * l],
            [22.0 * l, 4.0 * l2, 13.0 * l, -3.0 * l2],
            [54.0, 13.0 * l, 156.0, -22.0 * l],
            [-13.0 * l, -3.0 * l2, -22.0 * l, 4.0 * l2],
        ]
    )
