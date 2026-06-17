from __future__ import annotations

import numpy as np


def string_stiffness(tension: float, length: float) -> np.ndarray:
    """Local stiffness matrix for a 2-node linear-string element.

    Parameters
    ----------
    tension : float
        Axial tension (T) carried by the string.
    length : float
        Element length.

    Returns
    -------
    numpy.ndarray
        2x2 stiffness matrix with DOF ordering ``[v1, v2]``.
    """
    return tension / length * np.array(
        [
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )


def string_mass(mass_per_length: float, length: float) -> np.ndarray:
    """Consistent mass matrix for a 2-node taut-string element.

    Parameters
    ----------
    mass_per_length : float
        Mass per unit length of the string.
    length : float
        Element length.

    Returns
    -------
    numpy.ndarray
        2x2 consistent mass matrix with DOF ordering ``[v1, v2]``.
    """
    return mass_per_length * length / 6.0 * np.array(
        [
            [2.0, 1.0],
            [1.0, 2.0],
        ]
    )
