from __future__ import annotations

import numpy as np


def wrap_load_position(x: float, x_min: float, x_max: float) -> float:
    """Wrap a load position into the periodic domain ``[x_min, x_max)``.

    Parameters
    ----------
    x : float
        Current load position along the track.
    x_min : float
        Lower bound of the periodic domain.
    x_max : float
        Upper bound of the periodic domain (exclusive after wrapping).

    Returns
    -------
    float
        Position wrapped into ``[x_min, x_max)``.
    """
    span = x_max - x_min
    while x < x_min:
        x += span
    while x >= x_max:
        x -= span
    return x


def load_shape_vector(
    node_x: np.ndarray,
    x: float,
    track_length: float,
    n_dof: int,
) -> np.ndarray:
    """Evaluate Hermite shape functions for a unit vertical point load.

    The load is applied at position ``x`` on the beam mesh. Shape
    function values are written to the vertical and rotational DOFs of
    the two nodes of the containing element.

    Parameters
    ----------
    node_x : numpy.ndarray
        Longitudinal coordinate of each node.
    x : float
        Load position along the track.
    track_length : float
        Total length of one periodic track span.
    n_dof : int
        Total number of global degrees of freedom.

    Returns
    -------
    numpy.ndarray
        Shape vector of length ``n_dof``; nonzero only on the element
        containing ``x``. Vertical entries sum to unity when the load
        lies in a single element interior.
    """
    shape = np.zeros(n_dof)
    n_nodes = node_x.size

    for element_index in range(n_nodes):
        node_left = element_index
        node_right = (element_index + 1) % n_nodes
        x1 = node_x[node_left]
        x2 = track_length if node_right == 0 else node_x[node_right]

        if x < x1 or x >= x2:
            continue

        length = x2 - x1
        fact = (x - x1) / length
        n1 = 1.0 - 3.0 * fact**2 + 2.0 * fact**3
        n2 = length * fact - 2.0 * length * fact**2 + length * fact**3
        n3 = 3.0 * fact**2 - 2.0 * fact**3
        n4 = -length * fact**2 + length * fact**3

        shape[2 * node_left] = n1
        shape[2 * node_left + 1] = n2
        shape[2 * node_right] = n3
        shape[2 * node_right + 1] = n4
        return shape

    return shape


def displacement_at_load(shape: np.ndarray, displacement: np.ndarray) -> float:
    """Interpolate vertical displacement at the load position.

    Parameters
    ----------
    shape : numpy.ndarray
        Load shape vector from :func:`load_shape_vector`.
    displacement : numpy.ndarray
        Global displacement vector.

    Returns
    -------
    float
        Vertical displacement at the load point (``shape @ displacement``).
    """
    return float(shape @ displacement)
