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
    """Evaluate linear shape functions for a unit transverse point load.

    The load is applied at position ``x`` on the string mesh. Shape
    function values are written to the transverse DOFs of the two nodes
    of the containing element.

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
        containing ``x``. Entries sum to unity within an element.
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
        shape[node_left] = 1.0 - fact
        shape[node_right] = fact
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
