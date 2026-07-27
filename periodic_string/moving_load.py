from __future__ import annotations

import numpy as np


def wrap_load_position(x: float, x_min: float, x_max: float) -> float:
    """Wrap a load position into the periodic domain ``[x_min, x_max)``."""
    span = x_max - x_min
    if span <= 0.0:
        return x
    return x_min + (x - x_min) % span


def contact_element(
    x: float,
    element_length: float,
    n_nodes: int,
) -> tuple[int, int, float, float]:
    """Locate the element containing ``x`` and its shape-function weights.

    The mesh is uniform and periodic, so the element index is arithmetic —
    no search. (The previous implementation scanned every element in a
    Python loop, which is O(n_nodes) per time step.)

    Parameters
    ----------
    x : float
        Load position along the catenary, already wrapped into the domain.
    element_length : float
        Uniform element length.
    n_nodes : int
        Number of string nodes (also the number of elements, as the mesh
        closes on itself).

    Returns
    -------
    node_left, node_right : int
        Global node indices of the containing element.
    w_left, w_right : float
        Linear shape-function values, summing to unity.
    """
    pos = x / element_length
    index = int(pos)
    if index >= n_nodes:          # guard against x exactly at the wrap point
        index = n_nodes - 1
    frac = pos - index
    return index, (index + 1) % n_nodes, 1.0 - frac, frac


def load_shape_vector(
    node_x: np.ndarray,
    x: float,
    catenary_length: float,
    n_dof: int,
) -> np.ndarray:
    """Dense shape vector for a unit transverse point load at ``x``.

    Retained for tests and for the static-equilibrium path. The time loop
    uses :func:`contact_element` instead and never forms this densely.
    """
    shape = np.zeros(n_dof)
    n_nodes = node_x.size
    element_length = catenary_length / n_nodes
    left, right, w_left, w_right = contact_element(x, element_length, n_nodes)
    shape[left] = w_left
    shape[right] = w_right
    return shape


def displacement_at_load(shape: np.ndarray, displacement: np.ndarray) -> float:
    """Interpolate vertical displacement at the load position."""
    return float(shape @ displacement)


def contact_direction(
    node_x: np.ndarray,
    x: float,
    catenary_length: float,
    n_dof: int,
    contact_dof: int,
) -> np.ndarray:
    """Dense rank-1 direction vector ``d(t) = [N(t); -1; 0]``.

    Retained for the static-equilibrium path and for tests. The time loop
    uses the sparse three-entry form returned by :func:`contact_triplet`.
    """
    d = load_shape_vector(node_x, x, catenary_length, n_dof)
    d[contact_dof] = -1.0
    return d


def contact_triplet(
    x: float,
    element_length: float,
    n_nodes: int,
    contact_dof: int,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Sparse form of ``d(t)``: the three DOF indices and their values.

    Returns
    -------
    dofs : numpy.ndarray of int
        ``[node_left, node_right, contact_dof]``.
    values : numpy.ndarray of float
        ``[w_left, w_right, -1.0]``.
    node_left, node_right : int
        Element node indices (also returned separately for the solve cache).
    w_left, w_right : float
        Shape-function weights.
    """
    left, right, w_left, w_right = contact_element(x, element_length, n_nodes)
    dofs = np.array([left, right, contact_dof], dtype=np.intp)
    values = np.array([w_left, w_right, -1.0], dtype=float)
    return dofs, values, left, right, w_left, w_right
