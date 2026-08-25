from __future__ import annotations

import numpy as np


def wrap_load_position(x: float, x_min: float, x_max: float) -> float:
    """Wrap a load position into the periodic domain ``[x_min, x_max)``.

    Parameters
    ----------
    x : float
        Current load position along the catenary.
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


def contact_element(
    x: float,
    element_length: float,
    n_nodes: int,
) -> tuple[int, int, float, float]:
    """Locate the element containing ``x`` and its shape-function weights.

    The mesh is uniform and periodic, so the element index is arithmetic —
    no search. Equivalent to the scan in :func:`load_shape_vector`, but O(1)
    per call, which matters inside the time loop.

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


def contact_triplet(
    x: float,
    element_length: float,
    n_nodes: int,
    mass_dof: int,
) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Sparse form of ``d(t) = [N(t); -1]``: three DOF indices and values.

    Returns
    -------
    dofs : numpy.ndarray of int
        ``[node_left, node_right, mass_dof]``.
    values : numpy.ndarray of float
        ``[w_left, w_right, -1.0]``.
    node_left, node_right : int
        Element node indices (also returned separately for the solve cache).
    w_left, w_right : float
        Shape-function weights.
    """
    left, right, w_left, w_right = contact_element(x, element_length, n_nodes)
    dofs = np.array([left, right, mass_dof], dtype=np.intp)
    values = np.array([w_left, w_right, -1.0], dtype=float)
    return dofs, values, left, right, w_left, w_right


def load_shape_vector(
    node_x: np.ndarray,
    x: float,
    catenary_length: float,
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
        Load position along the catenary.
    catenary_length : float
        Total length of one periodic catenary span.
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
        x2 = catenary_length if node_right == 0 else node_x[node_right]

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

def contact_direction(
    node_x: np.ndarray,
    x: float,
    catenary_length: float,
    n_dof: int,
    mass_dof: int,
) -> np.ndarray:
    """Build the rank-1 direction vector for the moving contact coupling.

    Returns ``d(t) = [N(t); -1]`` of length ``n_dof``, where ``N(t)`` is
    the linear shape function vector at the load position ``x`` and the
    final entry corresponds to the moving-mass DOF. The contact
    stiffness contribution to the global stiffness matrix is
    ``K * d(t) d(t).T``.

    Parameters
    ----------
    node_x : numpy.ndarray
        Longitudinal coordinate of each node.
    x : float
        Load position along the catenary.
    catenary_length : float
        Total length of one periodic catenary span.
    n_dof : int
        Total number of global degrees of freedom (string + mass).
    mass_dof : int
        Global index of the moving-mass DOF.

    Returns
    -------
    numpy.ndarray
        Direction vector ``d(t)`` of length ``n_dof``.
    """
    d = load_shape_vector(node_x, x, catenary_length, n_dof)
    d[mass_dof] = -1.0
    return d
