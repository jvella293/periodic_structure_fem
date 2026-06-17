from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from tqdm import tqdm

from periodic_string.elements import (
    beam_mass,
    beam_stiffness,
)


@dataclass(frozen=True)
class AssembledModel:
    """Global finite-element model of a periodic beam on springs.

    Attributes
    ----------
    node_x : numpy.ndarray
        Longitudinal coordinate of each node.
    spring_nodes : numpy.ndarray
        Node indices where vertical and rotational springs are attached.
    track_length : float
        Total length of one periodic track span.
    n_dof : int
        Number of global degrees of freedom (``2 * n_nodes``).
    mass : scipy.sparse.csr_matrix
        Global mass matrix.
    stiffness : scipy.sparse.csr_matrix
        Global stiffness matrix.
    damping : scipy.sparse.csr_matrix
        Global damping matrix.
    free_dofs : numpy.ndarray
        Indices of unconstrained degrees of freedom.
    """

    node_x: np.ndarray
    spring_nodes: np.ndarray
    track_length: float
    n_dof: int
    mass: sparse.csr_matrix
    stiffness: sparse.csr_matrix
    damping: sparse.csr_matrix
    free_dofs: np.ndarray


def mesh_parameters(
    spacing: float,
    element_length_requested: float,
    n_cells: int,
) -> tuple[float, int, int, float]:
    """Compute mesh discretisation parameters for a periodic track.

    The requested element length is adjusted so that each cell spacing
    is divided by an integer number of elements.

    Parameters
    ----------
    spacing : float
        Length of one periodic cell.
    element_length_requested : float
        Target beam element length before rounding.
    n_cells : int
        Number of periodic cells along the track.

    Returns
    -------
    element_length : float
        Actual element length after adjustment.
    n_elements_per_cell : int
        Number of beam elements per cell.
    n_nodes : int
        Total number of nodes in the mesh.
    track_length : float
        Total track length (``n_cells * spacing``).
    """
    n_elements_per_cell = max(1, round(spacing / element_length_requested))
    element_length = spacing / n_elements_per_cell
    n_nodes = n_cells * n_elements_per_cell
    track_length = n_cells * spacing
    return element_length, n_elements_per_cell, n_nodes, track_length


def _dof_indices(node_index: int) -> tuple[int, int]:
    """Map a node index to its vertical and rotational DOF indices.

    Parameters
    ----------
    node_index : int
        Zero-based node index.

    Returns
    -------
    vertical_dof : int
        Global index of the vertical displacement DOF.
    rotational_dof : int
        Global index of the rotational DOF.
    """
    return 2 * node_index, 2 * node_index + 1


def _add_matrix(
    triplets: list[tuple[float, int, int]],
    local: np.ndarray,
    global_dofs: list[int],
) -> None:
    """Scatter a dense local matrix into COO-format triplets.

    Parameters
    ----------
    triplets : list of tuple of (float, int, int)
        Mutable list of ``(value, row, col)`` entries to append to.
    local : numpy.ndarray
        Dense element matrix in local DOF ordering.
    global_dofs : list of int
        Global DOF indices corresponding to each local row and column.

    Returns
    -------
    None
        Entries are appended in place to ``triplets``.
    """
    for local_row, global_row in enumerate(global_dofs):
        for local_col, global_col in enumerate(global_dofs):
            value = local[local_row, local_col]
            if value != 0.0:
                triplets.append((value, global_row, global_col))


def _triplets_to_csr(triplets: list[tuple[float, int, int]], n_dof: int) -> sparse.csr_matrix:
    """Build a CSR sparse matrix from COO-format triplets.

    Duplicate ``(row, col)`` entries are summed by SciPy during construction.

    Parameters
    ----------
    triplets : list of tuple of (float, int, int)
        Matrix entries as ``(value, row, col)``.
    n_dof : int
        Number of rows and columns in the global matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Assembled sparse matrix of shape ``(n_dof, n_dof)``.
    """
    if not triplets:
        return sparse.csr_matrix((n_dof, n_dof))
    data = [value for value, _, _ in triplets]
    rows = [row for _, row, _ in triplets]
    cols = [col for _, _, col in triplets]
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_dof, n_dof))


def assemble_model(
    *,
    ei: float,
    damp_rail: float,
    mass_per_length: float,
    kv: float,
    kt: float,
    damp_rp: float,
    element_length: float,
    n_elements_per_cell: int,
    n_nodes: int,
    track_length: float,
    show_progress: bool = True,
) -> AssembledModel:
    """Assemble global mass, stiffness, and damping matrices for the beam.

    Beam elements form a closed periodic loop. Vertical and rotational
    springs to ground are placed at the first node of each cell.

    Parameters
    ----------
    ei : float
        Bending rigidity of the beam.
    damp_rail : float
        Rayleigh-type damping factor applied to beam bending stiffness.
    mass_per_length : float
        Mass per unit length of the beam.
    kv : float
        Vertical spring stiffness at cell boundaries.
    kt : float
        Rotational spring stiffness at cell boundaries.
    damp_rp : float
        Damping factor applied to spring stiffnesses.
    element_length : float
        Length of each beam element.
    n_elements_per_cell : int
        Number of beam elements per periodic cell.
    n_nodes : int
        Total number of nodes in the mesh.
    track_length : float
        Total length of one periodic track span.
    show_progress : bool, optional
        If ``True``, display tqdm progress bars during assembly.

    Returns
    -------
    AssembledModel
        Assembled sparse global model ready for time integration.
    """
    n_dof = 2 * n_nodes
    node_x = np.arange(n_nodes, dtype=float) * element_length
    spring_nodes = np.arange(n_nodes, step=n_elements_per_cell, dtype=int)

    cei = ei * damp_rail
    ck = kv * damp_rp
    ckt = kt * damp_rp

    k_triplets: list[tuple[float, int, int]] = []
    c_triplets: list[tuple[float, int, int]] = []
    m_triplets: list[tuple[float, int, int]] = []

    for element_index in tqdm(
        range(n_nodes),
        desc="Assembling beam elements",
        disable=not show_progress,
    ):
        node_a = element_index
        node_b = (element_index + 1) % n_nodes
        dofs = [*_dof_indices(node_a), *_dof_indices(node_b)]

        _add_matrix(k_triplets, beam_stiffness(ei, element_length), dofs)
        _add_matrix(c_triplets, beam_stiffness(cei, element_length), dofs)
        _add_matrix(m_triplets, beam_mass(mass_per_length, element_length), dofs)

    for spring_node in spring_nodes:
        vertical_dof = 2 * spring_node
        rotational_dof = 2 * spring_node + 1
        k_triplets.append((kv, vertical_dof, vertical_dof))
        c_triplets.append((ck, vertical_dof, vertical_dof))
        k_triplets.append((kt, rotational_dof, rotational_dof))
        c_triplets.append((ckt, rotational_dof, rotational_dof))

    stiffness = _triplets_to_csr(k_triplets, n_dof)
    damping = _triplets_to_csr(c_triplets, n_dof)
    mass = _triplets_to_csr(m_triplets, n_dof)
    stiffness = 0.5 * (stiffness + stiffness.T)
    damping = 0.5 * (damping + damping.T)
    mass = 0.5 * (mass + mass.T)

    return AssembledModel(
        node_x=node_x,
        spring_nodes=spring_nodes,
        track_length=track_length,
        n_dof=n_dof,
        mass=mass,
        stiffness=stiffness,
        damping=damping,
        free_dofs=np.arange(n_dof, dtype=int),
    )
