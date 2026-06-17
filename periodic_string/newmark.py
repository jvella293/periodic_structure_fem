from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


@dataclass
class NewmarkState:
    """State vector at one time step of Newmark integration.

    Attributes
    ----------
    displacement : numpy.ndarray
        Global displacement vector.
    velocity : numpy.ndarray
        Global velocity vector.
    acceleration : numpy.ndarray
        Global acceleration vector.
    """

    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass(frozen=True)
class NewmarkIntegrator:
    """Average-acceleration Newmark integrator for linear dynamics
    with a time-varying rank-1 contact-stiffness contribution.

    The static part of the effective stiffness ``K_static + a1 * C + a0 * M`` 
    is assembled once and stored as a sparse matrix. 
    At each step the contact contribution ``K * d(t) d(t).T`` 
    is added and the resulting system is factorisedand solved.

    Uses ``beta = 0.25`` and ``gamma = 0.5`` (unconditionally stable
    for linear systems).

    Attributes
    ----------
    dt : float
        Fixed time step size.
    beta : float
        Newmark beta parameter.
    gamma : float
        Newmark gamma parameter.
    a0, a1, a2, a3, a4, a5 : float
        Newmark integration coefficients.
    static_equivalent_stiffness : scipy.sparse.csc_matrix or None
        Static effective stiffness on free DOFs.
    free_dofs : numpy.ndarray or None
        Indices of unconstrained degrees of freedom.
    """

    dt: float
    beta: float = 0.25
    gamma: float = 0.5
    a0: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    a3: float = 0.0
    a4: float = 0.0
    a5: float = 0.0
    static_equivalent_stiffness: sparse.csc_matrix | None = None
    free_dofs: np.ndarray | None = None

    @classmethod
    def from_model(
        cls,
        mass: sparse.csr_matrix,
        stiffness: sparse.csr_matrix,
        damping: sparse.csr_matrix,
        free_dofs: np.ndarray,
        dt: float,
    ) -> NewmarkIntegrator:
        """Build an integrator and assemble the static effective stiffness.

        The static effective stiffness ``K + a1 * C + a0 * M`` is stored
        on free DOFs; the time-varying contact contribution is added
        and factorised inside :meth:`step`.

        Parameters
        ----------
        mass : scipy.sparse.csr_matrix
            Global mass matrix.
        stiffness : scipy.sparse.csr_matrix
            Global stiffness matrix.
        damping : scipy.sparse.csr_matrix
            Global damping matrix.
        free_dofs : numpy.ndarray
            Indices of unconstrained degrees of freedom.
        dt : float
            Fixed time step size.

        Returns
        -------
        NewmarkIntegrator
            Configured integrator ready for time stepping.
        """
        beta = 0.25
        gamma = 0.5
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)

        static_equivalent = stiffness + a1 * damping + a0 * mass
        static_equivalent_free = static_equivalent[free_dofs, :][:, free_dofs].tocsr()

        return cls(
            dt=dt,
            beta=beta,
            gamma=gamma,
            a0=a0,
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            a5=a5,
            static_equivalent_stiffness=static_equivalent_free,
            free_dofs=free_dofs,
        )

    def initial_state(self, n_dof: int) -> NewmarkState:
        """Create a zero initial state for all DOFs.

        Parameters
        ----------
        n_dof : int
            Number of global degrees of freedom.

        Returns
        -------
        NewmarkState
            State with zero displacement, velocity, and acceleration.
        """
        return NewmarkState(
            displacement=np.zeros(n_dof),
            velocity=np.zeros(n_dof),
            acceleration=np.zeros(n_dof),
        )

    def step(
        self,
        state: NewmarkState,
        contact_direction: np.ndarray,
        contact_stiffness: float,
        external_force: np.ndarray,
        mass: sparse.csr_matrix,
        damping: sparse.csr_matrix,
    ) -> NewmarkState:
        """Advance the state by one Newmark time step.

        The external load is a constant force vector ``external_force``
        (e.g. gravity applied to the moving-mass DOF). The contact
        coupling between the moving mass and the string at the current
        load position is added as a rank-1 update
        ``contact_stiffness * d d.T``, where ``d = contact_direction``.

        Parameters
        ----------
        state : NewmarkState
            State at the current time.
        contact_direction : numpy.ndarray
            Direction vector ``d(t) = [N(t); -1]`` of length ``n_dof``.
        contact_stiffness : float
            Contact spring stiffness ``K``.
        external_force : numpy.ndarray
            Constant nodal force vector (gravity on the mass DOF).
        mass : scipy.sparse.csr_matrix
            Global mass matrix.
        damping : scipy.sparse.csr_matrix
            Global damping matrix.

        Returns
        -------
        NewmarkState
            Updated displacement, velocity, and acceleration.
        """
        free = self.free_dofs
        d = contact_direction
        d_free = d[free]

        # Rank-1 contact stiffness update on free DOFs.
        # Built as a sparse outer product to keep matrix arithmetic sparse.
        d_sp = sparse.csr_matrix(d_free.reshape(-1, 1))
        contact_block = contact_stiffness * (d_sp @ d_sp.T)
        equivalent_free = (self.static_equivalent_stiffness + contact_block).tocsc()
        factorization = splu(equivalent_free)

        rhs = (
            external_force
            + mass    @ (self.a0 * state.displacement + self.a2 * state.velocity + self.a3 * state.acceleration)
            + damping @ (self.a1 * state.displacement + self.a4 * state.velocity + self.a5 * state.acceleration)
        )

        displacement = np.zeros_like(state.displacement)
        displacement[free] = factorization.solve(rhs[free])

        velocity = self.a1 * (displacement - state.displacement) - self.a4 * state.velocity - self.a5 * state.acceleration
        acceleration = self.a0 * (displacement - state.displacement) - self.a2 * state.velocity - self.a3 * state.acceleration

        return NewmarkState(
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
        )
