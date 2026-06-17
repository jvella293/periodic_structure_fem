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
    with a time-varying rank-1 contact-stiffness contribution, solve
    via Sherman-Morrison

    The static effective stiffness ``A = K_static + a1 * C + a0 * M``
    is assembled and LU-factorised once at construction. Each step
    folds in the time-varying contact contribution ``K * d(t) d(t).T``
    using the Sherman-Morrison identity, costing only two
    back-substitutions per step rather than a full re-factorisation.

    Uses ``beta = 0.25`` and ``gamma = 0.5`` (unconditionally stable
    for linear systems); the time-varying rank-1 update
    introduces a small parametric perturbation but is benign in
    practice for slowly-moving contact points).

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
    static_factorization : scipy.sparse.linalg.splu or None
        Pre-computed LU factorisation of the static effective
        stiffness ``A`` on free DOFs.
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
    static_factorization: splu | None = None
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

        The static effective stiffness ``A = K + a1 * C + a0 * M`` is
        LU-factorised once on free DOFs. The factorisation is reused at
        every time step inside :meth:`step` via Sherman-Morrison.

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
        static_factorization = splu(static_equivalent_free)

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
            static_factorization=static_factorization,
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
        """Advance the state by one Newmark time step via Sherman-Morrison.

        The static effective stiffness ``A`` is already factorised. The
        rank-1 contact update ``K * d d.T`` is folded in cheaply:

            (A + K d d.T)^-1 b = A^-1 b
                                 - (K * (A^-1 d) * (d.T @ A^-1 b))
                                 / (1 + K * d.T @ A^-1 d)

        Costs two back-substitutions per step.

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
        d_free = contact_direction[free]

        rhs = (
            external_force
            + mass    @ (self.a0 * state.displacement + self.a2 * state.velocity + self.a3 * state.acceleration)
            + damping @ (self.a1 * state.displacement + self.a4 * state.velocity + self.a5 * state.acceleration)
        )
        b_free = rhs[free]

        # Two back-substitutions: A^-1 b and A^-1 d.
        y = self.static_factorization.solve(b_free)
        w = self.static_factorization.solve(d_free)

        # Sherman-Morrison correction.
        denom = 1.0 + contact_stiffness * (d_free @ w)
        coef = contact_stiffness * (d_free @ y) / denom

        displacement = np.zeros_like(state.displacement)
        displacement[free] = y - coef * w

        velocity = self.a1 * (displacement - state.displacement) - self.a4 * state.velocity - self.a5 * state.acceleration
        acceleration = self.a0 * (displacement - state.displacement) - self.a2 * state.velocity - self.a3 * state.acceleration

        return NewmarkState(
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
        )
