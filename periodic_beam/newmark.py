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
    """Average-acceleration Newmark integrator for linear dynamics.

    Uses ``beta = 0.25`` and ``gamma = 0.5`` (unconditionally stable
    for linear systems). The effective stiffness matrix is factorized
    once at construction.

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
    equivalent_stiffness : scipy.sparse.csc_matrix or None
        Effective stiffness on free DOFs.
    factorization : scipy.sparse.linalg.splu or None
        LU factorization of ``equivalent_stiffness``.
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
    equivalent_stiffness: sparse.csc_matrix | None = None
    factorization: splu | None = None
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
        """Build an integrator with pre-factorized effective stiffness.

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

        equivalent = stiffness + a1 * damping + a0 * mass
        equivalent_free = equivalent[free_dofs, :][:, free_dofs].tocsc()
        factorization = splu(equivalent_free)

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
            equivalent_stiffness=equivalent_free,
            factorization=factorization,
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
        load_shape: np.ndarray,
        force: float,
        omega_p: float,
        time: float,
        mass: sparse.csr_matrix,
        damping: sparse.csr_matrix,
    ) -> NewmarkState:
        """Advance the state by one Newmark time step.

        The external load is a point load of magnitude ``force``,
        modulated by ``cos(omega_p * time)`` and distributed via
        ``load_shape``.

        Parameters
        ----------
        state : NewmarkState
            State at the current time.
        load_shape : numpy.ndarray
            Shape vector for a unit vertical point load.
        force : float
            Point load magnitude [N].
        omega_p : float
            Parametric excitation circular frequency.
        time : float
            Current simulation time.
        mass : scipy.sparse.csr_matrix
            Global mass matrix.
        damping : scipy.sparse.csr_matrix
            Global damping matrix.

        Returns
        -------
        NewmarkState
            Updated displacement, velocity, and acceleration.
        """
        load_term = -force * load_shape * np.cos(omega_p * time)
        rhs = (
            load_term
            + mass @ (self.a0 * state.displacement + self.a2 * state.velocity + self.a3 * state.acceleration)
            + damping @ (self.a1 * state.displacement + self.a4 * state.velocity + self.a5 * state.acceleration)
        )

        displacement = np.zeros_like(state.displacement)
        displacement[self.free_dofs] = self.factorization.solve(rhs[self.free_dofs])

        velocity = self.a1 * (displacement - state.displacement) - self.a4 * state.velocity - self.a5 * state.acceleration
        acceleration = self.a0 * (displacement - state.displacement) - self.a2 * state.velocity - self.a3 * state.acceleration

        return NewmarkState(
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
        )
