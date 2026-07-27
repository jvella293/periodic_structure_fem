from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


# Single source of truth for the Newmark scheme parameters.
# gamma = 0.5 is the non-dissipative average-acceleration scheme.
# Set gamma > 0.5 (with beta = 0.25*(gamma+0.5)**2) to add algorithmic
# damping that suppresses the spurious high-frequency mesh mode.
#
# NOTE for stability studies: gamma > 0.5 biases Re(lambda) DOWNWARD and can
# make a genuinely unstable point look stable. Use gamma = 0.5 when measuring
# growth rates.
NEWMARK_GAMMA = 0.5
NEWMARK_BETA = 0.25 * (NEWMARK_GAMMA + 0.5) ** 2


@dataclass
class NewmarkState:
    """State vector at one time step of Newmark integration."""

    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass(frozen=True)
class NewmarkIntegrator:
    """Average-acceleration Newmark integrator with a time-varying rank-1
    contact-stiffness contribution, solved via Sherman-Morrison.

    The static effective stiffness A = K_static + a1*C + a0*M is assembled
    and LU-factorised once at construction.

    Performance notes
    -----------------
    The naive implementation costs TWO back-substitutions per step: one for
    A^-1 b (b changes every step, unavoidable) and one for A^-1 d. But d(t)
    only changes when the contact point crosses into a new element, which
    happens every ``element_length / (V * dt)`` steps — 9 steps at typical
    settings. So ``step`` takes A^-1 d as an argument and the caller reuses
    it; see ContactSolveCache. This nearly halves the cost per step.

    d(t) is also never formed densely. It has exactly three nonzeros (two
    string nodes and the contact DOF), so the Sherman-Morrison inner
    products d.T @ y and d.T @ w are O(1) rather than O(n_dof).
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
    all_free: bool = True

    @classmethod
    def from_model(
        cls,
        mass: sparse.csr_matrix,
        stiffness: sparse.csr_matrix,
        damping: sparse.csr_matrix,
        free_dofs: np.ndarray,
        dt: float,
    ) -> NewmarkIntegrator:
        """Build an integrator and LU-factorise the static effective stiffness."""
        beta = NEWMARK_BETA
        gamma = NEWMARK_GAMMA
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)

        n_dof = mass.shape[0]
        all_free = bool(
            free_dofs.size == n_dof and np.array_equal(free_dofs, np.arange(n_dof))
        )

        static_equivalent = stiffness + a1 * damping + a0 * mass
        if all_free:
            # No constraints: skip the fancy-index copy entirely.
            static_equivalent_free = static_equivalent.tocsc()
        else:
            static_equivalent_free = static_equivalent[free_dofs, :][:, free_dofs].tocsc()

        return cls(
            dt=dt, beta=beta, gamma=gamma,
            a0=a0, a1=a1,
            a2=1.0 / (beta * dt),
            a3=1.0 / (2.0 * beta) - 1.0,
            a4=gamma / beta - 1.0,
            a5=dt * (gamma / (2.0 * beta) - 1.0),
            static_factorization=splu(static_equivalent_free),
            free_dofs=free_dofs,
            all_free=all_free,
        )

    def initial_state(self, n_dof: int) -> NewmarkState:
        """Create a zero initial state for all DOFs."""
        return NewmarkState(
            displacement=np.zeros(n_dof),
            velocity=np.zeros(n_dof),
            acceleration=np.zeros(n_dof),
        )

    def solve_static(self, vector: np.ndarray) -> np.ndarray:
        """Apply A^-1 to a full-length vector, returning a full-length result."""
        if self.all_free:
            return self.static_factorization.solve(vector)
        out = np.zeros_like(vector)
        out[self.free_dofs] = self.static_factorization.solve(vector[self.free_dofs])
        return out

    def step(
        self,
        state: NewmarkState,
        contact_dofs: np.ndarray,
        contact_values: np.ndarray,
        contact_solved: np.ndarray,
        contact_stiffness: float,
        external_force: np.ndarray,
        mass: sparse.csr_matrix,
        damping: sparse.csr_matrix,
    ) -> NewmarkState:
        """Advance the state by one Newmark step via Sherman-Morrison.

            (A + K d d.T)^-1 b = A^-1 b
                                 - K (A^-1 d) (d.T A^-1 b) / (1 + K d.T A^-1 d)

        Parameters
        ----------
        state : NewmarkState
            State at the current time.
        contact_dofs : numpy.ndarray
            The three global DOF indices where d(t) is nonzero:
            ``[node_left, node_right, contact_dof]``.
        contact_values : numpy.ndarray
            The matching nonzero values ``[N_left, N_right, -1.0]``.
        contact_solved : numpy.ndarray
            ``A^-1 d`` for this d, full length. Supplied by the caller so it
            can be reused across the ~9 steps that share an element.
        contact_stiffness : float
            Contact spring stiffness ``K``.
        external_force : numpy.ndarray
            Constant nodal force vector (zero in perturbation runs).
        mass, damping : scipy.sparse.csr_matrix
            Global mass and damping matrices.

        Returns
        -------
        NewmarkState
            Updated displacement, velocity, and acceleration.
        """
        rhs = (
            external_force
            + mass @ (self.a0 * state.displacement
                      + self.a2 * state.velocity
                      + self.a3 * state.acceleration)
            + damping @ (self.a1 * state.displacement
                         + self.a4 * state.velocity
                         + self.a5 * state.acceleration)
        )

        y = self.solve_static(rhs)
        w = contact_solved

        # d has three nonzeros: both inner products are O(1).
        d_dot_w = float(contact_values @ w[contact_dofs])
        d_dot_y = float(contact_values @ y[contact_dofs])

        denom = 1.0 + contact_stiffness * d_dot_w
        coef = contact_stiffness * d_dot_y / denom

        displacement = y - coef * w

        velocity = (self.a1 * (displacement - state.displacement)
                    - self.a4 * state.velocity - self.a5 * state.acceleration)
        acceleration = (self.a0 * (displacement - state.displacement)
                        - self.a2 * state.velocity - self.a3 * state.acceleration)

        return NewmarkState(
            displacement=displacement,
            velocity=velocity,
            acceleration=acceleration,
        )


class ContactSolveCache:
    """Supplies ``A^-1 d(t)`` while avoiding a solve on most steps.

    d(t) = N_left * e_left + N_right * e_right - e_contact, so

        A^-1 d = N_left * (A^-1 e_left) + N_right * (A^-1 e_right) - A^-1 e_contact

    ``A^-1 e_contact`` is constant (solved once). The two nodal columns change
    only when the load crosses into a new element, and the incoming element's
    left column is the outgoing element's right column — so a single solve per
    element crossing suffices, not one per time step.
    """

    def __init__(self, integrator: NewmarkIntegrator, n_dof: int, contact_dof: int):
        self.integrator = integrator
        self.n_dof = n_dof
        self.contact_dof = contact_dof

        unit = np.zeros(n_dof)
        unit[contact_dof] = 1.0
        self.col_contact = integrator.solve_static(unit)

        self._node_left: int | None = None
        self._node_right: int | None = None
        self._col_left: np.ndarray | None = None
        self._col_right: np.ndarray | None = None
        self.n_solves = 0

    def _column(self, node: int) -> np.ndarray:
        unit = np.zeros(self.n_dof)
        unit[node] = 1.0
        self.n_solves += 1
        return self.integrator.solve_static(unit)

    def get(self, node_left: int, node_right: int,
            w_left: float, w_right: float) -> np.ndarray:
        """Return ``A^-1 d`` for the element spanning the two given nodes."""
        if node_left != self._node_left or node_right != self._node_right:
            # Reuse whichever cached column still applies (usually the
            # previous right node becomes the new left node).
            new_left = (self._col_right if node_left == self._node_right
                        else self._col_left if node_left == self._node_left
                        else None)
            new_right = (self._col_left if node_right == self._node_left
                         else self._col_right if node_right == self._node_right
                         else None)
            self._col_left = new_left if new_left is not None else self._column(node_left)
            self._col_right = new_right if new_right is not None else self._column(node_right)
            self._node_left, self._node_right = node_left, node_right

        return w_left * self._col_left + w_right * self._col_right - self.col_contact
