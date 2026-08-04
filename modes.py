"""True coupled modes of the string + 2-DOF oscillator system.

Why this exists
---------------
The rigid-string estimate (a 2x2 eigenproblem with the contact point held
fixed) is a crude orientation figure, and it is NOT a reliable bound: in
the string's stopband the driving-point reactance can be mass-like, which
pushes a coupled mode ABOVE the rigid-string value. Using it as a
pass/fail criterion produces wrong answers.

This module solves the actual generalised eigenproblem

    K(x_c) phi = omega^2 M phi

with the contact spring attached at a frozen position x_c, using the same
FEM assembly the time-domain solver uses. Because the system is
time-periodic, the modes BREATHE as the contact traverses a cell: each
mode is a band, not a single frequency. Sweeping x_c over one cell gives
that band, which is what an observed spectral line should be compared to.

Why it matters for spectrum.py
------------------------------
A Floquet comb contains f0 and (p*f_pass - f0) whatever the mechanism. To
tell a genuine combination resonance from one mode with parametric
sidebands, you must ask whether there is a real MODE at each of the two
frequencies. Only the true modes can answer that — and mode-shape contrast
cannot substitute, because a sideband is a forced off-resonant response
whose |z2|/|z1| differs from any modal ratio.

Usage
-----
    python modes.py                     # uses the parameters in main.py
    python modes.py --n-positions 12 --n-modes 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from periodic_string.assembly import assemble_model, mesh_parameters
from periodic_string.moving_load import contact_direction

CACHE = Path("cache") / "mode_bands.json"


def modes_at(x_contact, *, tension, mass_per_length, kv, k01, k12,
             m1, m2, element_length, n_elements_per_cell, n_nodes,
             catenary_length, n_modes=6, single_dof=False):
    """Modes with the contact spring frozen at ``x_contact``.

    Returns
    -------
    freqs : numpy.ndarray
        Natural frequencies [Hz], ascending.
    shape_ratio : numpy.ndarray
        |z2| / |z1| for each mode — the modal signature used to tell
        modes apart.
    osc_fraction : numpy.ndarray
        Share of the mode's kinetic energy carried by the two oscillator
        DOFs. Near 1 => oscillator mode; near 0 => string mode.
    """
    model = assemble_model(
        tension=tension, damp_string=0.0, mass_per_length=mass_per_length,
        kv=kv, damp_rp=0.0, head_mass=m1, frame_mass=m2,
        susp_stiffness=k12, susp_damping=0.0,
        base_stiffness=0.0, base_damping=0.0, contact_stiffness=k01,
        element_length=element_length,
        n_elements_per_cell=n_elements_per_cell,
        n_nodes=n_nodes, catenary_length=catenary_length,
        show_progress=False,
    )
    d = contact_direction(model.node_x, x_contact, catenary_length,
                          model.n_dof, model.contact_dof)
    ds = sparse.csr_matrix(d.reshape(-1, 1))
    K = (model.stiffness + k01 * (ds @ ds.T)).tocsc()
    M = model.mass.tocsc()

    if single_dof:
        # Drop the constrained frame DOF so it contributes no spurious mode.
        keep = np.delete(np.arange(model.n_dof), model.frame_dof)
        K = K[keep, :][:, keep].tocsc()
        M = M[keep, :][:, keep].tocsc()

    vals, vecs = eigsh(K, k=n_modes, M=M, sigma=0.0, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    freqs = np.sqrt(np.clip(vals, 0.0, None)) / (2 * np.pi)

    c_dof, f_dof = model.contact_dof, model.frame_dof
    ratio, osc = [], []
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        z1 = abs(v[c_dof])
        z2 = 0.0 if single_dof else abs(v[f_dof])
        ratio.append(z2 / max(z1, 1e-30))
        ke_osc = m1 * z1 ** 2 + (0.0 if single_dof else m2 * z2 ** 2)
        ke_tot = float(v @ (M @ v))
        osc.append(ke_osc / max(ke_tot, 1e-30))
    return freqs, np.array(ratio), np.array(osc)


def mode_bands(*, tension, mass_per_length, kv, k01, k12, m1, m2,
               spacing, element_length_requested=0.1, n_cells=200,
               n_modes=6, n_positions=8, use_cache=True, single_dof=False):
    """Frequency band of each mode as the contact traverses one cell.

    The contact position is swept across a single periodic cell; the
    resulting min/max of each mode is the band an observed spectral line
    should be compared against.
    """
    key = json.dumps({
        "H": tension, "rhoA": mass_per_length, "kv": kv, "k01": k01,
        "k12": k12, "m1": m1, "m2": m2, "L": spacing,
        "el": element_length_requested, "nc": n_cells,
        "nm": n_modes, "np": n_positions, "sdof": single_dof,
    }, sort_keys=True)

    if use_cache and CACHE.exists():
        try:
            store = json.loads(CACHE.read_text())
            if store.get("key") == key:
                return (np.array(store["low"]), np.array(store["high"]),
                        np.array(store["ratio"]), np.array(store["osc"]))
        except (json.JSONDecodeError, KeyError):
            pass

    el, npc, nn, Lcat = mesh_parameters(spacing, element_length_requested,
                                        n_cells)
    x0 = 0.5 * Lcat
    freqs, ratios, oscs = [], [], []
    for j in range(n_positions):
        x = x0 + (j / n_positions) * spacing
        f, r, o = modes_at(
            x, tension=tension, mass_per_length=mass_per_length, kv=kv,
            k01=k01, k12=k12, m1=m1, m2=m2, element_length=el,
            n_elements_per_cell=npc, n_nodes=nn, catenary_length=Lcat,
            n_modes=n_modes, single_dof=single_dof)
        freqs.append(f)
        ratios.append(r)
        oscs.append(o)

    freqs = np.array(freqs)
    low, high = freqs.min(axis=0), freqs.max(axis=0)
    ratio = np.median(np.array(ratios), axis=0)
    osc = np.median(np.array(oscs), axis=0)

    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps({
        "key": key, "low": low.tolist(), "high": high.tolist(),
        "ratio": ratio.tolist(), "osc": osc.tolist(),
    }))
    return low, high, ratio, osc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-positions", type=int, default=8)
    ap.add_argument("--n-modes", type=int, default=6)
    ap.add_argument("--n-cells", type=int, default=200)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    import main as mn

    low, high, ratio, osc = mode_bands(
        tension=mn.tension, mass_per_length=mn.m, kv=mn.Kv,
        k01=mn.k01, k12=mn.k12, m1=mn.m1, m2=mn.m2, spacing=mn.spacing,
        element_length_requested=0.1, n_cells=args.n_cells,
        n_modes=args.n_modes, n_positions=args.n_positions,
        use_cache=not args.no_cache,
        single_dof=(mn.model == "sdof"),
    )

    f_pass = mn.V / mn.spacing
    print(f"m1 = {mn.m1:g} kg, m2 = {mn.m2:g} kg, "
          f"k01 = {mn.k01:g}, k12 = {mn.k12:g}")
    print(f"f_pass = {f_pass:.4f} Hz\n")
    print("True coupled modes, swept over one cell "
          f"({args.n_positions} contact positions):")
    print(f"  {'#':>2}  {'band [Hz]':>19}  {'|z2/z1|':>8}  {'osc. frac':>9}  "
          f"{'2f/f_pass':>10}")
    for i, (lo, hi) in enumerate(zip(low, high), 1):
        mid = 0.5 * (lo + hi)
        kind = "oscillator" if osc[i-1] > 0.5 else "string"
        print(f"  {i:>2}  {lo:8.4f} – {hi:8.4f}  {ratio[i-1]:8.3f}  "
              f"{osc[i-1]:9.3f}  {2*mid/f_pass:10.4f}   {kind}")

    print("\n  '2f/f_pass' near an integer n indicates a simple parametric")
    print("  resonance of that mode (n = 1 is the principal 2T tongue).")

    from itertools import combinations
    print("\nCombination conditions (f_i + f_j) / f_pass:")
    mid = 0.5 * (low + high)
    for i, j in combinations(range(len(mid)), 2):
        s = (mid[i] + mid[j]) / f_pass
        dm = abs(mid[i] - mid[j]) / f_pass
        flag = "  <-- near integer: possible SUM combination" if abs(s - round(s)) < 0.06 and round(s) >= 1 else ""
        print(f"  modes {i+1}+{j+1}: sum = {s:.4f}, diff = {dm:.4f}{flag}")


if __name__ == "__main__":
    main()
