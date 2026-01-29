import numpy as np
from ase import Atoms
from .grid import gen_grid

def compute_magnetic_moments(R: np.array, q: np.array, S: np.array):
    """
    This assumes that only q (or -q) has been specified.
    """
    Rq = R @ q
    moments = np.einsum('ij,k->kij', S, np.exp(-1.j * Rq))
    moments += np.einsum('ij,k->kij', np.conj(S), np.exp(1.j * Rq))

    return np.real_if_close(moments)


def compute_field(atoms: Atoms, Rc: float = 100.0):
    """
    Real-space dipolar field.

    Parameters
    ----------
    atoms : ase.Atoms
        Unit cell, with muon positions, propagation vectors and fourier components already stored in "info['mu']", "info['q']" and "get_array('fc')".
    Rc : float or None
        Real-space cutoff in Angstrom.

    Returns
    -------
    B : (3,) ndarray
        Dipolar field, including the Lorentz field.
    """

    cell = atoms.cell

    na = len(atoms)
    positions = atoms.positions

    qs = atoms.info["q"].reshape((-1, 3))
    nqs = len(qs)

    mups = atoms.info["mu"].reshape((-1, 3))

    S = atoms.get_array("fc").reshape((na, nqs, 3))
    S = np.swapaxes(S, 0, 1)

    # To Cartesian
    qs = cell.reciprocal().cartesian_positions(qs) * 2 * np.pi
    mups = cell.cartesian_positions(mups)

    # This is already in cartesian coordinates
    R = gen_grid(cell, Rc)

    # what is faster?
    # sc_positions = positions[:, None, :] + R[None, :, :]
    sc_positions = positions[None, :, :] + R[:, None, :]

    B = np.zeros([len(mups), len(qs), 3])
    B_L = np.zeros([len(mups), len(qs), 3])

    for i, q in enumerate(qs):
        # compute moments for all cells
        moments = compute_magnetic_moments(R, q, S[i])
        for j, mup in enumerate(mups):

            # --- Dipolar field calculation ---
            r = sc_positions - mup
            r2 = np.sum(r * r, axis=2)

            # remove positions ooutside the sphere
            mask = r2 < Rc**2
            r = r[mask]
            r2 = r2[mask]
            m = moments[mask]

            r_norm = np.sqrt(r2)
            del r2

            r_hat = r / r_norm[:, None]

            m_dot_r = np.einsum('ij,ij->i', m, r_hat)

            inv_r3 = 1.0 / r_norm**3
            del r_norm

            B[j, i] = 0.92740101 * (
                np.einsum('i,ij->j', 3.0 * m_dot_r * inv_r3, r_hat)
                - np.einsum('i,ij->j', inv_r3, m)
            )

            # Sligtly slower:
            #
            #B[j, i] = 0.92740101 * np.sum(
            #    (3.0 * mu_dot_r[:, None] * r_hat - m) / r_norm[:, None]**3,
            #    axis=0
            #)

            # --- Lorentz field calculation ---
            # magnetic_constant * 1 bohr_magneton = 11.654064 T⋅Å^3
            B_L[j, i] = (1./3.) * 11.654064 *  np.sum(m, axis=0) / ((4/3) * np.pi * Rc**3)

    return B + B_L

