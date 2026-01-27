import numpy as np
import spglib
from ase import Atoms


def real_imag_phase_to_fc(atoms: Atoms):
    """
    Converts description based on Real, Imaginary and Phase
    parts of the fourier coefficients into a single complex quantity.

    See Section 5 of EPJ Web of Conferences 22, 00010 (2012)
    DOI: 10.1051/epjconf/20122200010
    """

    nat = len(atoms)
    nqs = len(atoms.info["q"].reshape((-1, 3)))

    ri = atoms.get_array("ri").reshape((nat, nqs, 3))
    phi = atoms.get_array("phi").reshape((nat, nqs, 1))

    fc = 0.5 * ri * np.exp(-2.0j * np.pi * phi)
    return fc


def add_equivalent_muon_sites(atoms: Atoms):
    """
    Add equivalent muon sites according to crystal symmetries.
    Muon positions are taken from atoms.info['mu'], which is
    updated with the new positions found.
    """
    mup = atoms.info["mu"]
    mup.shape = (-1, 3)

    positions = get_symmetry_equivalent_positions(atoms, mup)
    atoms.info["mu"] = positions

def get_symmetry_equivalent_positions(
    atoms: Atoms,
    frac_position: np.array,
    magnetic_symmetry: bool = False,
    symprec: float = 1e-5,
):
    """
    Find symmetry-equivalent fractional positions for a given coordinate,
    optionally considering collinear magnetic symmetry.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE atoms object
    frac_position : array-like, shape (3,)
        Fractional coordinates of the position of interest
    magnetic_symmetry : bool
        If True, use spglib magnetic symmetry (requires atoms.get_initial_magnetic_moments() to return magnetic moments)
        If False, use crystallographic symmetry only
    symprec : float
        Symmetry tolerance for spglib

    Returns
    -------
    equiv_positions : np.ndarray, shape (N, 3)
        Unique symmetry-equivalent fractional positions in [0, 1)
    """

    frac_position = np.asarray(frac_position, dtype=float)

    # --- Build common cell data ---
    lattice = atoms.cell.array
    positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()

    if magnetic_symmetry:
        magmoms = atoms.get_initial_magnetic_moments()

        if magmoms is None:
            raise ValueError(
                "magnetic_symmetry=True requires non-zero magnetic moments."
            )

        cell = (lattice, positions, numbers, magmoms)
    else:
        cell = (lattice, positions, numbers)

    sym = spglib.get_symmetry(cell, symprec=symprec)

    rotations = sym["rotations"]
    translations = sym["translations"]

    equiv_positions = []
    for position in frac_position:
        for R, t in zip(rotations, translations):
            new_pos = R @ position + t
            equiv_positions.append(new_pos % 1.0)

    # --- Deduplicate (numerical tolerance) ---
    equiv_positions = np.unique(
        np.round(
            np.asarray(equiv_positions), decimals=np.log10(1 / symprec).astype(int)
        ),
        axis=0,
    )

    return equiv_positions
