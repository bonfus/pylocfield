import numpy as np

from pylocfield.realspace import compute_magnetic_moments
from ase.atoms import Atoms


def save_mcif(label: str, atoms: Atoms, supercell_matrix: np.array):
    """
    This function assumes that one q is specified for
    each pair of (+q, -q) vectors.
    """
    from pymatgen.core import Structure
    from ase.build.supercells import make_supercell

    na = len(atoms)
    fcs = atoms.get_array("fc")
    qs = atoms.info["q"].reshape(-1, 3)

    cell = atoms.cell

    sc = make_supercell(atoms, supercell_matrix, order="cell-major", wrap=False)

    R = cell.cartesian_positions(np.floor(cell.scaled_positions(sc.positions[0::na])))

    # Check that the same happens if we use the second atom as reference
    if na > 1:
        assert(np.allclose(
            cell.cartesian_positions(
                np.floor(cell.scaled_positions(sc.positions[1::na]))
            ),
            R,
        ))

    for i, q in enumerate(qs):
        mom = compute_magnetic_moments(R, q, fcs[:, i, :]).reshape(-1, 3)

        sc.set_initial_magnetic_moments(mom)

        st = Structure.from_ase_atoms(sc)
        st.to(f"{label}-{st.chemical_system}-q{i + 1}.mcif")
