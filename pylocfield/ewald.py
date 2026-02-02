import numpy as np

from ase.geometry import get_distances

from .misra import misra_m
from .grid import gen_grid


def compute_dipolar_tensor(
    atoms,
    mup: np.ndarray,
    q: np.ndarray,
    fcs: np.ndarray,
    R: np.ndarray,
    G: np.ndarray,
    rho: float = 1.0,
    eps: float = 1e-8,
):
    """
    Compute dipolar tensors using the Ewald summation approach.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing the crystal structure and
        additional required information.
    mup : np.ndarray
        Muon position in scaled (fractional) coordinates.
        Shape: (3,).
    q : np.ndarray
        Wave vector in the first Brillouin zone, given in scaled
        reciprocal-space coordinates.
        Shape: (3,).
    fcs : np.ndarray
        Fourier components used to select which atoms contribute
        to the sum. Atoms with |fcs| < eps are skipped.
    R : np.ndarray
        Real-space lattice points in Cartesian coordinates to perform the real space sum.
        Shape: (NR, 3).
    G : np.ndarray
        Reciprocal-space lattice point in Cartesian coordinates to perform reciprocal space sum.
        Shape: (NG, 3).
    rho : float, optional
        Ewald convergence parameter controlling the relative
        convergence of real- and reciprocal-space sums.
        Default is 1.0.
    eps : float, optional
        Threshold below which foureis components and q vectors are considered zero.
        Default is 1e-8.

    Returns
    -------
    dipolar_tensor : np.ndarray
        Computed dipolar tensor.
    """

    # Kronecker delta
    delta = np.equal

    # to Cartesian coordinates
    mup = atoms.cell.cartesian_positions(mup)
    q = atoms.cell.reciprocal().cartesian_positions(q) * 2*np.pi
    vc = atoms.get_volume()

    # get distance from basis atoms
    muds = -get_distances(mup, atoms.positions)[0]
    muds.shape = (-1,3)


    # initialize result array
    D = np.zeros([len(atoms), 3,3], dtype=complex)

    # often used below
    q2 = np.dot(q,q)
    G_q = G+q

    for idx, fc in enumerate(fcs):
        # fcs are only used to determine when to skip the computation because the tensor will
        # eventually be multiplied by 0
        if np.linalg.norm(fc) < eps:
            continue

        # Get distance from this atom
        mud = muds[idx]
        R_mud = R+mud

        # These are the elements appearing in the sum in Eq. 5.
        A = misra_m( (np.linalg.norm(G_q, axis=1)/(2*rho))**2 ,0) * np.exp(-1.j * np.dot(G, mud))
        B = misra_m( (np.linalg.norm(R_mud, axis=1)*rho)**2 , 3/2)
        C = misra_m( (np.linalg.norm(R_mud, axis=1)*rho)**2 , 1/2)

        E = np.exp(1.j * ((R+mud)@q))

        # pretty stupid to compute all elements, but that's so fast!
        for alpha in range(3):
            for beta in range(3):
                if q2 > eps**2:
                    D[idx, alpha, beta] += -4*np.pi * ((q[alpha]*q[beta])/q2 ) * np.exp(-q2/(4*rho**2))

                D[idx, alpha, beta] += -(np.pi/(rho**2)) * np.sum(((G_q)[:,alpha]) * ((G_q)[:, beta]) * A)

                D[idx, alpha, beta] +=  ((2*vc*rho**3 )/(np.pi**(1/2))) * \
                                    np.sum ( (  2*(rho**2) * (R_mud)[:,alpha] * (R_mud)[:,beta] * B - delta(alpha,beta)*C) * E )


    return D


def compute_dipolar_tensors(atoms, r_c: float = 12.0, Gc: float = 12.0):
    """
    Compute dipolar tensors for multiple q vectors and muon positions.

    This function evaluates dipolar tensors using an Ewald-based approach
    for all q vectors and muon positions stored in the provided ASE
    ``Atoms`` object. Tensors are computed for each atom; however, only atoms
    with non-zero Fourier components are actually evaluated.

    The resulting dipolar tensors are stored in the ``atoms`` object
    under the key ``"D"`` with the following layout:

        D[n_atoms, n_mu, n_q, 3, 3]

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic positions, q vectors,
        muon positions, and Fourier components.
    r_c : float, optional
        Real-space cutoff radius (in Angstrom) used to truncate
        the Ewald real-space sum.
        Default is 12.0.
    Gc : float, optional
        Reciprocal-space cutoff radius (in Angstrom^-1) used to
        truncate the Ewald reciprocal-space sum.
        Default is 12.0.

    Returns
    -------
    None
        Dipolar tensors are stored directly in the ``atoms`` object.
        This allows to reuse them with different fourier components.
    """

    na = len(atoms)
    qs = atoms.info['q'].reshape((-1,3))
    nq = len(qs)

    # reorder FCs as nq, na, 3 (i.e axes go 0 1 2 -> 1 0 2)
    fcs = atoms.get_array("fc").reshape(na, nq, 3).transpose(1, 0, 2)

    # muon positions should be (nmu, 3)
    mups = atoms.info['mu'].reshape((-1,3))

    # Generate grids, gen_grid returns Cartesian positions
    R = gen_grid(atoms.cell, r_c)
    G = gen_grid(atoms.cell.reciprocal(), Gc, remove_origin=True)
    G *= 2*np.pi

    D = np.zeros([na, len(mups), len(qs), 3, 3], dtype=complex)
    for i, mup in enumerate(mups):
        for j, q in enumerate(qs):
            D[:, i, j, :, :] = compute_dipolar_tensor(atoms, mup, q, fcs[j], R, G)

    # reset previous value
    atoms.set_array("D", None)
    # dipolar tensors for each atom with order mu,q,3x3
    atoms.set_array("D", D)


def compute_field(atoms, use_cc: bool = True):
    """
    Compute the dipolar magnetic field at muon sites.

    This function evaluates the dipolar field at the muon positions
    stored in the provided ASE ``Atoms`` object, using precomputed
    dipolar tensors. The dipolar tensors must already be present
    in the ``atoms`` object (e.g. under the key ``"D"``).

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing muon positions, magnetic moments,
        and precomputed dipolar tensors.
    use_cc : bool, optional
        If True, assume that the dipolar tensors for couples +q and -q appear only once.
        If False, each q vector is treated independently and you should explicitly provide +q and -q tensors.

    Returns
    -------
    field : np.ndarray
        Dipolar magnetic field evaluated at each muon site.
        For k=0 magnetic orders, it include the Lorentz field.
        Shape is [nmu, nq, 3]
    """

    #cell
    na = len(atoms)
    vc = atoms.get_volume()

    # mag info
    qs = atoms.info['q'].reshape((-1,3))
    nq = len(qs)
    fcs = atoms.get_array("fc").reshape(na, nq, 3).transpose(1, 0, 2)

    mups = atoms.info['mu'].reshape((-1,3))

    reciprocal_cell = atoms.cell.reciprocal()

    # previously calculated
    D = atoms.get_array('D')

    # initialize output
    B = np.zeros([len(mups), len(qs), 3], dtype=complex)

    for i, mup in enumerate(mups):
        mup = atoms.cell.cartesian_positions(mup)
        # distance from magnetic atoms
        r0 = -get_distances(mup, atoms.positions)[0]
        r0.shape = (-1,3)

        for j, q in enumerate(qs):

            # to cartesian
            q = reciprocal_cell.cartesian_positions(q) * 2 * np.pi

            # (μ_0/4pi) μ_B = 0.927 401 01 T·Å³
            #b = (0.92740101 / vc) * np.einsum('i,nij,ij->j', np.exp(-1.j * (r0 @ q)), D[i,j], fcs[j])
            b = (0.92740101 / vc) * np.einsum(
                "n,nij,nj->i", np.exp(-1.0j * (r0 @ q)), D[:, i, j], fcs[j]
            )
            # see Eq. 5.63
            B[i, j] += 2*b.real if use_cc else b

    return np.real_if_close(B)



