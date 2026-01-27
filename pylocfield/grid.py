import numpy as np

def gen_grid(cell, cutoff: float, remove_origin=False):
    """
    Generate a Cartesian grid of lattice vectors within a spherical cutoff.

    This function constructs all integer lattice vectors
    :math:`\\mathbf{N} = (n_1, n_2, n_3)` whose corresponding Cartesian
    vectors lie within a sphere of radius ``cutoff``. The lattice is defined
    by the cell vectors contained in ``cell``.

    Parameters
    ----------
    cell : object
        An ASE cell object defining the lattice geometry.

    cutoff : float
        Radial cutoff in Cartesian space in Angstrom.

    remove_origin : bool, optional
        If ``True``, the lattice vector ``(0, 0, 0)`` is excluded from the
        output. Default is ``False``.

    Returns
    -------
    C : ndarray of shape (M, 3)
        Cartesian coordinates of the lattice points inside the
        cutoff radius.

    Examples
    --------
    Generate reciprocal lattice vectors within a cutoff of 10.0
    (excluding the origin):

    >>> G = gen_grid(cell, cutoff=10.0, remove_origin=True)
    >>> G.shape
    (M, 3)
    """


    # Estimate indexes needed
    n1, n2, n3 = np.ceil(
                    np.linalg.norm(cutoff*cell.reciprocal(), axis=1)
                ).astype(int)

    # create the grid
    #n1, n2, n3 = np.meshgrid(, n2, n3, indexing='ij')
    i, j, k = np.indices((2*n1 + 1, 2*n2 + 1, 2*n3 + 1))
    i = i - n1
    j = j - n2
    k = k - n3

    # stack into vectors of shape (..., 3)
    N = np.stack([i, j, k], axis=-1)

    # remove (0,0,0)
    if remove_origin:
        mask_G0 = np.any(N != 0, axis=-1).reshape(-1)
    else:
        mask_G0 = True

    # convert to Cartesian coordinates
    C = cell.cartesian_positions(N.reshape(-1, 3))  # shape (Ntot, 3)

    # norm cutoff
    mask_Cmax = np.linalg.norm(C, axis=1) <= cutoff

    # apply both masks
    return C[mask_G0 & mask_Cmax]
