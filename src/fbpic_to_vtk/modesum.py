import numpy as np


def calculate_modesum( field: np.ndarray,
					   theta: np.ndarray 
                     ) -> np.ndarray:

    """
    Reconstruct 3D diagnostic on a cylindrical grid

    field: numpy array of the shape ( 2*Nmodes + 1 , Nr , Nz )
    theta: numpy array of the shape ( Ntheta, ) 

    returns the sum over all modes for given theta
    shape of output array is  ( Ntheta, Nr, Nz ) 

    """

    assert field.ndim == 3, "Field must be a 3D array"
    assert theta.ndim == 1, "Field must be a 1D array"

    assert field.shape[0] % 2 == 1, "Field must have an odd number of modes"

    thetagrid  = theta[:, np.newaxis, np.newaxis]

    Nmodes = field.shape[0] // 2 + 1

    modefunctions = [ 1 + 0. * thetagrid ]

    for m in range( 1 , Nmodes ):

        modefunctions.append( np.cos( m*thetagrid ) )
        modefunctions.append( np.sin( m*thetagrid ) )

    return np.sum( [ mode * field[i] for i,mode in enumerate(modefunctions) ] , axis = 0 )


def scalar_field_reconstruction( sclar_field: np.ndarray,
                                 theta: np.ndarray
                               ) -> np.ndarray:
    """
    Reconstruct scalar field on a cylindrical grid

    sclar_field: numpy array of the shape ( 2*Nmodes + 1 , Nr , Nz )
    theta: numpy array of the shape ( Ntheta, ) 

    returns the sum over all modes for given theta
    shape of output array is  ( Ntheta, Nr, Nz ) 
    """
    return calculate_modesum( sclar_field, theta )


def vector_field_reconstruction(vector_field: np.ndarray,
                                theta: np.ndarray 
                               ) -> (np.ndarray,np.ndarray,np.ndarray):
    """
    Reconstruct vector field on a cylindrical grid

    vector_field: (v_t, v_r, v_z) cylindrical components where each component is a numpy array of the shape ( 2*Nmodes + 1 , Nr , Nz )
    theta: numpy array of the shape ( Ntheta, ) 

    returns cartesian componenets of the mode reconstructed vector field as a tuple
    the shape of each component array is  ( Ntheta, Nr, Nz ) 

    """

    thetagrid = theta[:, np.newaxis, np.newaxis]

    # Reconstruct cylindrical components of the vector field
    v_t, v_r, v_z  = ( calculate_modesum(v, theta) for v in vector_field )

    # Convert cylindrical components to Cartesian components
    v_x = v_r * np.cos(thetagrid) - v_t * np.sin(thetagrid)
    v_y = v_r * np.sin(thetagrid) + v_t * np.cos(thetagrid)
    v_z = v_z

    # Combine the components into a single array
    return (v_x, v_y, v_z)

def cartesian_coordinate_arrays( theta: np.ndarray,
                                 r: np.ndarray, 
                                 z: np.ndarray, 
                               ) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Convert cylindrical coordinates to Cartesian coordinates.

    Parameters:
        r: 1D array of radial coordinates.
        z: 1D array of axial coordinates.
        theta: 1D array of azimuthal coordinates.

    Returns:
        xgrid: broadcastable array of x-coordinates, shape (Ntheta, Nr, 1)
        ygrid: broadcastable array of y-coordinates, shape (Ntheta, Nr, 1)
        zgrid: broadcastable array of z-coordinates, shape (1, 1, Nz)
    """

    assert theta.ndim == 1, "Azimuthal coordinates must be a 1D array"
    assert r.ndim == 1, "Radial coordinates must be a 1D array"
    assert z.ndim == 1, "Axial coordinates must be a 1D array"


    Ntheta = len(theta)
    Nr = len(r)
    Nz = len(z)

    thetagrid = theta[:, np.newaxis, np.newaxis]
    rgrid = theta[np.newaxis, :, np.newaxis]
    zgrid = z[np.newaxis, np.newaxis, :]


    xgrid = rgrid * np.cos(thetagrid)
    ygrid = rgrid * np.sin(thetagrid)

    # Ensure the shapes are broadcastable
    assert xgrid.shape == (Ntheta, Nr, 1), "x-coordinates shape mismatch"
    assert ygrid.shape == (Ntheta, Nr, 1), "y-coordinates shape mismatch"
    assert zgrid.shape == (1, 1, Nz), "z-coordinates shape mismatch"

    return xgrid, ygrid, zgrid
