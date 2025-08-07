import numpy as np


def calculate_modesum( field: np.ndarray,
					   theta: np.ndarray ) -> np.ndarray:

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
                                 theta: np.ndarray ) -> np.ndarray:
    """
    Reconstruct scalar field on a cylindrical grid

    sclar_field: numpy array of the shape ( 2*Nmodes + 1 , Nr , Nz )
    theta: numpy array of the shape ( Ntheta, ) 

    returns the sum over all modes for given theta
    shape of output array is  ( Ntheta, Nr, Nz ) 
    """
    return calculate_modesum( sclar_field, theta )


def vector_field_reconstruction( vector_field: np.ndarray,
                                 theta: np.ndarray ) -> np.ndarray:
    """
    Reconstruct vector field on a cylindrical grid

    vector_field: (v_t, v_r, v_z) cylindrical components where each component is a numpy array of the shape ( 2*Nmodes + 1 , Nr , Nz )
    theta: numpy array of the shape ( Ntheta, ) 

    returns the sum over all modes for given theta
    shape of output array is  ( Ntheta, Nr, Nz, 3 ) 

    """

    thetagrid  = theta[:, np.newaxis, np.newaxis]

    # Reconstruct cylindrical components of the vector field
    v_t, v_r, v_z  = ( calculate_modesum(v, theta) for v in vector_field )

    # Convert cylindrical components to Cartesian components
    v_x = v_r * np.cos(thetagrid) - v_t * np.sin(thetagrid)
    v_y = v_r * np.sin(thetagrid) + v_t * np.cos(thetagrid)
    v_z = v_z

    # Combine the components into a single array
    return (v_x, v_y, v_z)



