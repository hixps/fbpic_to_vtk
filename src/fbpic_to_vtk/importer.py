import h5py
import numpy as np


def read_hdf5_file(source_filename, timestep, scalar_diag_names, vector_diag_names, downsample_3d=(slice(None), slice(None), slice(None)) ):
    """
    Reads scalar and vector diagnostics from an HDF5 file at a specified timestep.

    Parameters:
        source_filename (str): Path to the HDF5 file.
        timestep (int): The timestep to read data from.
        scalar_diag_names (list of str): List of scalar diagnostic names to read.
        vector_diag_names (list of str): List of vector diagnostic names to read.

    Returns:
        dict: A dictionary containing the requested scalar and vector diagnostics.
    """
    scalar_diags = {}
    vector_diags = {}

    with h5py.File(source_filename, 'r') as source:
        for name in scalar_diag_names:
            scalar_diags[name] = source[f'data/{timestep}/fields/{name}'][()][downsample_3d]
        
        for name in vector_diag_names:
            vector_diags[name] = (
                source[f'data/{timestep}/fields/{name}/t'][()][downsample_3d],
                source[f'data/{timestep}/fields/{name}/r'][()][downsample_3d],
                source[f'data/{timestep}/fields/{name}/z'][()][downsample_3d]
            )
    
    return scalar_diags, vector_diags


def read_rz_coordinates(source_filename, timestep, diag_name, downsample_r=None, downsample_z=None):
    """
    Reads scalar and vector diagnostics from an HDF5 file at a specified timestep.

    Parameters:
        source_filename (str): Path to the HDF5 file.
        timestep (int): The timestep to read data from.
        scalar_diag_names (list of str): List of scalar diagnostic names to read.
        vector_diag_names (list of str): List of vector diagnostic names to read.

    Returns:
        dict: A dictionary containing the requested scalar and vector diagnostics.
    """

    # diag_names = scalar_diag_names | vector_diag_names

    # name = diag_names[0]


    with h5py.File(source_filename, 'r') as source:
        dr, dz = source[f'data/{timestep}/fields/{diag_name}'].attrs['gridSpacing']

        try:
            Nr, Nz = source[f'data/{timestep}/fields/{diag_name}'][()].shape[1:]
        except TypeError:
            Nr, Nz = source[f'data/{timestep}/fields/{diag_name}/t'][()].shape[1:]

    r = dr * np.arange(Nr)[downsample_r]
    z = dz * np.arange(Nz)[downsample_z]

    return r, z