import numpy as np

from . import importer, exporter, modesum


def fbpicToVTK( source_filename: str,
                timestep: int,
                scalar_diag_names: tuple[str],
                vector_diag_names: tuple[str],
                downsample_r: slice = slice(None),
                downsample_z: slice = slice(None),
                Ntheta: int = 32,
                output_filename: str="fbpic_data.vtu"
            ) -> None:
    """
    Convert FBPic data to VTK format.
    
    Parameters:
        source_filename (str): Path to the FBPic HDF5 file.
        timestep (int): Timestep to read from the file.
        scalar_diag_names (list of str): Names of scalar diagnostics.
        vector_diag_names (list of str): Names of vector diagnostics.
        downsample_r (int, optional): Downsampling factor for radial coordinates.
        downsample_z (int, optional): Downsampling factor for axial coordinates.
    
    Returns:
        r, z: Radial and axial coordinates.
    """ 

    # create the azimuthal coordinate grid
    theta = np.linspace(0        , 2*np.pi,   Ntheta, endpoint=False )    


    diag_name = (scalar_diag_names + vector_diag_names)[0]

    # Read radial and axial coordinates
    r, z = importer.read_rz_coordinates(
        source_filename,
        timestep,
        diag_name,
        downsample_r=downsample_r,
        downsample_z=downsample_z
        )

    # Read scalar and vector diagnostics from the HDF5 file
    scalar_diags_modes, vector_diags_modes = importer.read_hdf5_file(
        source_filename,
        timestep,
        scalar_diag_names,
        vector_diag_names,
        downsample_3d=(slice(None),downsample_r, downsample_z)
    )


    # reconstruct the scalar field data on the cylindrical grid
    scalar_diags_reconstructed = dict( (name, modesum.scalar_field_reconstruction(field, theta)) for name, field in scalar_diags_modes.items() )

    # reconstruct the vector field data on the cylindrical grid and convert to Cartesian coordinates (vx, vy, vz)
    vector_diags_reconstructed = dict( (name, modesum.vector_field_reconstruction(field, theta)) for name, field in vector_diags_modes.items() )

    # hardcode the rescaling of the coordinates to micrometers
    theta_coords = theta
    r_coords = 1e6 * r
    z_coords = 1e6 * z

    exporter.export_to_VTK(
        theta_coords = theta_coords, 
        r_coords = r_coords, 
        z_coords = z_coords, 
        scalar_diags = scalar_diags_reconstructed, 
        vector_diags = vector_diags_reconstructed,
        filename = output_filename
        )
    

    return None


