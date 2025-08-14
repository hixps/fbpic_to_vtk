import numpy as np
import meshio

def export_to_VTK(
    theta_coords: np.ndarray, 
    r_coords: np.ndarray,
    z_coords: np.ndarray, 
    scalar_diags: dict, 
    vector_diags: dict,
    filename: str = "fbpic_cylinder_data.vtu"
    ):
    """
    Exports a solid cylinder VTK file from 3D data arrays with flexible scalar
    and vector data input, using a fast, vectorized approach.

    Args:
        theta_coords (np.ndarray): 1D array of theta coordinates (shape: (ntheta,)).
        r_coords (np.ndarray): 1D array of radial coordinates (shape: (nr,)).
        z_coords (np.ndarray): 1D array of z coordinates (shape: (nz,)).
        scalar_diags (dict): A dictionary where keys are string names for the
                             scalar fields and values are 3D numpy arrays
                             (shape: (ntheta, nr, nz)).
        vector_diags (dict): A dictionary where keys are string names for the
                             vector fields and values are tuples of three 3D numpy
                             arrays (vx, vy, vz) with shape (ntheta, nr, nz).
        filename (str): The name of the output VTK file.
    """
    nr = len(r_coords)
    ntheta = len(theta_coords)
    nz = len(z_coords)

    # --- Vectorized Point Generation ---
    # The central axis points (r=0) are handled separately
    z_axis_points = np.column_stack((
        np.zeros(nz), 
        np.zeros(nz), 
        z_coords
    ))

    # The outer points (r>0) are generated using vectorized operations
    theta_grid, r_grid, z_grid = np.meshgrid(
        theta_coords, r_coords[1:], z_coords, indexing='ij'
    )
    
    x_outer = r_grid * np.cos(theta_grid)
    y_outer = r_grid * np.sin(theta_grid)
    z_outer = z_grid

    outer_points = np.column_stack((
        x_outer.flatten(),
        y_outer.flatten(),
        z_outer.flatten()
    ))
    
    points = np.concatenate((z_axis_points, outer_points))

    # --- Vectorized Data Processing ---
    # Process each scalar field in the dictionary
    processed_scalar_diags = {}
    for name, data_3d in scalar_diags.items():
        # Average data for the central axis (r=0)
        scalar_r0 = np.mean(data_3d[:, 0, :], axis=0)
        
        # Flatten the outer data directly, ensuring correct order
        outer_scalar_data = data_3d[:, 1:, :].flatten()
        
        # Combine central and outer data
        processed_scalar_diags[name] = np.concatenate((scalar_r0, outer_scalar_data)).astype('float32')

    # Process each vector field in the dictionary
    processed_vector_diags = {}
    for name, (vx_3d, vy_3d, vz_3d) in vector_diags.items():
        # Average data for the central axis (r=0)
        vx_r0 = np.mean(vx_3d[:, 0, :], axis=0)
        vy_r0 = np.mean(vy_3d[:, 0, :], axis=0)
        vz_r0 = np.mean(vz_3d[:, 0, :], axis=0)
        
        # Flatten the outer data directly, ensuring correct order
        outer_vx_data = vx_3d[:, 1:, :].flatten()
        outer_vy_data = vy_3d[:, 1:, :].flatten()
        outer_vz_data = vz_3d[:, 1:, :].flatten()
        
        # Combine central and outer data
        vx_final = np.concatenate((vx_r0, outer_vx_data)).astype('float32')
        vy_final = np.concatenate((vy_r0, outer_vy_data)).astype('float32')
        vz_final = np.concatenate((vz_r0, outer_vz_data)).astype('float32')
        
        # Combine the three components into a single vector array
        processed_vector_diags[name] = np.column_stack((vx_final, vy_final, vz_final))

    # --- Vectorized Cell Connectivity ---
    nz_cells = nz - 1
    nr_outer_cells = nr - 1
    ntheta_cells = ntheta
    
    outer_point_index_offset = nz
    slice_factor = nr_outer_cells * nz

    # 1. Wedge connectivity (at the center, r=0 to r=r_coords[1])
    j_grid, k_grid = np.meshgrid(
        np.arange(ntheta_cells), np.arange(nz_cells), indexing='ij'
    )
    
    p_center_bottom = k_grid.flatten()
    p_outer1_bottom = outer_point_index_offset + j_grid.flatten() * slice_factor + k_grid.flatten()
    p_outer2_bottom = outer_point_index_offset + ((j_grid.flatten() + 1) % ntheta_cells) * slice_factor + k_grid.flatten()

    p_center_top = p_center_bottom + 1
    p_outer1_top = p_outer1_bottom + 1
    p_outer2_top = p_outer2_bottom + 1

    wedge_connectivity = np.column_stack((
        p_center_bottom, p_outer1_bottom, p_outer2_bottom,
        p_center_top, p_outer1_top, p_outer2_top
    ))

    # 2. Hexahedron connectivity (all other cells, r > r_coords[1])
    j_grid, i_grid, k_grid = np.meshgrid(
        np.arange(ntheta_cells), np.arange(1, nr_outer_cells), np.arange(nz_cells), indexing='ij'
    )
    
    p1_idx = outer_point_index_offset + j_grid * slice_factor + (i_grid - 1) * nz + k_grid
    p2_idx = outer_point_index_offset + j_grid * slice_factor + i_grid * nz + k_grid
    
    j_plus_one_grid = (j_grid + 1) % ntheta_cells
    p4_idx = outer_point_index_offset + j_plus_one_grid * slice_factor + (i_grid - 1) * nz + k_grid
    p3_idx = outer_point_index_offset + j_plus_one_grid * slice_factor + i_grid * nz + k_grid

    p5_idx = p1_idx + 1
    p6_idx = p2_idx + 1
    p7_idx = p3_idx + 1
    p8_idx = p4_idx + 1

    hexahedron_connectivity = np.column_stack((
        p1_idx.flatten(), p2_idx.flatten(), p3_idx.flatten(), p4_idx.flatten(),
        p5_idx.flatten(), p6_idx.flatten(), p7_idx.flatten(), p8_idx.flatten()
    ))
    
    cells = [
        ("wedge", wedge_connectivity),
        ("hexahedron", hexahedron_connectivity)
    ]
    
    # Create the point_data dictionary for meshio
    point_data = {}
    point_data.update(processed_scalar_diags)
    point_data.update(processed_vector_diags)

    meshio.write_points_cells(
        filename,
        points,
        cells,
        point_data=point_data
    )

