# fbpic_to_vtk
Export FBPIC simulation data onto cylindrical meshes for visualization in Paraview

## Installation

Clone repository
```bash
git clone git@github.com:hixps/fbpic_to_vtk.git
```

Install
```bash
pip install fbpic_to_vtk
```


## Useage

* Define input and output filenames, simulation timestep
    ```python
    timestep        = 40000
    source_filename = f'fbpic_simulation_output.h5'
    output_filename = 'fbpic_data_on_vtk_grid.vtu'
    ```
* Define the names of the diagnostics to be exported, they need to be the same as the diagnostics in the fbpic input deck
    ```python
    scalar_diag_names = ('rho_electron', 'rho_inner', 'rho_ion')
    vector_diag_names = ('E',)
    ```

* Settings for the vtk mesh
    * Define numer of points along theta ```Ntheta = 32```
    * Define slices along z and r directions for downsampling
    ```python
    downsample_r = slice(None, -2, 2)
    downsample_z = slice(None, None, 5)
    ```
* Run the exporter
    ```python
    from fbpic_to_vtk import fbpicToVTK
    
    fbpicToVTK( source_filename,
            timestep,
            scalar_diag_names,
            vector_diag_names,
            downsample_r = downsample_r,
            downsample_z = downsample_z,
            Ntheta = Ntheta,
            output_filename = output_filename
            )
    ```

## Example

<img src="./imgs/wakefield_render.png" width=600>


