# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:20:23 2024

@author: ADearling
"""

#%% Libraries

import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator

import main as pm


#%% Functions

def convert_to_polar(coords):
    '''Convert Cartesian (x, y) coordinates to polar (r, theta) for cylindrical symmetry'''
    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    theta = np.arctan2(coords[:, 1], coords[:, 0])  # Azimuthal angle
    z = coords[:, 2]  # Height
    return r, theta, z


def convert_to_spherical(coords):
    '''Convert Cartesian (x, y, z) coordinates to spherical (r, theta, phi) for spherical symmetry'''
    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
    theta = np.arctan2(coords[:, 1], coords[:, 0])  # Azimuthal angle
    phi = np.arccos(coords[:, 2] / r)  # Polar angle
    return r, theta, phi


def calc_grid_spacing(grid, grid_type="uniform"):
    '''Get the grid spacing'''
    
    # Find the distance to next point, excluding the first point on the grid
    distance = grid[1:] - grid[:-1]
    
    # Identify variation in the distance between adjacent points
    variation = distance[1:] - distance[:-1]
    
    # Check that variation is small, i.e. grid is regular, or exit
    if grid_type == "uniform":
        variation_max = np.amax(np.absolute(variation))
        if variation_max > 1e-12:
            print("Grid is not regular, max variation is {:.3e}".format(variation_max))
            sys.exit()
        spacing = distance[0]
            
    # If spacing is not expected to be uniform, find spacing
    elif grid_type == "log":
        spacing = distance
    else:
        spacing = distance
    
    # print(spacing)
    
    return spacing


def find_grid_centre(grid, grid_type):
    '''Obtains the cell centres (where variables are defined) from the cell boundary grid'''
    
    # Calculate the grid spacing
    grid_dx = calc_grid_spacing(grid, grid_type=grid_type)
    
    # Find the location of the centre of each cell
    grid_centre = grid[:-1] + grid_dx/2
    
    return grid_centre


def estimate_cell_widths(grid_centres):
    '''Estimate cell widths for an irregular grid'''
    
    widths = np.empty_like(grid_centres)
    
    # Internal cells: half distance to neighbors
    widths[1:-1] = 0.5 * (grid_centres[2:] - grid_centres[:-2])
    
    # Edge cells: extrapolate using nearest neighbor difference
    widths[0] = grid_centres[1] - grid_centres[0]
    widths[-1] = grid_centres[-1] - grid_centres[-2]
    
    return widths


def estimate_cell_boundaries(grid_centres):
    '''Estimate cell boundaries for an irregular grid'''
    
    widths = estimate_cell_widths(grid_centres)
    
    lefts = grid_centres - 0.5 * widths
    rights = grid_centres + 0.5 * widths
    
    return np.concatenate(([lefts[0]], 0.5 * (rights[:-1] + lefts[1:]), [rights[-1]]))


def modify_grid_x(data, x_old, x_new, background=None, plot=False):
    '''Change from one set of grid positions to another'''
        
    # Warn if new values lie outside old grid boundaries
    if (max(x_old) < max(x_new)) or (min(x_old) > min(x_new)):
        if background == None:
            print("Values lie outside old boundaries. Exiting.")
            sys.exit()
        else:
            print("WARNING: New grid points lie outside of old boundaries. Using bg = {}."
                  .format(background))
        
    # Currently works by interpolating in 1D
    if background == None:
        x_interp = interp1d(x_old, data)
    else:
        x_interp = interp1d(x_old, data, bounds_error=False, fill_value=background)
    data_regrid = x_interp(x_new)
    
    if plot == True:
        fig, ax = pm.Plot_Figure_Axis("small", 1)
        if data.ndim == 1:
            ax.plot(x_old, data)
            ax.plot(x_new, data_regrid)
        elif data.ndim == 2:
            ax.plot(x_old, data[-1])
            ax.plot(x_new, data_regrid[-1])
        elif data.ndim == 3:
            ax.plot(x_old, data[-1,-1])
            ax.plot(x_new, data_regrid[-1,-1])
    
    return data_regrid


def rodrigues_rotation_matrix(axis, theta):
    '''General rotation matrix for rotation around a given axis [vx,vy,vz].'''
    # Normalize the axis vector (in case it's not already a unit vector)
    axis = axis / np.linalg.norm(axis)
    
    # Extract components of the axis vector
    v_x, v_y, v_z = axis
    
    # Rodrigues' rotation matrix
    R = np.array([
        [np.cos(theta) + v_x**2 * (1 - np.cos(theta)), 
         v_x * v_y * (1 - np.cos(theta)) - v_z * np.sin(theta),
         v_x * v_z * (1 - np.cos(theta)) + v_y * np.sin(theta)],
        
        [v_y * v_x * (1 - np.cos(theta)) + v_z * np.sin(theta),
         np.cos(theta) + v_y**2 * (1 - np.cos(theta)),
         v_y * v_z * (1 - np.cos(theta)) - v_x * np.sin(theta)],
        
        [v_z * v_x * (1 - np.cos(theta)) - v_y * np.sin(theta),
         v_z * v_y * (1 - np.cos(theta)) + v_x * np.sin(theta),
         np.cos(theta) + v_z**2 * (1 - np.cos(theta))]
    ])
    
    return R


def rotate_vector_field(arr3d, axis=[0,0,1], theta=0):
    '''
    Rotate each point in a vector field by a given azimuthal (theta) and 
    polar angle (phi).
    
    axis: the rotation axis ([1,0,0] is rotation around x (y-z plane),
                             [0,1,0] is rotation around y (z-x plane),
                             [0,0,1] is rotation around z (x-y plane))
    '''
    
    # Rotation matrix for azimuthal rotation around the given axis
    R = rodrigues_rotation_matrix(axis, -theta)
    
    # Reshape the vector field array into a 2D array of shape (N, 3) where N is the number of points
    shape = arr3d.shape
    reshaped_arr = arr3d.reshape(-1, 3)  # Flattening the 3D array into a 2D array of vectors
    
    # Apply the rotation to all vectors at once
    rotated_vectors = reshaped_arr @ R.T  # Matrix multiplication with the rotation matrix
    
    # Reshape the rotated vectors back into the original 3D shape
    rotated_arr3d = rotated_vectors.reshape(shape)
    
    return rotated_arr3d


def rotate_cartesian_array(arr3d, x, y, z, axis=[0,0,1], theta=0):
    '''
    Rotate an array around the origin by a given azimuthal (theta) and 
    polar angle (phi). Polar rotation occurs before azimuthal rotation. 
    
    axis: the rotation axis ([1,0,0] is rotation around x (y-z plane),
                             [0,1,0] is rotation around y (z-x plane),
                             [0,0,1] is rotation around z (x-y plane))
    '''
    
    # Create meshgrid in 3D space
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten the grid coordinates (X, Y, Z) into a 2D array (each column is a point)
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    # Rotation matrix for azimuthal rotation around the given axis
    R = rodrigues_rotation_matrix(axis, theta)

    # Apply the rotation matrix to the coordinate points directly (matrix multiplication)
    rotated_coords = np.dot(coords, R.T)
    
    # Reshape rotated coordinates back into 3D grids
    x_rot, y_rot, z_rot = rotated_coords[:, 0].reshape(X.shape), rotated_coords[:, 1].reshape(Y.shape), rotated_coords[:, 2].reshape(Z.shape)
    
    # Create an interpolator for the original data based on the grid
    interpolator = RegularGridInterpolator((x, y, z), arr3d, bounds_error=False, fill_value=0)
    
    # Stack rotated coordinates into a grid for interpolation
    rotated_grid = np.stack([x_rot, y_rot, z_rot], axis=-1).reshape(-1, 3)
    
    # Use the interpolator to get the rotated data
    rotated_data = interpolator(rotated_grid)
    
    # Reshape the rotated data back into the original shape of arr3d
    return rotated_data.reshape(arr3d.shape)


def interpolate_edge_to_cell_centre(data, field):
    """
    Interpolates the electric fields to the cell centre in a 3D Yee grid.
    
    data: Field at cell edges (numpy arrays)
    field: Field unit vector
    
    Returns:
    data_c: Interpolated field at cell centres
    """
    
    # Assuming the input arrays have dimensions (nx, ny, nz) for 3D grid points
    nx, ny, nz = data.shape

    # Initialize arrays for the interpolated values at the cell centres
    data_c = np.zeros((nx - 1, ny - 1, nz - 1))

    # Interpolation for Electric Fields (on cell edges)
    for i in range(0, nx - 1):
        for j in range(0, ny - 1):
            for k in range(0, nz - 1):
                # Interpolate Ex at the centre of the cell (on cell edges in y-z)
                if field == "x":
                    data_c[i, j, k] = (data[i, j, k] + data[i, j+1, k] + 
                                       data[i, j, k+1] + data[i, j+1, k+1]) / 4
                
                # Interpolate Ey at the centre of the cell (on cell edges in x-z)
                elif field == "y":
                    data_c[i, j, k] = (data[i, j, k] + data[i, j, k+1] + 
                                       data[i+1, j, k] + data[i+1, j, k+1]) / 4
                
                # Interpolate Ez at the centre of the cell (on cell edges in x-y)
                elif field == "z":
                    data_c[i, j, k] = (data[i, j, k] + data[i+1, j, k] + 
                                       data[i, j+1, k] + data[i+1, j+1, k]) / 4

                else:
                    raise Exception("Incorrect field type ({}).".format(field))

    return data_c


def interpolate_face_to_cell_centre(Bx, By, Bz):
    """
    Interpolates the magnetic fields to the cell centre in a 3D Yee grid.
    
    Bx, By, Bz: Magnetic fields at cell faces (numpy arrays)
    
    Returns:
    Bx_c, By_c, Bz_c: Interpolated magnetic fields at cell centres
    """
    
    # Assuming the input arrays have dimensions (nx, ny, nz) for 3D grid points

    # Shape of the input arrays
    nx, ny, nz = Bx.shape
    
    # Initialize arrays for the interpolated values at the cell centres
    Bx_c = np.zeros((nx - 1, ny - 1, nz - 1))
    By_c = np.zeros((nx - 1, ny - 1, nz - 1))
    Bz_c = np.zeros((nx - 1, ny - 1, nz - 1))
    
    # Interpolation for Magnetic Fields (on cell faces)
    for i in range(0, nx - 1):
        for j in range(0, ny - 1):
            for k in range(0, nz - 1):
                # Interpolation for Bx at the centre of the cell (x direction)
                Bx_c[i, j, k] = (Bx[i, j, k] + Bx[i+1, j, k]) / 2
                
                # Interpolation for By at the centre of the cell (y direction)
                By_c[i, j, k] = (By[i, j, k] + By[i, j+1, k]) / 2
                
                # Interpolation for Bz at the centre of the cell (z direction)
                Bz_c[i, j, k] = (Bz[i, j, k] + Bz[i, j, k+1]) / 2

    return Bx_c, By_c, Bz_c


def project_array_3d(arr3d, magnitude=0, frac=0.025, fig=None, ax=None, title=None):
    '''
    Plot the position of data points on a 3D axis coloured according to their
    value.
    
    magnitude: the magnitude above which points are plotted.
    frac: the fraction of the available particles that are to be shown.
    '''
    
    if ax is None:
        # Create a figure and 3D axis
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    
    # Example 3D array (you can replace this with your own 3D data)
    # Here we create a simple 3D array of shape (x, y, z)
    x = np.arange(arr3d.shape[0])  # x-axis
    y = np.arange(arr3d.shape[1])  # y-axis
    z = np.arange(arr3d.shape[2])  # z-axis

    # Create a meshgrid for the coordinates
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Reshape values to be a 1D array for scatter plot or surface plot
    values_flat = arr3d.flatten()
    
    num_to_remove = int((1 - frac) * len(values_flat))
    ind_remove = np.random.choice(len(values_flat), size=num_to_remove, replace=False)
    values_flat[ind_remove] = 0

    valid_points = abs(values_flat) > magnitude

    # Scatter Plot (you can choose this or Surface Plot)
    # For a scatter plot, we will flatten the coordinates to 1D
    ax.scatter(X.flatten()[valid_points], Y.flatten()[valid_points], Z.flatten()[valid_points],
               c=values_flat[valid_points], cmap='seismic', marker='o')

    # Surface Plot (alternative to scatter plot)
    # ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(values), rstride=1, cstride=1, alpha=0.7)

    # Labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    
    # fig.tight_layout()
    
    return


#%% Test script
if __name__ == "__main__":
    
    if 0: # Test grid rotation
    
        # Example cylindrical data
        x = np.linspace(-1,1,140)
        y = np.linspace(-1,1,120)
        z = np.linspace(-1,1,80)
    
        cylinder = np.zeros((len(x),len(y),len(z)))
    
        for i in range(len(x)):
            for j in range(len(y)):    
                for k in range(len(z)):
                    if abs(z[k]) < 0.25:
                        if np.sqrt(x[i]**2+y[j]**2) < 0.25:
                            if x[i] < 0 or y[j] < 0:
                                cylinder[i,j,k] = 1
                    
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()
        cylinder_flat = cylinder.flatten()
    
        # Filter the points that have data (i.e., where cylindrical_data is non-zero)
        valid_points = cylinder_flat > 0

        # Plotting in 3D of original structure
        fig, ax = pm.plot_figure_axis(nplots=2)
        a0_position = ax[0].get_position()
        a1_position = ax[1].get_position()
        ax[0].remove()
        ax[1].remove()
        ax[0] = fig.add_subplot(a0_position, projection='3d')
        ax[1] = fig.add_subplot(a1_position, projection='3d')
    
        ax[0].scatter(X_flat[valid_points], Y_flat[valid_points], Z_flat[valid_points], c='r', s=1)
        ax[0].set_title('Original')
        
        # Set rotation axis and angle (in radians)
        axis = [0,0,1]
        theta = np.pi/2 # np.pi / 4  # 45 degrees
    
        # Rotate the array
        rotated_arr3d = rotate_cartesian_array(cylinder, x, y, z, axis, theta)
    
        rotated_cylinder_flat = rotated_arr3d.flatten()
    
        # Filter the points that have data (i.e., where cylindrical_data is non-zero)
        valid_points = rotated_cylinder_flat > 0

        # Plot rotated structure
        ax[1].scatter(X_flat[valid_points], Y_flat[valid_points], Z_flat[valid_points], c='r', s=1,)
        ax[1].set_title('Rotation')
        
        for axi in ax:
            axi.set_xlabel('x')
            axi.set_ylabel('y')
            axi.set_zlabel('z')
    
        fig.tight_layout()
    
    if 0: # Test vector rotation
    
        x = np.linspace(-1, 1, 3)
        y = np.linspace(-1, 1, 3)
        z = np.linspace(-1, 1, 3)
    
        # Create some sample data for the vector field
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        # Example 3D vector field (data in arr3d)
        arr3d = np.stack([X, Y, Z], axis=-1)  # Create a vector field (x, y, z) at each grid point
        
        # Set rotation axis and angle (in radians)
        axis = [0,0,1]
        theta = np.pi/2 # np.pi / 4  # 45 degrees
        
        # Rotate the vector field
        rotated_arr3d = rotate_vector_field(arr3d, axis, theta)
        
        # Check the original and rotated vector field (print a few values for inspection)
        print("Original vector field at (0, 0, 0):", arr3d[0, 0, 0])
        print("Rotated vector field at (0, 0, 0):", rotated_arr3d[0, 0, 0])
        
        # Visualizing the original and rotated vector fields (optional)
        fig = plt.figure(figsize=(12, 6))
        
        # Plot original vector field
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.quiver(X, Y, Z, arr3d[:,:,:,0], arr3d[:,:,:,1], arr3d[:,:,:,2])
        ax1.set_title("Original Vector Field")
        
        # Plot rotated vector field
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.quiver(X, Y, Z, rotated_arr3d[:,:,:,0], rotated_arr3d[:,:,:,1], rotated_arr3d[:,:,:,2])
        ax2.set_title("Rotated Vector Field")
        
        for axi in [ax1, ax2]:
            axi.set_xlabel('X')
            axi.set_ylabel('Y')
            axi.set_zlabel('Z')
        
        fig.tight_layout()
        
    if 1: # Test array and vector rotation
    
        x = np.linspace(-2, 2, 10)
        y = np.linspace(-2, 2, 10)
        z = np.linspace(-2, 1.5, 10)
    
        # Create some sample data for the vector field
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        # Example 3D vector field (data in arr3d)
        arr3d = np.stack([X, Y, Z], axis=-1)  # Create a vector field (x, y, z) at each grid point
        for i in range(len(x)):
            for j in range(len(y)):
                for k in range(len(z)):
                    if np.sqrt(x[i]**2 + y[j]**2) > 1:
                        arr3d[i,j,k] = 0
                    elif x[i] >= 0 and y[j] >= 0:
                        arr3d[i,j,k] = 0
        
        # Set rotation axis and angle (in radians)
        axis = [0,1,0]
        theta = np.pi/2 # np.pi / 4  # 45 degrees
        
        # Rotate the vector field
        rotated_arr3d = rotate_vector_field(arr3d, axis, theta)
        
        # Rotate the array
        rotated_arr3d = rotate_cartesian_array(rotated_arr3d, x, y, z, axis, theta)
        
        # Visualizing the original and rotated vector fields (optional)
        fig = plt.figure(figsize=(12, 6))
        
        # Plot original vector field
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.quiver(X, Y, Z, arr3d[:,:,:,0], arr3d[:,:,:,1], arr3d[:,:,:,2])
        ax1.set_title("Original Vector Field")
        
        # Plot rotated vector field
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.quiver(X, Y, Z, rotated_arr3d[:,:,:,0], rotated_arr3d[:,:,:,1], rotated_arr3d[:,:,:,2])
        ax2.set_title("Rotated Vector Field")
        
        for axi in [ax1, ax2]:
            axi.set_xlabel('X')
            axi.set_ylabel('Y')
            axi.set_zlabel('Z')
        
        fig.tight_layout()
        
        plt.show()
    
    if 0: # Test Yee interpolation
    
        # Example usage with 3D field arrays (nx x ny x nz)
        nx, ny, nz = 2, 2, 2  # Grid size
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.arange(nz)
    
        xc = np.arange(nx-1) + 0.5
        yc = np.arange(ny-1) + 0.5
        zc = np.arange(nz-1) + 0.5
    
        # Initialize sample field data
        Ex = np.random.rand(nx, ny, nz)
        Ey = np.random.rand(nx, ny, nz)
        Ez = np.random.rand(nx, ny, nz)
    
        # Interpolate to cell centres
        Ex_c, Ey_c, Ez_c = interpolate_edge_to_cell_centre(Ex, Ey, Ez)
    
        # Print the shapes of the interpolated fields
        print("Interpolated Electric Fields:")
        print("Ex_c shape:", Ex_c.shape)
        print("Ey_c shape:", Ey_c.shape)
        print("Ez_c shape:", Ez_c.shape)
    
        # Generate the grid of coordinates
        xx, yy, zz = np.meshgrid(xc, y, z)
    
        # Flatten the 3D arrays to 1D arrays for plotting
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        z_flat = zz.flatten()
        scalar_flat = Ex[0,:,:].flatten()  # Corresponding scalar values
    
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
        # Scatter plot with coloring based on scalar values
        scatter = ax.scatter(x_flat, y_flat, z_flat, c=scalar_flat, cmap='viridis')
        scatter = ax.scatter(xc, yc, zc, c=Ex_c, cmap='viridis')
    
        # Add a color bar to indicate the scalar values
        fig.colorbar(scatter, ax=ax, label='Scalar Value')
    
        # Label axes
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
