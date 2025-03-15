"""
This library provides formulas for Step 1
"""
__author__ = "Weiqi Liu"
__copyright__ = "Copyright (C) 2024 Weiqi Liu"
__license__ = "NIEER"
__version__ = "2025.03"
__Reference paper__ == "The Evaluation of the Suitability of Potential Evapotranspiration Models for Drought Monitoring based on Observed Pan Evaporation and Potential Evapotranspiration from Eddy Covariance"

import numpy as np
import xarray as xa
from scipy import ndimage

# Load NetCDF file and extract SPEI data
data_path = 'data/optimalPET/MeanoptSPEI03.nc'
dataset = xa.open_dataset(data_path)
spei_data = dataset.variables['spei'][:, :, :]

# Set drought thresholds and minimum drought patch size (in grid cells)
SPEI_THRESHOLD = -1  # SPEI drought threshold
MIN_DROUGHT_PATCH_SIZE = 1150  # Minimum drought patch size (number of grid cells)

def identify_connected_patches(binary_drought_map):
    """
    Identify and filter drought patches.
    Uses a 3×3 structuring element to connect grid cells and removes patches smaller than MIN_DROUGHT_PATCH_SIZE.
    
    Parameters:
    binary_drought_map (numpy.ndarray): Binary drought map (1 for drought, 0 for non-drought).
    
    Returns:
    filtered_labeled_matrix (numpy.ndarray): Labeled matrix of filtered drought patches.
    large_components (list): List of binary matrices for each identified drought patch.
    num_valid_patches (int): Number of valid drought patches after filtering.
    """
    # Define a 3×3 structuring element to connect adjacent drought cells
    structure = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
    
    # Label connected components
    labeled_matrix, num_features = ndimage.label(binary_drought_map, structure=structure)
    
    # Count the number of grid cells in each component
    component_sizes = ndimage.sum(binary_drought_map, labeled_matrix, range(num_features + 1))
    
    # Filter out patches smaller than MIN_DROUGHT_PATCH_SIZE
    valid_labels = np.where(component_sizes >= MIN_DROUGHT_PATCH_SIZE)[0]
    filtered_matrix = np.where(np.isin(labeled_matrix, valid_labels), labeled_matrix, 0)
    
    # Relabel filtered drought patches
    filtered_labeled_matrix, num_filtered_features = ndimage.label(filtered_matrix > 0, structure=structure)
    
    # Count final drought patches
    component_sizes = ndimage.sum(filtered_matrix > 0, filtered_labeled_matrix, range(num_filtered_features + 1))
    large_components = [filtered_labeled_matrix == i for i in range(1, num_filtered_features + 1)]
    
    #print(f"Number of valid drought patches: {num_filtered_features}")
    
    return filtered_labeled_matrix, large_components, num_filtered_features

def extract_drought_patches(spei_map):
    """
    Extract drought regions and apply 3×3 neighborhood filtering.
    
    Parameters:
    spei_map (numpy.ndarray): SPEI data for the current time step.
    
    Returns:
    numpy.ndarray: Binary drought map after 3×3 filtering.
    """
    rows, cols = spei_map.shape
    drought_patches = np.zeros_like(spei_map, dtype=int)
    
    # Identify grid cells where SPEI is below the threshold
    drought_cells = spei_map < SPEI_THRESHOLD
    drought_patches[drought_cells] = 1
    
    # Apply 3×3 neighborhood filtering
    padded_drought_patches = np.pad(drought_patches, ((1, 1), (1, 1)), mode='constant')
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if padded_drought_patches[i, j] == 1:
                # Compute the mean of the 3×3 neighborhood
                adjacent_cells = padded_drought_patches[i - 1:i + 2, j - 1:j + 2]
                if np.nanmean(adjacent_cells) > 1 / 9:  # Retain if at least two drought cells are present
                    drought_patches[i - 1, j - 1] = 1
                else:
                    drought_patches[i - 1, j - 1] = 0
    
    return drought_patches

# Process all time steps in the SPEI dataset
drought_cluster = np.zeros_like(spei_data, dtype=int)
for t in range(spei_data.shape[0]):
    drought_patches = extract_drought_patches(spei_data[t, :, :])
    drought_cluster[t, :, :] = identify_connected_patches(drought_patches)[0]

# Save the processed drought patch data
drought_cluster_dataset = xa.Dataset(
    {
        'drought_cluster': xa.DataArray(drought_cluster,
                                         coords={'time': dataset.time,
                                                 'latitude': dataset.latitude,
                                                 'longitude': dataset.longitude},
                                         dims=['time', 'latitude', 'longitude'])
    }
)

output_path = 'data/optimalPET/tree-dimension/MeanoptSPEIcluster222.nc'
drought_cluster_dataset.to_netcdf(output_path)

print(f"Processed drought clusters saved to: {output_path}")
