import numpy as np
import xarray as xa
from scipy import ndimage

# Load the dataset containing drought clusters
dataset_path = 'data/optimalPET/tree-dimension/MeanoptSPEIcluster.nc'
dataset = xa.open_dataset(dataset_path)
drought_cluster = np.array(dataset.variables['drought_cluster'][:,:,:])

# Define a structuring element for connected component analysis
structure = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])

# Minimum overlap threshold for considering temporal continuity
clust_threshold = 1150  # Equivalent to 46 grid cells

def component(c):
    """
    Identifies connected components (drought patches) within a given 2D array.
    """
    labeled_matrix, num_features = ndimage.label(c > 0, structure=structure)
    return [labeled_matrix == i for i in range(1, num_features + 1)]

def track_drought_patch(initial_patch, time):
    """
    Tracks a drought patch through consecutive months if the overlap meets the threshold.
    """
    t = 1  # Time step counter
    patch_series = [initial_patch]  # Store connected patches over time
    current_patch = initial_patch
    best_patch = None
    
    while time + t < drought_cluster.shape[0]:  # Ensure within time range
        next_patches = component(drought_cluster[time + t, :, :])  # Extract patches for the next time step
        
        for next_patch in next_patches:
            overlap = np.sum(current_patch * next_patch.astype(int))
            
            if overlap > clust_threshold:
                if best_patch is None or np.sum(next_patch) > np.sum(best_patch):
                    best_patch = next_patch.astype(int)
        
        if best_patch is not None:
            patch_series.append(best_patch)
            current_patch = best_patch
            t += 1
            best_patch = None  # Reset for the next iteration
        else:
            break  # Stop if no valid connection is found
    
    if t >= 3:  # Consider only drought events lasting more than 3 months
        return t, patch_series
    else:
        return 0, []

# Initialize an array to store labeled drought events
drought_patches_labeled = np.zeros_like(drought_cluster, dtype=int)
z = 1  # Drought event label counter

# Iterate over each time step
for time in range(drought_cluster.shape[0] - 2):  # Exclude last two months
    patches = component(drought_cluster[time, :, :])  # Extract drought patches
    
    for patch in patches:
        patch = patch.astype(int)
        duration, tracked_patches = track_drought_patch(patch, time)
        
        if duration >= 3:
            mm = len(tracked_patches)
            drought_patches_labeled[time:time+mm, :, :][np.array(tracked_patches) == True] = z
            z += 1  # Increment label for next event
            drought_cluster[time:time+mm, :, :][np.array(tracked_patches) == True] = 0  # Mark as processed

# Create a new xarray dataset for labeled drought events
labeled_dataset = xa.Dataset()
labeled_dataset['drought_label'] = xa.DataArray(
    drought_patches_labeled,
    coords={
        'time': dataset.time,
        'latitude': dataset.latitude,
        'longitude': dataset.longitude,
    },
)

# Save the labeled drought event data
output_path = 'data/optimalPET/tree-dimension/MeanoptSPEIlabel.nc'
labeled_dataset.to_netcdf(output_path)
