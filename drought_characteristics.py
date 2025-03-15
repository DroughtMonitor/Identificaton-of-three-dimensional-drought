"""
This library provides formulas for Step 2
"""
__author__ = "Weiqi Liu"
__copyright__ = "Copyright (C) 2024 Weiqi Liu"
__license__ = "NIEER"
__version__ = "2025.03"
__Reference paper__ == "The Evaluation of the Suitability of Potential Evapotranspiration Models for Drought Monitoring based on Observed Pan Evaporation and Potential Evapotranspiration from Eddy Covariance"


import xarray as xa
import numpy as np
import csv

# Load SPEI dataset
spei_ds = xa.open_dataset('data/optimalPET/MeanoptSPEI03.nc')
spei = np.array(spei_ds.variables['spei'][:,:,:])

# Load drought label dataset
label_ds = xa.open_dataset('data/optimalPET/tree-dimension/MeanoptSPEIlabel222.nc')
label = np.array(label_ds.sel(time=slice("1950-01-01", "2022-12-31")).drought_label)

# Reorganize drought event labels
NewLabel = np.zeros_like(label, dtype=int)
MonthScale = 3  # Minimum duration of a drought event (in months)

t = 1  # Drought event counter
for i in range(1, int(np.max(label)) + 1):
    event_mask = np.where(label == i, 1, 0)
    active_months = np.max(event_mask, axis=(1, 2))
    event_indices = np.where(active_months == 1)[0]
    
    if len(event_indices) > 0:
        duration = event_indices[-1] - event_indices[0] + 1
        if duration >= MonthScale:
            NewLabel[label == i] = t
            t += 1

event_month = []  # Store the duration of each event (in months)
for i in range(np.max(NewLabel)):
    event_mask = np.where(NewLabel == i + 1, 1, 0)
    active_months = np.max(event_mask, axis=(1, 2))
    
    first_idx = np.argmax(active_months == 1)
    last_idx = len(active_months) - 1 - np.argmax(active_months[::-1] == 1)
    
    event_month.append(last_idx - first_idx + 1)

# Get the time range for each event
Time = label_ds.sel(time=slice("1950-01-01", "2022-12-31")).time

def TimeRange(labelorder, NewLabel):
    event_mask = np.where(NewLabel == labelorder, 1, np.nan)
    
    for j in range(Time.shape[0]):
        if np.nanmax(event_mask[j, :, :]) == 1:
            onset = Time[j]
            end = Time[j + event_month[labelorder - 1] - 1]
            return np.array(onset), np.array(end), j, j + event_month[labelorder - 1] - 1

# Calculate drought-affected area
lat = np.array(spei_ds.variables['latitude'])
lon = np.array(spei_ds.variables['longitude'])
one = np.ones((lat.shape[0], lon.shape[0]))
D_lat = lat[:, np.newaxis] * one
D_lon = lon * one

Re = 6371  # Earth's radius (km)
resolution = 0.1  # Data resolution
area = (2 * np.pi / 360) * (Re ** 2) * resolution * (np.sin(np.radians(D_lat + resolution)) - np.sin(np.radians(D_lat)))

# Compute drought severity, intensity, and affected area
droughtThreshod = -1
severity, intensity, Area = [], [], []
for i in range(1, int(np.nanmax(NewLabel)) + 1):
    event_mask = np.where(NewLabel == i, 1, np.nan)
    affected_cells = np.nansum(event_mask, axis=0)
    DA = np.nansum(np.where(affected_cells > 0, 1, 0) * area)
    
    Area.append(DA)
    severity.append(np.nansum((spei - droughtThreshod) * event_mask * area))
    intensity.append(severity[-1] / np.nansum(event_mask * area))

# Compute centroid coordinates (latitude, longitude, time) for each drought event
Cx, Cy, Ct = [], [], []
for i in range(1, int(np.nanmax(NewLabel)) + 1):
    event_mask = np.where(NewLabel == i, 1, 0)
    
    Cx.append(np.nansum((spei - droughtThreshod) * event_mask * area * D_lat) / np.nansum((spei - droughtThreshod) * event_mask * area))
    Cy.append(np.nansum((spei - droughtThreshod) * event_mask * area * D_lon) / np.nansum((spei - droughtThreshod) * event_mask * area))
    
    event_mask_time = event_mask.copy()
    for j in range(event_mask.shape[0]):
        if np.max(event_mask[j, :, :]) == 1:
            break
    
    for k in range(1, event_month[i - 1] + 1):
        event_mask_time[k + j - 1, :, :] = np.where(event_mask_time[k + j - 1, :, :] > 0, k, 0)
    
    Ct.append(np.nansum((spei - droughtThreshod) * event_mask_time * area) / np.nansum((spei - droughtThreshod) * event_mask * area))

# Save data to CSV file
with open('data/optimalPET/tree-dimension/MeanoptSPEI.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['severity', 'Cy', 'Cx', 'Area', 'intensity', 'Duration'])
    for i in range(len(severity)):
        writer.writerow([severity[i], Cy[i], Cx[i], Area[i], intensity[i], event_month[i]])
