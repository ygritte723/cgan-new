# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
import nibabel as nib
import h5py


# Define data directory and file
data_dir = '/home/xzhon54/xinliuz/nips_data'
# data_file = data_dir + '/sub-CC00856XX15_ses-3530_desc-restore_T2w.nii.gz'
# data_file = data_dir + '/STA21.nii.gz'
# data_file = data_dir + '/M-CRIB_P01_T1.nii'
data_file = data_dir + '/1101_3.nii'

# Load the NIfTI file
nib.Nifti1Header.quaternion_threshold = -1e-06

nii = nib.load(data_file)
nii.header['qform_code'] = 0

data = nii.get_fdata()   # For Python 2.7 with older nibabel, you might need to use get_data()
print("Original image shape:", data.shape)  # e.g., (348, 384, 128)

# Select three neighboring slices from the z-dimension.
# For example, choose the middle slice and its immediate neighbors.
z_center = data.shape[2] // 2   # Middle slice index
slice_indices = [z_center - 1, z_center, z_center + 1]
data_slices = np.stack([data[:, :, idx] for idx in slice_indices], axis=0)  # Shape: (3, x, y)
print("Stacked slices shape:", data_slices.shape)

# Normalize the voxel intensities to the range [0, 1]
data_slices = (data_slices - data_slices.min()) / (data_slices.max() - data_slices.min())

# Expand dims to add a sample dimension:
# Current shape: (3, x, y) --> Desired intermediate shape: (3, 1, x, y)
data_x = data_slices[:, np.newaxis, :, :]
# Transpose dimensions to match expected (x, y, samples, neighboring slices)
data_x = np.transpose(data_x, (2, 3, 1, 0))  # Final shape: (x, y, 1, 3)
print("Processed data_x shape:", data_x.shape)

# For testing purposes, create a dummy target contrast by duplicating data_x.
data_y = data_x.copy()

# For testing, create a dummy target contrast by copying data_x.
data_y = data_x.copy()

# Save as an HDF5-based .mat file using h5py so that h5py.File can open it.
output_file = 'datasets/nips/test/data.mat'
with h5py.File(output_file, 'w') as f:
    f.create_dataset('data_x', data=data_x)
    f.create_dataset('data_y', data=data_y)

print("Saved HDF5-based .mat file at", output_file)