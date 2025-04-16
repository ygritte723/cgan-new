# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
import h5py

# ------------------------------
# Define Data Directories & Files
# ------------------------------

# Directories for T1 and T2 images.
data_dir_t1 = '/home/xzhon54/wangdata/data/M_CRIB/whole_brain/preprocessed_T1w'
data_dir_t2 = '/home/xzhon54/wangdata/data/M_CRIB/whole_brain/preprocessed_T2w'

# Get sorted lists of NIfTI filenames for each modality.
all_files_t1 = sorted([f for f in os.listdir(data_dir_t1) if f.endswith('.nii')])
all_files_t2 = sorted([f for f in os.listdir(data_dir_t2) if f.endswith('.nii')])

# Define splits:
# - Training: first 7 files.
# - Validation: files from index 7 to second-to-last.
# - Test: here using a fixed test subject.
training_files_t1 = [os.path.join(data_dir_t1, f) for f in all_files_t1[:7]]
training_files_t2 = [os.path.join(data_dir_t2, f) for f in all_files_t2[:7]]

val_files_t1 = [os.path.join(data_dir_t1, f) for f in all_files_t1[7:-1]]
val_files_t2 = [os.path.join(data_dir_t2, f) for f in all_files_t2[7:-1]]

test_file_t1 = [os.path.join(data_dir_t1, 'M-CRIB_P10_T1.nii')]
test_file_t2 = [os.path.join(data_dir_t2, 'M-CRIB_P10_T2.nii')]

# Pair the corresponding T1 and T2 files.
training_pairs = list(zip(training_files_t1, training_files_t2))
val_pairs      = list(zip(val_files_t1, val_files_t2))
test_pairs     = list(zip(test_file_t1, test_file_t2))

# Print paired file lists.
print("Training pairs:")
for t1, t2 in training_pairs:
    print("  T1:", t1)
    print("  T2:", t2)
    print("---")

print("Validation pairs:")
for t1, t2 in val_pairs:
    print("  T1:", t1)
    print("  T2:", t2)
    print("---")

print("Test pairs:")
for t1, t2 in test_pairs:
    print("  T1:", t1)
    print("  T2:", t2)

# ------------------------------
# Define Processing Function
# ------------------------------

def process_file(data_file):
    """
    Loads a NIfTI file, extracts three neighboring axial slices from the center,
    normalizes them to [0, 1], adds a channel dimension, and rearranges the axes
    to yield an array of shape (x, y, 1, 3).
    """
    print("Processing file:", data_file)
    nib.Nifti1Header.quaternion_threshold = -1e-06
    nii = nib.load(data_file)
    # Optionally disable qform to bypass header issues.
    nii.header['qform_code'] = 0
    data = nii.get_fdata()
    print("  Original image shape:", data.shape)  # e.g., (348, 384, 128)

    # Select three neighboring slices from the center along the 3rd dimension.
    z_center = data.shape[2] // 2
    slice_indices = [z_center - 1, z_center, z_center + 1]
    # Stack selected slices: result is of shape (3, x, y)
    data_slices = np.stack([data[:, :, idx] for idx in slice_indices], axis=0)
    print("  Stacked slices shape:", data_slices.shape)

    # Normalize slices to range [0, 1].
    data_slices = (data_slices - data_slices.min()) / (data_slices.max() - data_slices.min())
    
    # Add channel dimension: (3, x, y) -> (3, 1, x, y)
    data_slices = data_slices[:, np.newaxis, :, :]
    
    # Rearrange dimensions to (x, y, 1, 3)
    data_processed = np.transpose(data_slices, (2, 3, 1, 0))
    print("  Processed data shape for this file:", data_processed.shape)
    return data_processed

# ------------------------------
# Process and Save the Data
# ------------------------------

# A helper function to process a list of paired files and save to an HDF5-based MAT file.
def process_and_save(pairs, output_file):
    all_data_x = []  # T1 (source) images
    all_data_y = []  # T2 (target) images
    
    for t1_file, t2_file in pairs:
        # Process T1 and T2 files.
        processed_t1 = process_file(t1_file)
        processed_t2 = process_file(t2_file)
        all_data_x.append(processed_t1)
        all_data_y.append(processed_t2)
    
    # Concatenate along the sample axis (axis 2).
    combined_data_x = np.concatenate(all_data_x, axis=2)
    combined_data_y = np.concatenate(all_data_y, axis=2)
    print("Combined T1 (source) data shape:", combined_data_x.shape)
    print("Combined T2 (target) data shape:", combined_data_y.shape)
    
    # Create the directory if needed and save using h5py.
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data_x', data=combined_data_x)
        f.create_dataset('data_y', data=combined_data_y)
    print("Saved HDF5-based file at", output_file)

# Process Test Data
print("\nProcessing Test Data:")
process_and_save(test_pairs, 'datasets/infant/test/data.mat')

# Process Training Data
print("\nProcessing Training Data:")
process_and_save(training_pairs, 'datasets/infant/train/data.mat')

# Process Validation Data
print("\nProcessing Validation Data:")
process_and_save(val_pairs, 'datasets/infant/val/data.mat')