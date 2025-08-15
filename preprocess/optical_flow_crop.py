import nibabel as nib
import numpy as np
import cv2
import os

def calculate_optical_flow(frame1, frame2):
    """Calculate dense optical flow between two frames."""
    gray1 = cv2.normalize(frame1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gray2 = cv2.normalize(frame2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def extract_motion_center(flow):
    """Extract the centroid position of the motion region in the optical flow field."""
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    _, motion_mask = cv2.threshold(magnitude, np.mean(magnitude) + 2 * np.std(magnitude), 1, cv2.THRESH_BINARY)
    center = np.argwhere(motion_mask > 0)
    if len(center) > 0:
        return np.mean(center, axis=0).astype(int)  # Return center position (y, x)
    return None

def process_cine(data, crop_size):
    """
    Process NIfTI data, extract the middle depth slice, calculate myocardium position, and crop.
    Args:
        data: numpy.ndarray, shape (H, W, T, D)
        crop_size: int, half-size of the cropping box
    Returns:
        cropped_data: numpy.ndarray, cropped data
        final_position: tuple, final myocardium center position (y, x)
    """
    H, W, T, D = data.shape

    # Extract the middle depth slice (H, W, T)
    mid_slice = data[:, :, :, D // 2]

    # Calculate myocardium position for each frame
    positions = []
    for t in range(T - 1):
        frame1 = mid_slice[:, :, t]
        frame2 = mid_slice[:, :, t + 1]
        flow = calculate_optical_flow(frame1, frame2)
        center = extract_motion_center(flow)
        if center is not None:
            positions.append(center)

    try:
        # Determine final myocardium position
        if len(positions) > 0:
            final_position = np.mean(positions, axis=0).astype(int)  # Compute average position (y, x)
        # Set cropping box
        y, x = final_position
        y_start, y_end = max(0, y - crop_size), min(H, y + crop_size)
        x_start, x_end = max(0, x - crop_size), min(W, x + crop_size)

        # Crop each frame of the middle slice
        cropped_data = data[y_start:y_end, x_start:x_end, :, :]
        mark = 1
        return cropped_data, final_position, mark

    except Exception as e:
        print("No valid motion region detected.")
        mark = 0
        return data, (-1, -1), mark
    

def save_cropped_nifti(cropped_data, original_nifti_path, output_path):
    """Save cropped NIfTI data."""
    # Load the affine matrix and header from the original NIfTI
    original_nifti = nib.load(original_nifti_path)
    affine = original_nifti.affine

    cropped_nifti = nib.Nifti1Image(cropped_data, affine)
    nib.save(cropped_nifti, output_path)
    print(f"Cropped NIfTI data saved to: {output_path}")

