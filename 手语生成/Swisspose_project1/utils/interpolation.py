# utils/interpolation.py

import numpy as np

def interpolate_nans(pose_array):
    """
    Interpolates NaN values along the frame axis in the pose array.
    """
    num_frames, num_keypoints, num_coords = pose_array.shape
    for k in range(num_keypoints):
        for c in range(num_coords):
            coord_data = pose_array[:, k, c]
            nans = np.isnan(coord_data)
            if np.any(nans):
                not_nans = ~nans
                if np.sum(not_nans) == 0:
                    # If all values are NaN, set to zero
                    pose_array[:, k, c] = 0.0
                else:
                    interp = np.interp(
                        np.where(nans)[0],
                        np.where(not_nans)[0],
                        coord_data[not_nans]
                    )
                    pose_array[:, k, c][nans] = interp
    return pose_array
