import numpy as np
import json
import bisect
import os


def get_camera_parameters():

    """ Get camera calibration parameters from a JSON file.

    Returns:
        tuple: A tuple containing the camera matrix (mtx), distortion coefficients (dist),
        rotation vectors (rvecs), and translation vectors (tvecs).
        Returns (None, None, None, None) if calibration file not found.
    """
    
    calib_file = 'calibration/camera_calibration.json'
    
    if not os.path.exists(calib_file):
        print(f"Warning: Calibration file '{calib_file}' not found. Using default camera parameters.")
        return None, None, None, None

    try:
        with open(calib_file, 'r') as f:
            calibration_data = json.load(f)

        mtx = np.array(calibration_data['camera_matrix'])
        dist = np.array(calibration_data['distortion_coefficients'])
        rvecs = np.array(calibration_data['rotation_vectors'])
        tvecs = np.array(calibration_data['translation_vectors'])

        return mtx, dist, rvecs, tvecs
    except Exception as e:
        print(f"Error loading calibration file: {e}")
        return None, None, None, None

def insert_sorted(arr, value):

    """ Insert a value into a sorted list in a way that keeps the list sorted.

    Args:
        arr (list): The sorted list.
        value (float): The value to be inserted.

    Returns:
        list: The sorted list with the new value inserted.
    """

    bisect.insort(arr, value)

    return arr

def calculate_median(arr):

    """ Calculate the median value of a list.

    Args:
        arr (list): The input list of values.

    Returns:
        float: The median value of the list.
    """

    n = len(arr)
    mid = n // 2

    if n % 2 == 0:
        return (arr[mid - 1] + arr[mid]) / 2
    else:
        return arr[mid]