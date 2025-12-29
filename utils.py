import numpy as np
import bisect


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