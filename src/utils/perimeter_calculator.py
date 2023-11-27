import numpy as np


def stad_p(width, depth):
    """
    calculate the perimeter of a stadium
    Parameters
    ----------
    width: int
    depth: int

    Returns
    -------
    int
    """
    #radius = depth / 2
    #a = width - depth
    #perimeter = 2 * (np.pi * radius + a)
    perimeter = 2 * (width - depth) + np.pi * depth
    return perimeter


def circle_p(diag):
    """
    calculate the perimeter of a circle
    Parameters
    ----------
    diag: int
    Returns
    -------
    int
    """
    r = diag / 2
    return 2 * np.pi * r
def circle_p2(diag, depth):
    """
    calculate the perimeter of a circle
    Parameters
    ----------
    diag: int
    depth: int
    Returns
    -------
    int
    """
    res = 2 * ((diag + depth)/2)
    return res