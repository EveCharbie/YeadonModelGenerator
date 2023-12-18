import numpy as np


def stad_p(width: float, depth: float):
    """
    calculate the perimeter of a stadium
    Parameters
    ----------
    width: float
    depth: float

    Returns
    -------
    float
    """
    # radius = depth / 2
    # a = width - depth
    # perimeter = 2 * (np.pi * radius + a)
    # perimeter = 2 * (width - depth) + np.pi * depth
    perimeter = 2 * np.pi * np.sqrt((((width / 2) ** 2) + ((depth / 2) ** 2)) / 2)
    return perimeter


def circle_p(diag: float):
    """
    calculate the perimeter of a circle
    Parameters
    ----------
    diag: float
    Returns
    -------
    float
    """
    r = diag / 2
    return 2 * np.pi * r


def circle_p2(diag: float, depth: float):
    """
    calculate the perimeter of a circle
    Parameters
    ----------
    diag: float
    depth: float
    Returns
    -------
    float
    """
    res = 2 * ((diag + depth) / 2)
    return res
