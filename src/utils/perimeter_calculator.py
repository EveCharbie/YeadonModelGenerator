import numpy as np


def stadium_perimeter(width, depth):
    radius = depth / 2
    a = width - depth
    perimeter = 2 * (np.pi * radius + a)
    return perimeter


def circle_perimeter(diag):
    r = diag / 2
    return 2 * np.pi * r
