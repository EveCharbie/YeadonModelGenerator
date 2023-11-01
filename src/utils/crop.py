import numpy as np

def _crop(image, position_1, position_2):
    """Return the cropped image given two positions.

    Parameters
    ----------
    image : numpy array
        The image to be cropped.
    position_1 : tuple
        The position of the first corner of the image to be cropped.
    position_2 : tuple
        The position of the second corner of the image to be cropped.

    Returns
    -------
    numpy array
        The cropped image.
    """
    x1, y1 = map(int, position_1[0:2])
    x2, y2 = map(int, position_2[0:2])
    if x1 > x2:
        x2, x1 = x1, x2
    if y1 > y2:
        y2, y1 = y1, y2
    x = image[y1:y2, x1:x2].copy()
    return x