import numpy as np

from src.utils.crop import _crop

def find_acromion_right(edges, data, height):
    """Finds the acromion given an image and a set of keypoints.

    Parameters
    ----------
    im : numpy array
        The image to be processed.
    data : numpy array
        The keypoints of the image (given by openpifpaf WholeBody, predictions[0].data generally).

    Returns
    -------
    numpy array
        The coordinates of the acromion in the image.
    """

    r_ear, r_shoulder = data[0], data[6]
    cropped_img = _crop(edges, r_ear, r_shoulder)
    if height:
        r_ear, r_shoulder = data[0], data[6]
        cropped_img = _crop(edges, r_ear, r_shoulder)
        cropped_img[:int(len(cropped_img) / 2), :] = 0
        cropped_img[:, int(len(cropped_img[0]) / 2.5):] = 0
        reversed_image_array = np.fliplr(cropped_img)
    else:
        cropped_img[:int(len(cropped_img) / 2), :] = 0
        # cropped_img[:, :int(len(cropped_img[0]) / 2)] = 0
        reversed_image_array = np.fliplr(cropped_img)
    acromion = np.array([r_ear[0] - (np.where(reversed_image_array == 255)[1][0]),
                         r_ear[1] + np.where(reversed_image_array == 255)[0][0]])
    return acromion


def find_acromion_left(edges, data, height):
    """Finds the acromion given an image and a set of keypoints.

    Parameters
    ----------
    im : numpy array
        The image to be processed.
    data : numpy array
        The keypoints of the image (given by openpifpaf WholeBody, predictions[0].data generally).

    Returns
    -------
    numpy array
        The coordinates of the acromion in the image.
    """
    if height:
        l_ear, l_shoulder = data[0], data[5]
        cropped_img = _crop(edges, l_ear, l_shoulder)
        cropped_img[:int(len(cropped_img) / 2), :] = 0
        cropped_img[:int(len(cropped_img[0]) / 1.5), :] = 0
        reversed_image_array = np.fliplr(cropped_img)
        acromion = np.array([l_ear[0] + np.where(reversed_image_array == 255)[1][0],
                             l_ear[1] + np.where(reversed_image_array == 255)[0][0]])
    else:
        l_ear, l_shoulder = data[0], data[5]
        cropped_img = _crop(edges, l_ear, l_shoulder)
        cropped_img[:int(len(cropped_img) / 2), :] = 0
        # cropped_img[:, :int(len(cropped_img[0]) / 2)] = 0
        acromion = np.array(
            [l_ear[0] + np.where(cropped_img == 255)[1][0], l_ear[1] + np.where(cropped_img == 255)[0][0]])
    return acromion
def find_top_of_head(edges):
    """Finds the top of the head given an image.

    Parameters
    ----------
    image : numpy array
        The image to be processed.

    Returns
    -------
    numpy array
        The coordinates of the top of the head in the image.
    """
    # the first pixel from top to bottom to find the top_of_head
    top_of_head = [
        np.where(edges != 0)[1][0],
        np.where(edges != 0)[0][0],
    ]
    return top_of_head
def get_crotch_right_left(edges, data):

    # crop the image to see the right hip to the left knee
    crotch_zone = _crop(edges, data[12], data[13])
    # now the cropped image only has the crotch as an edge so we can get it like the head
    crotch_approx_crop = np.where(crotch_zone != 0)[1][0], np.where(crotch_zone != 0)[0][1]
    crotch_approx_right = np.array([data[12][0], data[12][1] + crotch_approx_crop[1]])
    crotch_approx_left = np.array([data[11][0], data[11][1] + crotch_approx_crop[1]])
    return crotch_approx_right, crotch_approx_left

def get_mid_thigh_right_left(data, r_crotch, l_crotch):
    mid_thigh_right = (data[14][0:2] + r_crotch) / 2
    mid_thigh_left = (data[13][0:2] + l_crotch) / 2
    return mid_thigh_right, mid_thigh_left