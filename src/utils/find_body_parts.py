import numpy as np

from src.utils.crop import _crop
from src.utils.get_maximum import *


def find_acromion_right(edges: np.ndarray, ear: np.ndarray, shoulder: np.ndarray, height: int):
    """Finds the acromion given an image and a set of keypoints.

    Parameters
    ----------
    edges : numpy array
        The image to be processed.
    ear : numpy array
        The ear keypoint of the image (given by openpifpaf WholeBody, predictions[0].data generally).
    shoulder : numpy array
        The shoulder keypoint of the image (given by openpifpaf WholeBody, predictions[0].data generally).
    height: int
        A bool to know if we want the acromion for the height or for the perimeter
    Returns
    -------
    numpy array
        The coordinates of the acromion in the image.
    """

    cropped_img = _crop(edges, ear, shoulder)
    cropped_img[: int(len(cropped_img) / 2), :] = 0
    if height:
        cropped_img[:, int(len(cropped_img[0]) / 2.5) :] = 0
    reversed_image_array = np.fliplr(cropped_img)
    acromion = np.array(
        [
            ear[0] - (np.where(reversed_image_array == 255)[1][0]),
            ear[1] + np.where(reversed_image_array == 255)[0][0],
        ]
    )
    return acromion


def find_acromion_left(edges: np.ndarray, data: np.ndarray, height: int):
    """Finds the acromion given an image and a set of keypoints.

    Parameters
    ----------
    edges : numpy array
        The image to be processed.
    data : numpy array
        The keypoints of the image (given by openpifpaf WholeBody, predictions[0].data generally).
    height: int
        A bool to know if we want the acromion for the height or for the perimeter
    Returns
    -------
    numpy array
        The coordinates of the acromion in the image.
    """
    l_ear, l_shoulder = data[0], data[5]
    cropped_img = _crop(edges, l_ear, l_shoulder)
    cropped_img[: int(len(cropped_img) / 2), :] = 0
    if height:
        cropped_img[:,:int(len(cropped_img[0])/1.5)] = 0
    acromion = np.array(
        [
            l_ear[0] + np.where(cropped_img == 255)[1][0],
            l_ear[1] + np.where(cropped_img == 255)[0][0],
        ]
    )
    return acromion


def find_top_of_head(data: np.ndarray, edges: np.ndarray):
    """Finds the top of the head given an image.

    Parameters
    ----------
    data : numpy array
        The keypoints of the image
    edges : numpy array
        The image to be processed.

    Returns
    -------
    numpy array
        The coordinates of the top of the head in the image.
    """
    # the first pixel from top to bottom to find the top_of_head
    nose = data[0]
    point = np.array([nose[1], nose[0]])
    point2 = np.array([point[0] - 5, point[1]])
    vector = np.array([point2[0] - point[0], point2[1] - point[1]])
    angle_radians = np.arctan2(vector[1], vector[0])
    top_of_head = find_edge(point, angle_radians, edges, save=[])
    return np.array(top_of_head[0])


def get_crotch_right_left(edges: np.ndarray, data: np.ndarray):
    # crop the image to see the right hip to the left knee
    crotch_zone = _crop(edges, data[12], data[13])
    # now the cropped image only has the crotch as an edge so we can get it like the head
    crotch_approx_crop = (
        np.where(crotch_zone != 0)[1][0],
        np.where(crotch_zone != 0)[0][1],
    )
    crotch_approx_right, crotch_approx_left = np.array(
        [data[12][0], data[12][1] + crotch_approx_crop[1]]
    ), np.array([data[11][0], data[11][1] + crotch_approx_crop[1]])
    return crotch_approx_right, crotch_approx_left


def get_mid_thigh_right_left(data: np.ndarray, r_crotch: np.ndarray, l_crotch: np.ndarray):
    mid_thigh_right, mid_thigh_left = (data[14][0:2] + r_crotch) / 2, (
        data[13][0:2] + l_crotch
    ) / 2
    return mid_thigh_right, mid_thigh_left

