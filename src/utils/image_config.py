import numpy as np
import PIL
from PIL import Image
from rembg import remove
import cv2 as cv
from scipy.ndimage import rotate
import os

from src.utils.crop import _crop

RESIZE_SIZE = 900  # the maximum size of the image to be processed (in pixels)


def _resize(im):
    """Resizes an image given a maximum size of RESIZE_SIZE.

    Parameters
    ----------
    im : PIL Image
        The image to be resized.

    Returns
    -------
    PIL Image
        The resized image.
    min_ratio
        The ratio of the resize
    """
    x_im, y_im = im.height, im.width
    x_ratio, y_ratio = RESIZE_SIZE / x_im, RESIZE_SIZE / y_im
    min_ratio = min(x_ratio, y_ratio)
    if min_ratio >= 1:
        return im.copy()
    x_resize, y_resize = int(min_ratio * x_im), int(min_ratio * y_im)
    return im.resize((y_resize, x_resize))


def canny_edges(im: np.ndarray, image: np.ndarray):
    """ Apply canny to the given image and returns the edges

    Parameters
    ----------
    im: np.ndarray
    image: np.ndarray

    Returns
    -------
    PIL Image
        the edges of the image after canny
    """
    grayscale_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    edged = cv.Canny(grayscale_image, 10, 30)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv.dilate(edged, kernel, iterations=1)

    contours, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    edges = np.zeros(image.shape)
    # draw the contours on a copy of the original image
    cv.drawContours(edges, contours, -1, (0, 255, 0), 2)
    return edges


def thresh(im: np.ndarray, image: np.ndarray, line_size):
    """ Apply thresh to the given image and returns the edges

    Parameters
    ----------
    im: np.ndarray
    image: np.ndarray

    Returns
    -------
    PIL Image
        the edges of the image after thresh
    """
    grayscale_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    _, binary_silhouette = cv.threshold(grayscale_image, 5, 255, cv.THRESH_BINARY)

    # find the contours in the grayscaled image
    contours, _ = cv.findContours(binary_silhouette, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    edges = np.zeros(image.shape)
    # draw the contours on a copy of the original image
    cv.drawContours(edges, contours, -1, (0, 255, 0), line_size)
    cv.drawContours(image, contours, -1, (0, 255, 0), line_size)
    return edges


def create_resize_remove_im(impath: str):
    """
    Take and image path and return image without background, resized and the pil version
    Parameters
    ----------
    impath: str

    Returns
    -------
    PIL Image
    """
    pil_im = PIL.Image.open(impath).convert("RGB")
    pil_im = _resize(pil_im)
    image_resized = np.asarray(pil_im)
    im = remove(image_resized)
    pil_im = pil_im.transpose(Image.ROTATE_270)
    image_resized = rotate(image_resized, -90, reshape=True, mode='nearest')
    im = rotate(im, -90, reshape=True, mode='nearest')
    return pil_im, image_resized, im


def better_edges(edges: np.ndarray, data: np.ndarray):
    crotch_zone = _crop(edges, data[12], data[13])
    height = np.array(
        [np.where(crotch_zone != 0)[1][0], np.where(crotch_zone != 0)[0][1]]
    )
    crotch = np.array([round(data[12][0] + height[0]), round(data[12][1] + height[1])])
    crotch_approx = np.array(
        [round(data[12][0] + height[0]), round(data[12][1] + height[1] - 5)]
    )
    cv.line(edges, crotch, crotch_approx, (0, 255, 0), 7)
    return edges



def get_ratio(img: np.ndarray):
    pattern_size = (5, 5)
    img1 = _crop(img, [0, 0], [img.shape[1] / 2, img.shape[0] / 2])
    img2 = _crop(img, [img.shape[1] / 2, img.shape[0] / 2], [img.shape[1], 0])
    img3 = _crop(img, [img.shape[1] / 2, img.shape[0] / 2], [img.shape[1], img.shape[0] / 1.3])
    img4 = _crop(img, [0, img.shape[0] / 2], [img.shape[1] / 2, img.shape[0]])

    imgs = []
    imgs.append(img1)
    imgs.append(img2)
    imgs.append(img3)
    imgs.append(img4)
    chess_points = []
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.00001)

    for image in imgs:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(image, pattern_size, None)
        if ret:  # Check if corners were found
            corners2 = cv.cornerSubPix(gray, corners, pattern_size, (-1, -1), criteria)
            # Get the contour of the chessboard pattern
            hull = cv.convexHull(corners2)
            epsilon = 0.02 * cv.arcLength(hull, True)
            corners_hull = cv.approxPolyDP(hull, epsilon, True)
            chessboard_contour = corners_hull[:4, 0, :]
            chess_points.append(np.mean(chessboard_contour, axis=0))
            cv.drawContours(image, [chessboard_contour.astype(int)], -1, (255, 0, 0), 1)
        else:
            print("Chessboard corners not found")
    chess_points[1] = chess_points[1] + [img.shape[1] / 2, 0]
    chess_points[2] = chess_points[2] + [img.shape[1] / 2, img.shape[0] / 2]
    chess_points[3] = chess_points[3] + [0, img.shape[0] / 2]
    ratio = np.linalg.norm(chess_points[0] - chess_points[1])
    return ratio, ratio


def get_new_ratio(origin: float, depth: float, width: int, pixel_width: int):
    """
    origin is the real distance between the camera and the person
    depth is the real distance between the wall and the person
    width is the real distance two chessboard in the wall
    """
    res = (depth * width / origin)
    return res / pixel_width, res / pixel_width
def get_ratio_meas_top(elbow, wrist):
    #return  25 / np.linalg.norm(elbow - wrist), 25 / np.linalg.norm(elbow - wrist)
    return 23 / np.linalg.norm(elbow - wrist), 23 / np.linalg.norm(elbow - wrist)
def get_ratio_meas_bottom(knee, ankle):
    #return 42 / np.linalg.norm(knee - ankle), 42 / np.linalg.norm(knee - ankle)
    return 41 / np.linalg.norm(knee - ankle), 41 / np.linalg.norm(knee - ankle)

def save_img(image, image_r_side, image_tuck, image_r_tuck, image_pike, name):
    if not os.path.exists(f"{name}_dir"):
        os.mkdir(f"{name}_dir")
    img = Image.fromarray(image)
    img.save(f"{name}_dir/{name}_front_t.jpg")
    img = Image.fromarray(image_r_side)
    img.save(f"{name}_dir/{name}_side.jpg")
    img = Image.fromarray(image_tuck)
    img.save(f"{name}_dir/{name}_tuck.jpg")
    img = Image.fromarray(image_r_tuck)
    img.save(f"{name}_dir/{name}_r_tuck_t.jpg")
    img = Image.fromarray(image_pike)
    img.save(f"{name}_dir/{name}_pike_t.jpg")

