import numpy as np
import PIL
from PIL import Image
from rembg import remove
import glob
import cv2 as cv

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
    """
    x_im, y_im = im.height, im.width
    x_ratio, y_ratio = RESIZE_SIZE / x_im, RESIZE_SIZE / y_im
    min_ratio = min(x_ratio, y_ratio)
    if min_ratio >= 1:
        return im.copy()
    x_resize, y_resize = int(min_ratio * x_im), int(min_ratio * y_im)
    return im.resize((y_resize, x_resize))


def canny_edges(im: np.ndarray, image: np.ndarray):
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
    pil_im = PIL.Image.open(impath).convert("RGB")
    pil_im = _resize(pil_im)
    image = np.asarray(pil_im)
    im = remove(image)
    return pil_im, image, im


def pil_resize_remove_im(undistorted_image: np.ndarray):
    pil_im = Image.fromarray(undistorted_image.astype("uint8"), "RGB")
    pil_im = _resize(pil_im)
    image = np.asarray(pil_im)
    im = remove(image)
    return pil_im, image, im


def undistortion(chessboard_images_path: str, img_with_chessboard_path: str):
    def recalibrate_image(chessboard_images, img_with_chessboard):
        chessboard_size = (5, 5)

        # Create a list to store object points (3D) and image points (2D)
        object_points = []
        image_points = []

        # Define the 3D coordinates of the chessboard corners (assuming a flat board)
        object_points_pattern = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        object_points_pattern[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        for chessboard_image in chessboard_images:
            img = cv.imread(chessboard_image)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

            if ret:
                object_points.append(object_points_pattern)
                image_points.append(corners)
            else:
                print(f"Failed to detect corners in image: {chessboard_image}")

        if not object_points or not image_points:
            print("No valid calibration images found. Check your input images.")

        image_size = (img_with_chessboard.shape[1], img_with_chessboard.shape[0])

        if object_points and image_points:
            ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv.calibrateCamera(object_points, image_points, image_size, None, None)
        else:
            camera_matrix = None
            distortion_coeffs = None

        if camera_matrix is not None:
            undistorted_image = cv.undistort(img_with_chessboard, camera_matrix, distortion_coeffs)
        else:
            undistorted_image = None
            print("Camera calibration failed. Check your calibration images.")

        mean_error = 0
        for i in range(len(object_points)):
            imgpoints2, _ = cv.projectPoints(object_points[i], rvecs[i], tvecs[i], camera_matrix, distortion_coeffs)
        error = cv.norm(image_points[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

        print( "total error: {}".format(mean_error/len(object_points)) )
        #cv.imwrite('undistorted_image.jpg', undistorted_image)
        return undistorted_image, camera_matrix, distortion_coeffs

    chessboard_images = glob.glob(chessboard_images_path)
    pil_im = PIL.Image.open(img_with_chessboard_path).convert('RGB')
    img_with_chessboard = np.asarray(pil_im)
    # get the camera matrix and the distortion_coef and the calibrated image
    calibrated_image, camera_matrix, distortion_coeffs = recalibrate_image(chessboard_images, img_with_chessboard)
    height, width = img_with_chessboard.shape[:2]
    # get the new camera matrix (alpha 0 will destroy some pixel)
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (width, height), 1, (width, height))
    # Undistort the image
    undistorted_img = cv.undistort(img_with_chessboard, camera_matrix, distortion_coeffs, None, new_camera_matrix)
    return undistorted_img



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


def get_ratio(img: np.ndarray, top : int, bot: int):
    if top:
        img = _crop(img, [img.shape[0] / 2.5, 0], [img.shape[0], img.shape[0] / 1.5])
    if bot:
        img = _crop(img, [0, img.shape[0]], [img.shape[1], img.shape[0] / 1.5])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(img, (5, 5), None)
    corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # Draw the corners on the image
    cv.drawChessboardCorners(img, (5, 5), corners2, ret)

    # Get the contour of the chessboard pattern
    hull = cv.convexHull(corners2)
    epsilon = 0.02 * cv.arcLength(hull, True)
    corners_hull = cv.approxPolyDP(hull, epsilon, True)
    chessboard_contour = corners_hull[:4, 0, :]
    cv.drawContours(img, [chessboard_contour.astype(int)], -1, (255, 0, 0), 1)

    ratio = np.linalg.norm(chessboard_contour[0] - chessboard_contour[3])
    ratio2 = np.linalg.norm(chessboard_contour[0] - chessboard_contour[1])
    print(ratio)
    print(ratio2)
    ratio = 9.8 / ratio
    ratio2 = 9.8 / ratio2
    return ratio, ratio2
def get_ratio2(img: np.ndarray, top: int, bot: int):
    if top:
        img = _crop(img, [img.shape[0] / 2.5, 0], [img.shape[0], img.shape[0] / 1.5])
    if bot:
        img = _crop(img, [0, img.shape[0]], [img.shape[1], img.shape[0] / 1.5])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(img, (5, 5), None)
    corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # Draw the corners on the image
    cv.drawChessboardCorners(img, (5, 5), corners2, ret)

    # Get the contour of the chessboard pattern
    hull = cv.convexHull(corners2)
    epsilon = 0.02 * cv.arcLength(hull, True)
    corners_hull = cv.approxPolyDP(hull, epsilon, True)
    chessboard_contour = corners_hull[:4, 0, :]
    cv.drawContours(img, [chessboard_contour.astype(int)], -1, (255, 0, 0), 1)

    ratio = np.linalg.norm(chessboard_contour[0] - chessboard_contour[3])
    ratio2 = np.linalg.norm(chessboard_contour[0] - chessboard_contour[1])
    print(ratio)
    print(ratio2)
    ratio = 9.8 / ratio
    ratio2 = 9.8 / ratio2
    return ratio, ratio2

def get_new_ratio(origin: int, depth: int, width: int, pixel_width: int):
    """
    origin is the real distance between the camera and the person
    depth is the real distance between the wall and the person
    width is the real distance two chessboard in the wall
    """
    res = (depth * width / origin)
    return res / pixel_width

def get_ratio3(img):

    return 0.285,0.285