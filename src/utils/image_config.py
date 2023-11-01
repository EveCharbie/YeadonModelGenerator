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


def canny_edges(im, image):
    grayscale_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    # thresh = cv.adaptiveThreshold(grayscale_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    edged = cv.Canny(grayscale_image, 10, 30)
    # laplacian = cv.Laplacian(thresh, cv.CV_64F)
    # laplacian = np.uint8(np.absolute(laplacian))
    # define a (3, 3) structuring element
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv.dilate(edged, kernel, iterations=2)

    # find the contours in the dilated image
    contours, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    edges = np.zeros(image.shape)
    # draw the contours on a copy of the original image
    cv.drawContours(edges, contours, -1, (0, 255, 0), 3)
    return edges


def create_resize_remove_im(impath):
    pil_im = PIL.Image.open(impath).convert("RGB")
    pil_im = _resize(pil_im)
    image = np.asarray(pil_im)
    im = remove(image)
    return pil_im, image, im


def pil_resize_remove_im(undistorted_image):
    pil_im = Image.fromarray(undistorted_image.astype('uint8'), 'RGB')
    pil_im = _resize(pil_im)
    image = np.asarray(pil_im)
    im = remove(image)
    return pil_im, image, im


def undistortion(chessboard_images_path, img_with_chessboard_pathc):
    def find_chessboard(undistorted_image, draw):
        CHECKERBOARD = (5, 5)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        gray_image = cv.cvtColor(undistorted_image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray_image, CHECKERBOARD,
                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            corners2 = cv.cornerSubPix(gray_image, corners, (3, 3), (-1, -1), criteria)
        if draw:
            cv.drawChessboardCorners(undistorted_image, (5, 5), corners2, ret)
        return undistorted_image, corners2

    def recalibrate_image(chessboard_images, img_with_chessboard):
        # Define the size of the chessboard pattern
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
            ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv.calibrateCamera(object_points, image_points,
                                                                                     image_size, None, None)
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

        print("total error: {}".format(mean_error / len(object_points)))
        # cv.imwrite('undistorted_image.jpg', undistorted_image)
        return undistorted_image, camera_matrix, distortion_coeffs

    def undistor(calibrated_image, H):
        height, width = calibrated_image.shape[:2]

        original_corners = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=np.float32)

        # Warp the corners using the homography matrix
        warped_corners = cv.perspectiveTransform(original_corners.reshape(-1, 1, 2), H)

        # Find the new dimensions of the output image to encompass the transformed corners
        min_x = int(min(warped_corners[:, 0, 0]))
        max_x = int(max(warped_corners[:, 0, 0]))
        min_y = int(min(warped_corners[:, 0, 1]))
        max_y = int(max(warped_corners[:, 0, 1]))

        output_width = max_x - min_x
        output_height = max_y - min_y

        # Calculate the translation matrix to move the warped image
        translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float32)

        # Combine the perspective transformation and translation
        transform_matrix = translation_matrix.dot(H)

        # Warp the original image into the output
        output_image = cv.warpPerspective(calibrated_image, transform_matrix, (output_width, output_height))
        return output_image

    chessboard_images = glob.glob(chessboard_images_path)
    pil_im = PIL.Image.open(img_with_chessboard_pathc).convert('RGB')
    img_with_chessboard = np.asarray(pil_im)
    # get the camera matrix and the distortion_coef as well a the image calibrated
    calibrated_image, camera_matrix, distortion_coeffs = recalibrate_image(chessboard_images, img_with_chessboard)
    # transform the image into pil image to use the resize function
    PIL_image = Image.fromarray(calibrated_image.astype('uint8'), 'RGB')
    calibrated_image = np.asarray(_resize(PIL_image))
    # get the corners for the two chessboards
    undistorted_image2, corners = find_chessboard(calibrated_image.copy(), 1)
    undistorted_image_drawed, corners2 = find_chessboard(undistorted_image2, 0)
    H, _ = cv.findHomography(corners2, corners)
    undistorted_image = undistor(calibrated_image, H)
    return undistorted_image

def better_edges(edges, data):
    crotch_zone = _crop(edges, data[12], data[13])
    height = np.array([np.where(crotch_zone != 0)[1][0], np.where(crotch_zone != 0)[0][1]])
    crotch = np.array([round(data[12][0] + height[0]), round(data[12][1] + height[1])])
    crotch_approx = np.array([round(data[12][0] + height[0]), round(data[12][1] + height[1] - 5)])
    cv.line(edges, crotch, crotch_approx, (0, 255, 0), 7)
    return edges
def get_ratio(img, top, bot):
    if top:
        img = _crop(img, [img.shape[0] / 2.5, 0], [img.shape[0], img.shape[0] / 1.5])
    if bot:
        img = _crop(img, [0, img.shape[0]], [img.shape[1] / 2, img.shape[0] / 2])

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(img, (5, 5), None)
    corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # Draw the corners on the image
    cv.drawChessboardCorners(img, (5, 5), corners2, ret)

    # Get the contour of the chessboard pattern
    rect = cv.boundingRect(corners2)
    x, y, w, h = rect
    chessboard_contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])

    # Draw the contour on the image
    cv.drawContours(img, [chessboard_contour], -1, (255, 0, 0), 1)

    ratio = np.linalg.norm(chessboard_contour[0] - chessboard_contour[1])
    ratio2 = np.linalg.norm(chessboard_contour[0] - chessboard_contour[3])
    print(ratio)
    print(ratio2)
    ratio = 9.8/ratio
    ratio2 = 9.8/ratio2
    return ratio, ratio2