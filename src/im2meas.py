import cv2 as cv
import numpy as np
import openpifpaf
import PIL


class YeadonModel:
    """A class used to represent a Yeadon Model.

    Attributes
    ----------
    keypoints : dict
        A dictionary containing the keypoints of the image. (Ls0, Ls1, ...)
    """

    def __init__(self, impath: str):
        """Creates a YeadonModel object from an image path.

        Parameters
        ----------
        impath : str
            The path to the image to be processed.

        Returns
        -------
        YeadonModel
            The YeadonModel object with the keypoints of the image.
        """
        pil_im = PIL.Image.open(impath).convert('RGB')
        im = np.asarray(pil_im)
        predictor = openpifpaf.Predictor(
            checkpoint='shufflenetv2k30-wholebody')
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
        data = predictions[0].data
        body_parts_index = {
            "nose": 0,
            "left_ear": 3,
            "right_ear": 4,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10,
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16,
            "left_base_of_thumb": 93,
            "right_base_of_thumb": 114,
            "left_heel": 19,
            "right_heel": 22,
            "left_toe_nail": 17,
            "right_toe_nail": 20
        }
        body_parts_pos = {k: data[v] for k, v in body_parts_index.items()}
        hand_pos = [96, 100, 104, 108, 117, 121, 125,
                    129, 98, 102, 106, 110, 119, 123, 127, 131]
        hand_part_pos = []
        for hand_position in hand_pos:
            hand_part_pos.append(data[hand_position])
        body_parts_pos["left_knuckles"] = np.mean(hand_part_pos[0:3], axis=0)
        body_parts_pos["right_knuckles"] = np.mean(hand_part_pos[4:7], axis=0)
        body_parts_pos["left_nails"] = np.mean(hand_part_pos[8:11], axis=0)
        body_parts_pos["right_nails"] = np.mean(hand_part_pos[12:15], axis=0)
        left_lowest_front_rib_approx = (data[5] + data[11])/2
        body_parts_pos["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data[6] + data[12])/2
        body_parts_pos["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos["left_nipple"] = (
            left_lowest_front_rib_approx + data[5]) / 2
        body_parts_pos["right_nipple"] = (
            right_lowest_front_rib_approx + data[6]) / 2
        body_parts_pos["umbiculus"] = (
            (left_lowest_front_rib_approx*3) + (data[11]*2))/5
        left_arch_approx = (data[17] + data[19]) / 2
        body_parts_pos["left_arch"] = left_arch_approx
        right_arch_approx = (data[20] + data[22]) / 2
        body_parts_pos["right_arch"] = right_arch_approx
        body_parts_pos["left_ball"] = (data[17] + left_arch_approx) / 2
        body_parts_pos["right_ball"] = (data[20] + right_arch_approx) / 2
        body_parts_pos["left_mid_arm"] = (data[5] + data[7]) / 2
        body_parts_pos["right_mid_arm"] = (data[6] + data[8]) / 2
        self.keypoints = {
            "Ls0": body_parts_pos["left_hip"],
            "Ls1": body_parts_pos["umbiculus"],
            "Ls2": body_parts_pos["left_lowest_front_rib"],
            "Ls3": body_parts_pos["left_nipple"],
            "Ls4": body_parts_pos["left_shoulder"],
            "Ls5": self._find_acromion(im, data),
            "Ls6": body_parts_pos["nose"],
            "Ls7": body_parts_pos["left_ear"],
            "Ls8": self._find_top_of_head(im),
            "La0": body_parts_pos["left_shoulder"],
            "La1": body_parts_pos["left_mid_arm"],
            "La2": body_parts_pos["left_elbow"],
            # TODO"La3": body_parts_pos["left_maximum_forearm"],
            "La4": body_parts_pos["left_wrist"],
            "La5": body_parts_pos["left_base_of_thumb"],
            "La6": body_parts_pos["left_knuckles"],
            "La7": body_parts_pos["left_nails"],
            "Lb0": body_parts_pos["right_shoulder"],
            "Lb1": body_parts_pos["right_mid_arm"],
            "Lb2": body_parts_pos["right_elbow"],
            # TODO"Lb3": body_parts_pos["right_maximum_forearm"],
            "Lb4": body_parts_pos["right_wrist"],
            "Lb5": body_parts_pos["right_base_of_thumb"],
            "Lb6": body_parts_pos["right_knuckles"],
            "Lb7": body_parts_pos["right_nails"],
            "Lj0": body_parts_pos["left_hip"],
            # TODO"Lj1": body_parts_pos["left_crotch"],
            # TODO"Lj2": body_parts_pos["left_mid_thigh"],
            "Lj3": body_parts_pos["left_knee"],
            # TODO"Lj4": body_parts_pos["left_maximum_calf"],
            "Lj5": body_parts_pos["left_ankle"],
            "Lj6": body_parts_pos["left_heel"],
            "Lj7": body_parts_pos["left_arch"],
            "Lj8": body_parts_pos["left_ball"],
            "Lj9": body_parts_pos["left_toe_nail"],
            "Lk0": body_parts_pos["right_hip"],
            # TODO"Lk1": body_parts_pos["right_crotch"],
            # TODO"Lk2": body_parts_pos["right_mid_thigh"],
            "Lk3": body_parts_pos["right_knee"],
            # TODO"Lk4": body_parts_pos["right_maximum_calf"],
            "Lk5": body_parts_pos["right_ankle"],
            "Lk6": body_parts_pos["right_heel"],
            "Lk7": body_parts_pos["right_arch"],
            "Lk8": body_parts_pos["right_ball"],
            "Lk9": body_parts_pos["right_toe_nail"]
        }

    def _find_acromion(self, im, data):
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

        # pre-process
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        blurred = cv.medianBlur(gray, 5)
        thresh = cv.adaptiveThreshold(
            blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

        # https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
        r_shoulder, r_ear = data[6], data[4]

        def crop(image, position_1, position_2):
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
            return image[y1:y2, x1:x2].copy()

        shoulder2ear = crop(thresh, r_shoulder, r_ear)

        # 'ignore the head', wipes the pixels going past the ear vertical line
        head_limit = np.min(np.where(shoulder2ear[0] == 0))
        shoulder2ear[:, head_limit::] = 255

        # for now taking the highest point, and the average between ear and shoulder
        crop_offset = [min(r_shoulder[1], r_ear[1]),
                       min(r_shoulder[0], r_ear[0])]
        acromion_y, acromion_x = np.where(shoulder2ear == 0)[
            0][0], shoulder2ear.shape[1] / 2
        acromion = np.array([acromion_y + crop_offset[0],
                            acromion_x + crop_offset[1]])

        return acromion

    def _find_top_of_head(self, image):
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
        grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(grayscale_image, 50, 150)
        _, binary_image = cv.threshold(edges, 0, 255, cv.THRESH_BINARY)

        # the first pixel from top to bottom to find the top_of_head
        top_of_the_head = [np.where(binary_image != 0)[
            1][0], np.where(binary_image != 0)[0][0]]

        return top_of_the_head


if __name__ == '__main__':
    pass
