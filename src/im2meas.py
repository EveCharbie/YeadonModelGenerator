import cv2 as cv
import numpy as np
import openpifpaf
import PIL
from rembg import remove
from PIL import Image
import glob

import matplotlib.pyplot as plt

RESIZE_SIZE = 900  # the maximum size of the image to be processed (in pixels)


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
        # front
        undistorted_image = self._undistortion('img/martin/chessboards/*', "img/martin/mar_front_t.jpg")
        #undistorted_image = self._undistortion('img/william/chessboards/*', "img/william/william_front_t.jpg")
        #undistorted_image = self._undistortion('img/chessboards/*', "img/al_front_t.jpg")

        pil_im, image, im = self._pil_resize_remove_im(undistorted_image)

        edges = self._canny_edges(im, image)
        predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k30-wholebody")
        predictor_simple = openpifpaf.Predictor(checkpoint="shufflenetv2k30")
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
        # You can find the index here:
        # https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
        # as "predictions" is an array the index starts at 0 and not at 1 like in the github
        data = predictions[0].data[:, 0:2]
        image_chessboard = image.copy()
        self.ratio, self.ratio2 = self._get_ratio(image_chessboard, 0, 0)
        self.bottom_ratio, self.bottom_ratio2 = self._get_ratio(image_chessboard, 0, 0)

        # right side
        #undistorted_r_image = self._undistortion('img/chessboards/*', "img/al_r_side.jpg")
        pil_r_side_im, image_r_side, im_r_side = self._create_resize_remove_im("img/martin/mar_r_side.jpg")
        #pil_r_side_im, image_r_side, im_r_side = self._create_resize_remove_im("img/william/william_r_side.jpg")
        #pil_r_side_im, image_r_side, im_r_side = self._create_resize_remove_im("img/al_r_side.jpg")


        edges_r_side = self._canny_edges(im_r_side, image_r_side)
        predictions2, gt_anns2, image_meta2 = predictor_simple.pil_image(pil_r_side_im)
        data_r_side = predictions2[0].data[:, 0:2]
        image_r_side_chessboard = image_r_side.copy()
        self.ratio_r_side, self.ratio_r_side2 = self._get_ratio(image_r_side_chessboard, 0,0)
        # front T pose but with the hand to the top
        #undistorted_up_image = self._undistortion('img/william/chessboards/*', "img/william/wil_front_t_up.jpg")
        pil_up_im, image_up, im_up = self._create_resize_remove_im("img/martin/mar_front_t_up.jpg")
        #pil_up_im, image_up, im_up = self._create_resize_remove_im("img/william/william_front_t_up.jpg")
        #pil_up_im, image_up, im_up = self._create_resize_remove_im("img/al_front_t_up.jpg")

        edges_up = self._canny_edges(im_up, image_up)
        predictions4, gt_anns4, image_meta4 = predictor.pil_image(pil_up_im)
        data_up = predictions4[0].data[:, 0:2]
        image_up_chessboard = image_up.copy()
        print("data_up",data_up[96])
        print(data_up[97])
        print(data_up[98])

        self.ratio_up, self._ratio_up2 = self._get_ratio(image_up_chessboard, 0, 0)
        # front pike
        #undistorted_pike_image = self._undistortion('img/chessboards/*', "img/al_front_pike.jpg")
        pil_pike_im, image_pike, im_pike = self._create_resize_remove_im("img/martin/mar_front_pike.jpg")
        #pil_pike_im, image_pike, im_pike = self._create_resize_remove_im("img/william/william_front_pike.jpg")
        #pil_pike_im, image_pike, im_pike = self._create_resize_remove_im("img/al_front_pike.jpg")

        img = Image.fromarray(image_pike)
        img.save("f_pike.jpg")
        edges_pike = self._canny_edges(im_pike, image_pike)
        predictions5, gt_anns5, image_meta5 = predictor.pil_image(pil_pike_im)
        data_pike = predictions5[0].data[:, 0:2]
        image_pike_chessboard = image_pike.copy()
        self.ratio_pike, self.ratio_pike2 = self._get_ratio(image_pike_chessboard, 0,0)
        #self.ratio_pike, self.ratio_pike2 = self._get_ratio(image_pike, 0, 1)
        # right side pike
        #undistorted_r_pike_image = self._undistortion('img/chessboards/*', "img/al_r_pike.jpg")
        pil_l_pike_im, image_l_pike, im_l_pike = self._create_resize_remove_im("img/martin/mar_r_pike.jpg")
        #pil_l_pike_im, image_l_pike, im_l_pike = self._create_resize_remove_im("img/william/william_r_pike.jpg")
        #pil_l_pike_im, image_l_pike, im_l_pike = self._create_resize_remove_im("img/al_r_pike.jpg")

        edges_l_pike = self._canny_edges(im_l_pike, image_l_pike)
        predictions6, gt_anns6, image_meta6 = predictor.pil_image(pil_l_pike_im)
        data_l_pike = predictions6[0].data[:, 0:2]
        image_r_pike_chessboard = image_l_pike.copy()
        self.ratio_l_pike, self.ratio_l_pike2 = self._get_ratio(image_r_pike_chessboard, 0, 0)
        #self.ratio_l_pike, self.ratio_l_pike2 = self._get_ratio(image_l_pike, 0, 1)
        # front
        body_parts_index = {
            "nose": 56,
            "nose_per": 80,
            "left_knuckle": 100,
            "right_knuckle": 121,
            "left_nail": 102,
            "right_nail": 123,
            "left_ear": 39,
            "right_ear": 23,
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
            "right_toe_nail": 20,
        }
        hand_pos = [
            97,
            101,
            105,
            109,
            118,
            122,
            126,
            130,
            98,
            102,
            106,
            110,
            119,
            123,
            127,
            131,
        ]
        body_parts_pos = {k: data[v] for k, v in body_parts_index.items()}
        # front up
        body_parts_pos_up = {k: data_up[v] for k, v in body_parts_index.items()}

        # right side
        body_parts_index_r = {
            "nose": 0,
            "right_ear": 4,
            "right_shoulder": 6,
            "right_elbow": 8,
            "right_wrist": 10,
            "right_hip": 12,
            "right_knee": 14,
            "right_ankle": 16,
        }

        body_parts_pos_r = {k: data_r_side[v] for k, v in body_parts_index_r.items()}
        # front pike
        body_parts_index_pike = {
            "left_hip": 11,
            "right_hip": 12,
            "left_knee": 13,
            "right_knee": 14,
            "left_ankle": 15,
            "right_ankle": 16,
            "left_heel": 19,
            "right_heel": 22,
            "left_toe_nail": 17,
            "right_toe_nail": 20,
        }
        body_parts_pos_pike = {k: data_pike[v] for k, v in body_parts_index_pike.items()}
        # left side pike
        body_parts_index_l_pike = {
            "right_ear": 4,
            "right_shoulder": 6,
            "right_elbow": 8,
            "right_hip": 12,
            "right_knee": 14,
            "right_ankle": 16,
            "right_heel": 22,
            "right_toe_nail": 20,
        }
        body_parts_pos_l_pike = {k: data_l_pike[v] for k, v in body_parts_index_l_pike.items()}
        # front
        hand_part_pos = []
        for hand_position in hand_pos:
            hand_part_pos.append(data[hand_position])
        body_parts_pos["left_knuckles"] = np.mean(hand_part_pos[0:3], axis=0)
        body_parts_pos["right_knuckles"] = np.mean(hand_part_pos[4:7], axis=0)
        body_parts_pos["left_nails"] = np.mean(hand_part_pos[8:11], axis=0)
        body_parts_pos["right_nails"] = np.mean(hand_part_pos[12:15], axis=0)
        for hand_position in hand_pos:
            if data[hand_position][0] < 0:
                body_parts_pos["left_knuckles"] = data[100]
                body_parts_pos["right_knuckles"] = data[121]
                body_parts_pos["left_nails"] = data[102]
                body_parts_pos["right_nails"] = data[123]
        left_lowest_front_rib_approx = (data[5] + data[11]) / 2
        body_parts_pos["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data[6] + data[12]) / 2
        body_parts_pos["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos["left_shoulder_perimeter_width"] = self._get_maximum_point(data[7], data[5], edges)
        body_parts_pos["right_shoulder_perimeter_width"] = self._get_maximum_point(data[8], data[6], edges)
        body_parts_pos["left_nipple"] = (left_lowest_front_rib_approx + data[5]) / 2
        body_parts_pos["right_nipple"] = (right_lowest_front_rib_approx + data[6]) / 2
        body_parts_pos["left_umbiculus"] = (
                                                   (left_lowest_front_rib_approx * 3) + (data[11] * 2)
                                           ) / 5
        body_parts_pos["right_umbiculus"] = (
                                                    (right_lowest_front_rib_approx * 3) + (data[12] * 2)
                                            ) / 5
        left_arch_approx = (data[17] + data[19]) / 2
        body_parts_pos["left_arch"] = left_arch_approx
        right_arch_approx = (data[20] + data[22]) / 2
        body_parts_pos["right_arch"] = right_arch_approx
        body_parts_pos["left_ball"] = (data[17] + left_arch_approx) / 2
        body_parts_pos["right_ball"] = (data[20] + right_arch_approx) / 2
        body_parts_pos["left_mid_arm"] = (data[5] + data[7]) / 2
        body_parts_pos["right_mid_arm"] = (data[6] + data[8]) / 2
        body_parts_pos["left_acromion"] = self._find_acromion_left(edges, data, 0)
        body_parts_pos["right_acromion"] = self._find_acromion_right(edges, data, 0)
        body_parts_pos["left_acromion_height"] = self._find_acromion_left(edges, data, 1)
        body_parts_pos["right_acromion_height"] = self._find_acromion_right(edges, data, 1)
        body_parts_pos["top_of_head"] = self._find_top_of_head(edges)
        body_parts_pos["left_mid_elbow_wrist"] = (data[9] + data[7]) / 2
        body_parts_pos["right_mid_elbow_wrist"] = (data[10] + data[8]) / 2
        body_parts_pos["right_maximum_forearm"] = self._get_maximum_point(body_parts_pos["right_mid_elbow_wrist"], data[8], edges)
        body_parts_pos["left_maximum_forearm"] = self._get_maximum_point(body_parts_pos["left_mid_elbow_wrist"], data[7], edges)
        body_parts_pos["right_maximum_calf"] = self._get_maximum_point(data[16], data[14], edges)
        body_parts_pos["left_maximum_calf"] = self._get_maximum_point(data[15], data[13], edges)
        body_parts_pos["right_crotch"], body_parts_pos["left_crotch"] = self._get_crotch_right_left(edges, data)
        body_parts_pos["right_mid_thigh"], body_parts_pos["left_mid_thigh"] = self._get_mid_thigh_right_left(data,
                                                                                                             body_parts_pos[
                                                                                                                 "right_crotch"],
                                                                                                             body_parts_pos[
                                                                                                                 "left_crotch"])
        body_parts_pos["left_crotch_width"] = self._get_maximum_start(body_parts_pos["left_crotch"], body_parts_pos["left_knee"], edges) * self.bottom_ratio
        body_parts_pos["right_crotch_width"] = self._get_maximum_start(body_parts_pos["right_crotch"], body_parts_pos["right_knee"], edges) * self.bottom_ratio
        if body_parts_pos["left_crotch_width"] > 30:
            body_parts_pos["left_crotch_width"] /= 2
        if body_parts_pos["right_crotch_width"] > 30:
            body_parts_pos["right_crotch_width"] /= 2
        # print(body_parts_pos)
        # front up
        left_lowest_front_rib_approx = (data_up[5] + data_up[11]) / 2
        body_parts_pos_up["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data_up[6] + data_up[12]) / 2
        body_parts_pos_up["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos_up["left_nipple"] = (left_lowest_front_rib_approx + data_up[5]) / 2
        body_parts_pos_up["right_nipple"] = (right_lowest_front_rib_approx + data_up[6]) / 2
        body_parts_pos_up["left_umbiculus"] = (
                                                   (left_lowest_front_rib_approx * 3) + (data_up[11] * 2)
                                           ) / 5
        body_parts_pos_up["right_umbiculus"] = (
                                                    (right_lowest_front_rib_approx * 3) + (data_up[12] * 2)
                                            ) / 5
        left_arch_approx = (data_up[17] + data_up[19]) / 2
        body_parts_pos_up["left_arch"] = left_arch_approx
        right_arch_approx = (data_up[20] + data_up[22]) / 2
        body_parts_pos_up["right_arch"] = right_arch_approx
        body_parts_pos_up["left_ball"] = (data_up[17] + left_arch_approx) / 2
        body_parts_pos_up["right_ball"] = (data_up[20] + right_arch_approx) / 2
        body_parts_pos_up["left_mid_arm"] = (data_up[5] + data_up[7]) / 2
        body_parts_pos_up["right_mid_arm"] = (data_up[6] + data_up[8]) / 2
        body_parts_pos_up["left_acromion"] = self._find_acromion_left(edges_up, data_up, 0)
        body_parts_pos_up["right_acromion"] = self._find_acromion_right(edges_up, data_up, 0)
        body_parts_pos_up["top_of_head"] = self._find_top_of_head(edges_up)
        body_parts_pos_up["right_maximum_forearm"] = self._get_maximum_point(data_up[10], data_up[8], edges_up)
        body_parts_pos_up["left_maximum_forearm"] = self._get_maximum_point(data_up[9], data_up[7], edges_up)
        body_parts_pos_up["right_maximum_calf"] = self._get_maximum_point(data_up[16], data_up[14], edges_up)
        body_parts_pos_up["left_maximum_calf"] = self._get_maximum_point(data_up[15], data_up[13], edges_up)
        # right side
        right_lowest_front_rib_approx = np.array([data_r_side[12][0],(data_r_side[6][1] + data_r_side[12][1]) / 2])
        body_parts_pos_r["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos_r["right_nipple"] = np.array([right_lowest_front_rib_approx[0],(right_lowest_front_rib_approx[1] + data_r_side[6][1]) / 2])
        body_parts_pos_r["right_umbiculus"] = (
                                                      (right_lowest_front_rib_approx * 3) + (data_r_side[12] * 2)
                                              ) / 5
        # front pike
        point, dist = self._get_maximum_pit(data_pike[16], edges_pike)
        body_parts_pos_pike["right_toe_nail"] = np.array([point[0], point[1] - 2 / self.ratio_pike])
        body_parts_pos_pike["right_ball"] = self._get_maximum_point(data_pike[16], body_parts_pos_pike["right_toe_nail"], edges_pike)
        body_parts_pos_pike["right_arch"] = (data_pike[16] + body_parts_pos_pike["right_ball"]) / 2
        # left side pike
        body_parts_pos_l_pike["right_toe_nail"], dist = self._get_maximum_pit(data_l_pike[16], edges_l_pike)
        body_parts_pos_l_pike["right_heel"] = self._get_maximum_point(data_l_pike[16], body_parts_pos_l_pike["right_toe_nail"], edges_l_pike)
        body_parts_pos_l_pike["right_ball"] = np.array([body_parts_pos_l_pike["right_toe_nail"][0], body_parts_pos_l_pike["right_toe_nail"][1] - 3 / self.ratio_l_pike])
        body_parts_pos_l_pike["right_arch"] = (body_parts_pos_l_pike["right_heel"] + body_parts_pos_l_pike["right_ball"]) / 2
        print(self._get_maximum_start(data_up[96], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up)

        # print(body_parts_pos_r)
        self.keypoints = {
            "Ls0": body_parts_pos["left_hip"],
            "Ls1": body_parts_pos["left_umbiculus"],
            "Ls2": body_parts_pos["left_lowest_front_rib"],
            "Ls3": body_parts_pos["left_nipple"],
            "Ls4": body_parts_pos["left_shoulder"],
            "Ls5": body_parts_pos["left_acromion"],
            "Ls6": body_parts_pos["nose"],
            "Ls7": body_parts_pos["left_ear"],
            "Ls8": body_parts_pos["top_of_head"],
            "La0": body_parts_pos["left_shoulder"],
            "La1": body_parts_pos["left_mid_arm"],
            "La2": body_parts_pos["left_elbow"],
            "La3": body_parts_pos["left_maximum_forearm"],
            "La4": body_parts_pos["left_wrist"],
            "La5": body_parts_pos["left_base_of_thumb"],
            "La6": body_parts_pos["left_knuckles"],
            "La7": body_parts_pos["left_nails"],
            "Lb0": body_parts_pos["right_shoulder"],
            "Lb1": body_parts_pos["right_mid_arm"],
            "Lb2": body_parts_pos["right_elbow"],
            "Lb3": body_parts_pos["right_maximum_forearm"],
            "Lb4": body_parts_pos["right_wrist"],
            "Lb5": body_parts_pos["right_base_of_thumb"],
            "Lb6": body_parts_pos["right_knuckles"],
            "Lb7": body_parts_pos["right_nails"],
            "Lj0": body_parts_pos["left_hip"],
            "Lj1": body_parts_pos["left_crotch"],
            "Lj2": body_parts_pos["left_mid_thigh"],
            "Lj3": body_parts_pos["left_knee"],
            "Lj4": body_parts_pos["left_maximum_calf"],
            "Lj5": body_parts_pos["left_ankle"],
            "Lj6": body_parts_pos["left_heel"],
            "Lj7": body_parts_pos["left_arch"],
            "Lj8": body_parts_pos["left_ball"],
            "Lj9": body_parts_pos["left_toe_nail"],
            "Lk0": body_parts_pos["right_hip"],
            "Lk1": body_parts_pos["right_crotch"],
            "Lk2": body_parts_pos["right_mid_thigh"],
            "Lk3": body_parts_pos["right_knee"],
            "Lk4": body_parts_pos["right_maximum_calf"],
            "Lk5": body_parts_pos["right_ankle"],
            "Lk6": body_parts_pos["right_heel"],
            "Lk7": body_parts_pos["right_arch"],
            "Lk8": body_parts_pos["right_ball"],
            "Lk9": body_parts_pos["right_toe_nail"],
            "Ls1L": abs(body_parts_pos["left_umbiculus"][1] - body_parts_pos["left_hip"][1]) * self.ratio2,
            "Ls2L": abs(
                body_parts_pos["left_lowest_front_rib"][1]
                - body_parts_pos["left_hip"][1]
            ) * self.ratio2,
            "Ls3L": abs(
                body_parts_pos["left_nipple"][1] - body_parts_pos["left_hip"][1]
            ) * self.ratio2,
            "Ls4L": abs(
                body_parts_pos["left_shoulder"][1] - body_parts_pos["left_hip"][1]
            ) * self.ratio2,
            "Ls5L": abs(body_parts_pos["left_acromion_height"][1] - body_parts_pos["left_hip"][1]) * self.ratio2,
            "Ls6L": abs(body_parts_pos["left_acromion_height"][1] - body_parts_pos["nose"][1]) * self.ratio2,
            "Ls7L": np.linalg.norm(body_parts_pos["left_acromion_height"] - body_parts_pos["left_ear"]) * self.ratio2,
            "Ls8L": abs(
                body_parts_pos["left_acromion_height"][1] - body_parts_pos["top_of_head"][1]
            ) * self.ratio2,

            "Ls0p": self._stadium_perimeter(
                self._get_maximum_start(body_parts_pos["right_hip"], body_parts_pos["right_knee"], edges) * self.ratio,
                self._get_maximum_start(body_parts_pos_r["right_hip"], body_parts_pos_r["right_knee"],
                                        edges_r_side) * self.ratio_r_side),
            "Ls1p": self._stadium_perimeter(self._get_maximum_line(body_parts_pos["left_umbiculus"], body_parts_pos["right_umbiculus"], edges) * self.ratio,self._get_maximum_start(body_parts_pos_r["right_umbiculus"], body_parts_pos_r["right_knee"],edges_r_side) * self.ratio_r_side),
            "Ls2p": self._stadium_perimeter(max(self._get_maximum_line(body_parts_pos["right_lowest_front_rib"],
                                                                   body_parts_pos["left_lowest_front_rib"], edges), np.linalg.norm(body_parts_pos["right_lowest_front_rib"] -
                                                                   body_parts_pos["left_lowest_front_rib"])) * self.ratio,
                                            self._get_maximum_start(body_parts_pos_r["right_lowest_front_rib"],
                                                                    body_parts_pos_r["right_hip"], edges_r_side) * self.ratio_r_side),
            "Ls3p": self._stadium_perimeter(
                self._get_maximum_line(body_parts_pos["right_nipple"], body_parts_pos["left_nipple"], edges) * self.ratio,
                self._get_maximum_start(body_parts_pos_r["right_nipple"], body_parts_pos_r["right_hip"], edges_r_side) * self.ratio_r_side),
            "Ls5p": self._circle_perimeter(
                abs(body_parts_pos["left_acromion"][0] - body_parts_pos["right_acromion"][0]) * self.ratio),
            "Ls6p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["nose_per"], [body_parts_pos["nose_per"][0],body_parts_pos["nose_per"][1] + 5], edges) * self.ratio),
            "Ls7p": self._circle_perimeter(np.linalg.norm(body_parts_pos["left_ear"] - body_parts_pos["right_ear"]) * self.ratio),

            "Ls0w": self._get_maximum_line(body_parts_pos["left_hip"], body_parts_pos["right_hip"], edges) * self.ratio,
            "Ls1w": self._get_maximum_line(body_parts_pos["left_umbiculus"], body_parts_pos["right_umbiculus"], edges) * self.ratio,
            "Ls2w": self._get_maximum_line(body_parts_pos["left_lowest_front_rib"],
                                           body_parts_pos["right_lowest_front_rib"], edges) * self.ratio,
            "Ls3w": max(self._get_maximum_line(body_parts_pos["left_nipple"], body_parts_pos["right_nipple"], edges) * self.ratio, np.linalg.norm(body_parts_pos["left_nipple"] - body_parts_pos["right_nipple"]) * self.ratio2),
            "Ls4w": np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["right_shoulder"]) * self.ratio,
            "Ls4d": self._get_maximum_start(body_parts_pos_r["right_shoulder"], body_parts_pos_r["right_elbow"], edges_r_side) * self.ratio_r_side,

            # Not needed"La1L": (np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_elbow"])) / 2,
            "La2L": abs(body_parts_pos["left_shoulder"][0] - body_parts_pos["left_elbow"][0]) * self.ratio,
            "La3L": abs(body_parts_pos["left_shoulder"][0] - body_parts_pos["left_maximum_forearm"][0]) * self.ratio,
            "La4L": abs(body_parts_pos["left_shoulder"][0] - body_parts_pos["left_wrist"][0]) * self.ratio,
            "La5L": abs(body_parts_pos["left_wrist"][0] - body_parts_pos["left_base_of_thumb"][0]) * self.ratio,
            "La6L": abs(body_parts_pos["left_wrist"][0] - body_parts_pos["left_knuckle"][0]) * self.ratio,
            "La7L": abs(body_parts_pos["left_wrist"][0] - body_parts_pos["left_nail"][0]) * self.ratio,

            "La0p": self._circle_perimeter(self._get_maximum_start(body_parts_pos["left_shoulder_perimeter_width"], body_parts_pos["left_elbow"], edges)) * self.ratio,
            "La1p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["left_mid_arm"], body_parts_pos["left_elbow"], edges)) * self.ratio,
            "La2p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["left_elbow"], body_parts_pos["left_mid_arm"], edges)) * self.ratio,
            "La3p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["left_maximum_forearm"], body_parts_pos["left_elbow"], edges)) * self.ratio,
            "La4p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["left_wrist"], body_parts_pos["left_elbow"], edges) * self.ratio, self._get_maximum_start(body_parts_pos_up["left_wrist"], body_parts_pos_up["left_elbow"], edges_up) * self.ratio_up),
            "La5p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["left_base_of_thumb"], data[94], edges) * self.ratio, 0),#self._get_maximum_start(data_up[96], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up),
            "La6p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["left_knuckles"], body_parts_pos["left_wrist"], edges) * self.ratio, 0),#self._get_maximum_start(data_up[97], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up),
            "La7p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["left_nails"], body_parts_pos["left_wrist"], edges) * self.ratio, 0),#self._get_maximum_start(data_up[98], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up),

            "La4w": self._get_maximum_start(body_parts_pos["left_wrist"], body_parts_pos["left_elbow"], edges) * self.ratio2,
            "La5w": self._get_maximum_start(body_parts_pos["left_base_of_thumb"], data[94], edges) * self.ratio2,
            "La6w": self._get_maximum_start(body_parts_pos["left_knuckles"], body_parts_pos["left_wrist"], edges) * self.ratio2,
            "La7w": self._get_maximum_start(body_parts_pos["left_nails"], body_parts_pos["left_wrist"], edges) * self.ratio2,

            # Not needed"Lb1L": (np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_elbow"])) / 2,
            "Lb2L": abs(body_parts_pos["right_shoulder"][0] - body_parts_pos["right_elbow"][0]) * self.ratio,
            "Lb3L": abs(body_parts_pos["right_shoulder"][0] - body_parts_pos["right_maximum_forearm"][0]) * self.ratio,
            "Lb4L": abs(body_parts_pos["right_shoulder"][0] - body_parts_pos["right_wrist"][0]) * self.ratio,
            "Lb5L": abs(body_parts_pos["right_wrist"][0] - body_parts_pos["right_base_of_thumb"][0]) * self.ratio,
            "Lb6L": abs(body_parts_pos["right_wrist"][0] - body_parts_pos["right_knuckle"][0]) * self.ratio,
            "Lb7L": abs(body_parts_pos["right_wrist"][0] - body_parts_pos["right_nail"][0]) * self.ratio,

            "Lb0p": self._circle_perimeter(self._get_maximum_start(body_parts_pos["right_shoulder_perimeter_width"], body_parts_pos["right_elbow"], edges)) * self.ratio,
            "Lb1p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["right_mid_arm"], body_parts_pos["right_elbow"], edges)) * self.ratio,
            "Lb2p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["right_elbow"], body_parts_pos["right_mid_arm"], edges)) * self.ratio,
            "Lb3p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["right_maximum_forearm"], body_parts_pos["right_wrist"], edges)) * self.ratio,
            "Lb4p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["right_wrist"], body_parts_pos["right_elbow"], edges) * self.ratio, self._get_maximum_start(body_parts_pos_up["right_wrist"], body_parts_pos_up["right_elbow"], edges_up) * self.ratio_up),
            "Lb5p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["right_base_of_thumb"], data[115], edges) * self.ratio, self._get_maximum_start(data_up[117], body_parts_pos_up["right_wrist"], edges_up) * self.ratio_up),
            "Lb6p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["right_knuckles"], body_parts_pos["right_wrist"], edges) * self.ratio, self._get_maximum_start(data_up[117], body_parts_pos_up["right_wrist"], edges_up) * self.ratio_up),
            "Lb7p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos["right_nails"], body_parts_pos["right_wrist"], edges) * self.ratio, self._get_maximum_start(data_up[119], body_parts_pos_up["right_wrist"], edges_up) * self.ratio_up),

            "Lb4w": self._get_maximum_start(body_parts_pos["right_wrist"], body_parts_pos["right_elbow"], edges) * self.ratio,
            "Lb5w": self._get_maximum_start(body_parts_pos["right_base_of_thumb"], data[115], edges) * self.ratio,
            "Lb6w": self._get_maximum_start(body_parts_pos["right_knuckles"], body_parts_pos["right_wrist"], edges) * self.ratio,
            "Lb7w": self._get_maximum_start(body_parts_pos["right_nails"], body_parts_pos["right_wrist"], edges) * self.ratio,

            "Lj1L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_crotch"]) * self.bottom_ratio2,
            # Not needed"Lj2L": (np.linalg.norm(body_parts_pos["left_hip"] - body_par&ts_pos["left_knee"])) / 2,
            "Lj3L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_knee"]) * self.bottom_ratio2,
            "Lj4L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_maximum_calf"]) * self.bottom_ratio2,
            "Lj5L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_ankle"]) * self.bottom_ratio2,
            "Lj6L": 1,
            # not measured "Lj7L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_arch"]),
            "Lj8L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_ball"]) * self.bottom_ratio2,
            "Lj9L": np.linalg.norm(body_parts_pos_pike["right_ankle"] - body_parts_pos_pike["right_toe_nail"]) * self.ratio_pike,

            # not measured "Lj0p":,
            "Lj1p": self._circle_perimeter(body_parts_pos["left_crotch_width"]),
            "Lj2p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["left_mid_thigh"], body_parts_pos["left_knee"], edges)) * self.bottom_ratio,
            "Lj3p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["left_knee"], body_parts_pos["left_hip"], edges)) * self.bottom_ratio,
            "Lj4p": self._circle_perimeter(self._get_maximum_start(body_parts_pos["left_maximum_calf"], body_parts_pos["left_knee"], edges)) * self.bottom_ratio,
            "Lj5p": self._circle_perimeter(self._get_maximum_start(body_parts_pos["left_ankle"], body_parts_pos["left_knee"], edges)) * self.bottom_ratio,
            "Lj6p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_toe_nail"], edges_pike) * self.ratio_pike, self._get_maximum_start(body_parts_pos_l_pike["right_ankle"], body_parts_pos_l_pike["right_toe_nail"], edges_l_pike) * self.ratio_l_pike),
            "Lj7p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos_pike["right_arch"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike, self._get_maximum_start(body_parts_pos_l_pike["right_arch"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lj8p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike, self._get_maximum_start(body_parts_pos_l_pike["right_ball"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lj9p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike, self._get_maximum_start(body_parts_pos_l_pike["right_toe_nail"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),

            "Lj8w": self._get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,
            "Lj9w": self._get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,

            "Lj6d": self._get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_knee"], edges_pike) * self.ratio_pike,

            "Lk1L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_crotch"]) * self.bottom_ratio2,
            # Not needed"Lk2L": (np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_knee"]) * self.ratio) / 2,
            "Lk3L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_knee"]) * self.bottom_ratio2,
            "Lk4L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_maximum_calf"]) * self.bottom_ratio2,
            "Lk5L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_ankle"]) * self.bottom_ratio2,
            "Lk6L": 1,
            # not measured "Lk7L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_arch"]),
            "Lk8L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_ball"]) * self.bottom_ratio2,
            "Lk9L": np.linalg.norm(body_parts_pos_pike["right_ankle"] - body_parts_pos_pike["right_toe_nail"]) * self.ratio_pike,

            # not measured "Lk0p":,
            "Lk1p": self._circle_perimeter(body_parts_pos["right_crotch_width"]),

            "Lk2p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["right_mid_thigh"], body_parts_pos["right_hip"], edges)) * self.bottom_ratio,
            "Lk3p": self._circle_perimeter(
                self._get_maximum_start(body_parts_pos["right_knee"], body_parts_pos["right_hip"], edges)) * self.bottom_ratio,
            "Lk4p": self._circle_perimeter(self._get_maximum_start(body_parts_pos["right_maximum_calf"], body_parts_pos["right_knee"], edges)) * self.bottom_ratio,
            "Lk5p": self._circle_perimeter(self._get_maximum_start(body_parts_pos["right_ankle"], body_parts_pos["right_knee"], edges)) * self.bottom_ratio,
            "Lk6p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_toe_nail"], edges_pike) * self.ratio_pike, self._get_maximum_start(body_parts_pos_l_pike["right_ankle"], body_parts_pos_l_pike["right_toe_nail"], edges_l_pike) * self.ratio_l_pike),
            "Lk7p": self._stadium_perimeter(
                self._get_maximum_start(body_parts_pos_pike["right_arch"], body_parts_pos_pike["right_ankle"],
                                        edges_pike) * self.ratio_pike,
                self._get_maximum_start(body_parts_pos_l_pike["right_arch"], body_parts_pos_l_pike["right_ankle"],
                                       edges_l_pike) * self.ratio_l_pike),
            "Lk8p": self._stadium_perimeter(self._get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,self._get_maximum_start(body_parts_pos_l_pike["right_ball"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lk9p": self._stadium_perimeter(
                self._get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"],
                                        edges_pike) * self.ratio_pike,
                self._get_maximum_start(body_parts_pos_l_pike["right_toe_nail"], body_parts_pos_l_pike["right_ankle"],
                                       edges_l_pike) * self.ratio_l_pike),

            "Lk8w": self._get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,
            "Lk9w": self._get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,

            "Lk6d": self._get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_knee"], edges_pike) * self.ratio_pike,
        }
        self._verify_keypoints()
        self._create_txt("alexandre.txt")

    def _get_maximum(self, start, end, edges, angle, is_start):
        def pt_from(origin, angle, distance):
            """
            compute the point [x, y] that is 'distance' apart from the origin point
            perpendicular
            """
            x = origin[1] + np.sin(angle) * distance
            y = origin[0] + np.cos(angle) * distance
            return np.array([int(y), int(x)])

        def find_edge(p1, p2, angle_radians):
            distance = 0
            save = []
            while True:
                # as we want the width of the "start", we choose p1
                x, y = pt_from(p1, angle_radians, distance)
                if x < 0 or x >= edges.shape[0] or y < 0 or y >= edges.shape[1]:
                    break
                hit_zone = edges[x, y] == 255
                if np.any(hit_zone):
                    save.append((y, x))
                    break

                distance += 0.01
            return save

        def get_points(start, end):
            p1 = start[0:2]
            p1 = np.array([p1[1], p1[0]])
            p2 = end[0:2]
            p2 = np.array([p2[1], p2[0]])
            return np.array([p1, p2])

        p1, p2 = get_points(start, end)
        vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        angle_radians = np.arctan2(vector[1], vector[0]) - angle
        max1 = find_edge(p1, p2, angle_radians)
        angle_radians = (np.arctan2(vector[1], vector[0]) + angle) * is_start
        max2 = find_edge(p1, p2, angle_radians)
        return np.linalg.norm(np.array(max1) - np.array(max2))

    def _get_maximum_line(self, start, end, edges):
        return self._get_maximum(start, end, edges, 0, -1)

    def _get_maximum_start(self, start, end, edges):
        return self._get_maximum(start, end, edges, np.pi / 2, 1)

    def _get_maximum_point(self, start, end, edges):
        def pt_from(origin, angle, distance):
            """
            compute the point [x, y] that is 'distance' apart from the origin point
            perpendicular
            """
            x = origin[1] + np.sin(angle) * distance
            y = origin[0] + np.cos(angle) * distance
            return np.array([int(y), int(x)])

        def get_max_approx(top_arr, bottom_arr):
            if len(top_arr) != len(bottom_arr):
                print("error not the same nbr of pts")
                return


            vector = np.array(top_arr) - np.array(bottom_arr)
            norms = np.linalg.norm(vector, axis=1)
            max_norm = norms[0]
            save_index = 0
            for i in range(len(top_arr)):
                if norms[i] > max_norm:
                    if norms[i] > norms[0] * 1.5:
                        break
                    max_norm = norms[i]
                    save_index = i
            return save_index

        def get_points(start, end):
            p1 = start[0:2]
            p1 = np.array([p1[1], p1[0]])
            p2 = end[0:2]
            p2 = np.array([p2[1], p2[0]])
            return np.array([p1, p2])

        def vector_angle_plus(p1, p2):
            vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            angle_radians = np.arctan2(vector[1], vector[0]) + np.pi / 2
            return angle_radians

        def vector_angle_minus(p1, p2):
            vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            angle_radians = np.arctan2(vector[1], vector[0]) - np.pi / 2
            return angle_radians

        def get_maximum_range(p1, p2, angle_radians):
            save = []
            # for all the points between p1 and p2
            for point in result:
                distance = 0
                while True:

                    x, y = pt_from(point, angle_radians, distance)
                    if x < 0 or x >= edges.shape[0] or y < 0 or y >= edges.shape[1]:
                        break

                    # Check if we've found an edge pixel
                    # hit_zone = edges[y-1:y+2, x-1:x+2] == 255# 3 x 3
                    hit_zone = edges[x, y] == 255
                    if np.any(hit_zone):
                        save.append((y, x))
                        break

                    distance += 0.01
            return save

        # get the maximums for calf and forearm
        p1, p2 = get_points(start, end)
        # create an array with 100 points between start and end
        x_values = np.linspace(p1[1], p2[1], 100)
        y_values = np.linspace(p1[0], p2[0], 100)
        result = [(y, x) for x, y in zip(x_values, y_values)]
        # set the angle in the direction of the edges
        angle_radians = vector_angle_plus(p1, p2)
        r_side = get_maximum_range(p1, p2, angle_radians)
        # set the angle in the direction of other side
        angle_radians = vector_angle_minus(p1, p2)
        l_side = get_maximum_range(p1, p2, angle_radians)
        # get the index of the max
        index = get_max_approx(r_side, l_side)
        return result[index][::-1]

    def _get_maximum_pit(self, start, edges):
        def pt_from(origin, angle, distance):
            """
            compute the point [x, y] that is 'distance' apart from the origin point
            perpendicular
            """
            x = origin[1] + np.sin(angle) * distance
            y = origin[0] + np.cos(angle) * distance
            return np.array([int(y), int(x)])

        def find_edge(p1, p2, angle_radians):
            distance = 0
            save = []
            while True:
                # as we want the width of the "start", we choose p1
                x, y = pt_from(p1, angle_radians, distance)
                if x < 0 or x >= edges.shape[0] or y < 0 or y >= edges.shape[1]:
                    break

                hit_zone = edges[x, y] == 255
                if np.any(hit_zone):
                    save.append((y, x))
                    break

                distance += 0.01
            return save

        point = start[0:2]
        point = np.array([point[1], point[0]])
        point2 = np.array([point[0] + 5, point[1]])
        vector = np.array([point2[0] - point[0], point2[1] - point[1]])
        angle_radians = np.arctan2(vector[1], vector[0])
        max_save = find_edge(point, point2, angle_radians)
        angle_radians_left = angle_radians
        angle_radians_right = angle_radians
        max_save_right = max_save
        max_save_left = max_save

        while True:
            angle_radians_left += 0.01
            max_last_right = find_edge(point, point2, angle_radians_left)
            if (np.linalg.norm(max_last_right) < np.linalg.norm(max_save_right)):
                break
            max_save_right = max_last_right
        while True:
            angle_radians_right -= 0.01
            max_last_left = find_edge(point, point2, angle_radians_right)
            if (np.linalg.norm(max_last_left) < np.linalg.norm(max_save_left)):
                break
            max_save_left = max_last_left
        delta = np.linalg.norm(abs(np.array(max_last_left) - np.array(max_last_right)))
        distance = np.linalg.norm(max(max_save_left, max_last_right))
        point = max(max_save_left, max_last_right)
        #if distance * 0.05 < delta:
        #    return 0, 0
        return point[0], int(distance)

    def _crop(self, image, position_1, position_2):
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
    def _get_crotch_right_left(self, edges, data):


        # crop the image to see the right hip to the left knee
        crotch_zone = self._crop(edges, data[12], data[13])
        # now the cropped image only has the crotch as an edge so we can get it like the head
        crotch_approx_crop = np.where(crotch_zone != 0)[1][0], np.where(crotch_zone != 0)[0][1]
        crotch_approx_right = np.array([data[12][0], data[12][1] + crotch_approx_crop[1]])
        crotch_approx_left = np.array([data[11][0], data[11][1] + crotch_approx_crop[1]])
        return crotch_approx_right, crotch_approx_left

    def _get_mid_thigh_right_left(self, data, r_crotch, l_crotch):
        mid_thigh_right = (data[14][0:2] + r_crotch) / 2
        mid_thigh_left = (data[13][0:2] + l_crotch) / 2
        return mid_thigh_right, mid_thigh_left

    def _find_acromion_right(self, edges, data, height):
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
        cropped_img = self._crop(edges, r_ear, r_shoulder)
        if height:
            r_ear, r_shoulder = data[0], data[6]
            cropped_img = self._crop(edges, r_ear, r_shoulder)
            cropped_img[:int(len(cropped_img) / 2), :] = 0
            cropped_img[:, int(len(cropped_img[0]) / 2.5):] = 0
            reversed_image_array = np.fliplr(cropped_img)
        else:
            cropped_img[:int(len(cropped_img) / 2), :] = 0
            #cropped_img[:, :int(len(cropped_img[0]) / 2)] = 0
            reversed_image_array = np.fliplr(cropped_img)
        acromion = np.array([r_ear[0] - (np.where(reversed_image_array == 255)[1][0]), r_ear[1] + np.where(reversed_image_array == 255)[0][0]])
        return acromion

    def _find_acromion_left(self, edges, data, height):
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
            cropped_img = self._crop(edges, l_ear, l_shoulder)
            cropped_img[:int(len(cropped_img) / 2), :] = 0
            cropped_img[:int(len(cropped_img[0]) / 1.5), :] = 0
            reversed_image_array = np.fliplr(cropped_img)
            acromion = np.array([l_ear[0] + np.where(reversed_image_array == 255)[1][0], l_ear[1] + np.where(reversed_image_array == 255)[0][0]])
        else:
            l_ear, l_shoulder = data[0], data[5]
            cropped_img = self._crop(edges, l_ear, l_shoulder)
            cropped_img[:int(len(cropped_img) / 2), :] = 0
            #cropped_img[:, :int(len(cropped_img[0]) / 2)] = 0
            acromion = np.array(
                [l_ear[0] + np.where(cropped_img == 255)[1][0], l_ear[1] + np.where(cropped_img == 255)[0][0]])
        return acromion

    def _find_top_of_head(self, edges):
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

    def _stadium_perimeter(self, width, depth):
        radius = depth / 2
        a = width - depth
        perimeter = 2 * (np.pi * radius + a)
        return perimeter

    def _circle_perimeter(self, diag):
        r = diag / 2
        return 2 * np.pi * r

    def _resize(self, im):
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

    def _canny_edges(self, im, image):
        grayscale_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        edged = cv.Canny(grayscale_image, 10, 30)

        # define a (3, 3) structuring element
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

        # apply the dilation operation to the edged image
        dilate = cv.dilate(edged, kernel, iterations=2)

        # find the contours in the dilated image
        contours, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        edges = np.zeros(image.shape)
        # draw the contours on a copy of the original image
        cv.drawContours(edges, contours, -1, (0, 255, 0), 2)
        return edges

    def _create_resize_remove_im(self, impath):
        pil_im = PIL.Image.open(impath).convert("RGB")
        pil_im = self._resize(pil_im)
        image = np.asarray(pil_im)
        im = remove(image)
        return pil_im, image, im
    def _pil_resize_remove_im(self, undistorted_image):
        pil_im = Image.fromarray(undistorted_image.astype('uint8'), 'RGB')
        pil_im = self._resize(pil_im)
        image = np.asarray(pil_im)
        im = remove(image)
        return pil_im, image, im

    def _undistortion(self, chessboard_images_path, img_with_chessboard_pathc):
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
        calibrated_image = np.asarray(self._resize(PIL_image))
        # get the corners for the two chessboards
        undistorted_image2, corners = find_chessboard(calibrated_image.copy(), 1)
        undistorted_image_drawed, corners2 = find_chessboard(undistorted_image2, 0)
        H, _ = cv.findHomography(corners2, corners)
        undistorted_image = undistor(calibrated_image, H)
        return undistorted_image


    def _get_ratio(self, img, top, bot):
        if top:
            img = self._crop(img, [0,0], [img.shape[0], img.shape[0] / 2])
        if bot:
            img = self._crop(img, [0, img.shape[0]], [img.shape[1], img.shape[0] / 2])

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

        ratio =  np.linalg.norm(chessboard_contour[0] - chessboard_contour[1])
        ratio2 =  np.linalg.norm(chessboard_contour[0] - chessboard_contour[3])
        print(ratio)
        print(ratio2)
        ratio = 9.8/ratio
        ratio2 = 9.8/ratio2
        return ratio, ratio2
    def _create_txt(self, file_name):
        with open(file_name, "w") as file_object:
            for key, value in self.keypoints.items():
                if key[-1].isalpha():
                    file_object.writelines("{} : {:.1f}\n".format(key, float(value)))
    def _verify_keypoints(self):
        def loop(keypoint_perim, keypoint_width):
            if keypoint_perim / keypoint_width <= 2:
                return (keypoint_perim / 2) - 0.1
            if keypoint_perim / keypoint_width >= np.pi:
                return (keypoint_perim / np.pi) + 0.1
            return keypoint_width

        self.keypoints["Ls0w"] = loop(self.keypoints["Ls0p"], self.keypoints["Ls0w"])
        self.keypoints["Ls1w"] = loop(self.keypoints["Ls1p"], self.keypoints["Ls1w"])
        self.keypoints["Ls2w"] = loop(self.keypoints["Ls2p"], self.keypoints["Ls2w"])
        self.keypoints["Ls3w"] = loop(self.keypoints["Ls3p"], self.keypoints["Ls3w"])

        self.keypoints["La4w"] = loop(self.keypoints["La4p"], self.keypoints["La4w"])
        self.keypoints["La5w"] = loop(self.keypoints["La5p"], self.keypoints["La5w"])
        self.keypoints["La6w"] = loop(self.keypoints["La6p"], self.keypoints["La6w"])
        self.keypoints["La7w"] = loop(self.keypoints["La7p"], self.keypoints["La7w"])

        self.keypoints["Lb4w"] = loop(self.keypoints["Lb4p"], self.keypoints["Lb4w"])
        self.keypoints["Lb5w"] = loop(self.keypoints["Lb5p"], self.keypoints["Lb5w"])
        self.keypoints["Lb6w"] = loop(self.keypoints["Lb6p"], self.keypoints["Lb6w"])
        self.keypoints["Lb7w"] = loop(self.keypoints["Lb7p"], self.keypoints["Lb7w"])

        self.keypoints["Lj6d"] = loop(self.keypoints["Lj6p"], self.keypoints["Lj6d"])
        self.keypoints["Lj8w"] = loop(self.keypoints["Lj8p"], self.keypoints["Lj8w"])
        self.keypoints["Lj9w"] = loop(self.keypoints["Lj9p"], self.keypoints["Lj9w"])

        self.keypoints["Lk6d"] = loop(self.keypoints["Lk6p"], self.keypoints["Lk6d"])
        self.keypoints["Lk8w"] = loop(self.keypoints["Lk8p"], self.keypoints["Lk8w"])
        self.keypoints["Lk9w"] = loop(self.keypoints["Lk9p"], self.keypoints["Lk9w"])

if __name__ == "__main__":
    pass
