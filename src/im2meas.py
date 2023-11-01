import openpifpaf

import matplotlib.pyplot as plt

from src.utils.get_maximum import *
from src.utils.find_body_parts import *
from src.utils.image_config import *
from src.utils.perimeter_calculator import *


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
        #undistorted_image = _undistortion('img/martin/chessboards/*', "img/martin/mar_front_t.jpg")
        undistorted_image = undistortion('img/william/chessboards/*', "img/william/william_front_t.jpg")
        #undistorted_image = _undistortion('img/chessboards/*', "img/al_front_t.jpg")

        pil_im, image, im = pil_resize_remove_im(undistorted_image)

        edges = canny_edges(im, image)
        predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k30-wholebody")
        predictor_simple = openpifpaf.Predictor(checkpoint="shufflenetv2k30")
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
        # You can find the index here:
        # https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
        # as "predictions" is an array the index starts at 0 and not at 1 like in the github
        data = predictions[0].data[:, 0:2]

        edges = better_edges(edges, data)
        image_chessboard = image.copy()
        img = Image.fromarray(image_chessboard)
        img.save("front_t.jpg")

        self.ratio, self.ratio2 = get_ratio(image_chessboard, 0, 0)
        self.bottom_ratio, self.bottom_ratio2 = get_ratio(image_chessboard, 0, 0)

        print("self.ratio", self.ratio)
        print("self.ratio2", self.ratio2)
        print("self.bottom_ratio", self.bottom_ratio)
        print("self.bottom_ratio2", self.bottom_ratio2)


        # right side
        #undistorted_r_image = _undistortion('img/chessboards/*', "img/al_r_side.jpg")
        #pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/martin/mar_r_side.jpg")
        pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/william/william_r_side.jpg")
        #pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/al_r_side.jpg")


        edges_r_side = canny_edges(im_r_side, image_r_side)
        predictions2, gt_anns2, image_meta2 = predictor_simple.pil_image(pil_r_side_im)
        data_r_side = predictions2[0].data[:, 0:2]
        image_r_side_chessboard = image_r_side.copy()
        self.ratio_r_side, self.ratio_r_side2 = get_ratio(image_r_side_chessboard, 0,0)
        # front T pose but with the hand to the top
        #undistorted_up_image = _undistortion('img/william/chessboards/*', "img/william/wil_front_t_up.jpg")
        #pil_up_im, image_up, im_up = create_resize_remove_im("img/martin/mar_front_t_up.jpg")
        pil_up_im, image_up, im_up = create_resize_remove_im("img/william/william_front_t_up.jpg")
        #pil_up_im, image_up, im_up = create_resize_remove_im("img/al_front_t_up.jpg")

        edges_up = canny_edges(im_up, image_up)
        predictions4, gt_anns4, image_meta4 = predictor.pil_image(pil_up_im)
        data_up = predictions4[0].data[:, 0:2]
        image_up_chessboard = image_up.copy()


        self.ratio_up, self.ratio_up2 = get_ratio(image_up_chessboard, 0, 0)
        # front pike
        #undistorted_pike_image = _undistortion('img/chessboards/*', "img/al_front_pike.jpg")
        #pil_pike_im, image_pike, im_pike = create_resize_remove_im("img/martin/mar_front_pike.jpg")
        pil_pike_im, image_pike, im_pike = create_resize_remove_im("img/william/william_front_pike.jpg")
        #pil_pike_im, image_pike, im_pike = create_resize_remove_im("img/al_front_pike.jpg")


        edges_pike = canny_edges(im_pike, image_pike)
        predictions5, gt_anns5, image_meta5 = predictor.pil_image(pil_pike_im)
        data_pike = predictions5[0].data[:, 0:2]
        image_pike_chessboard = image_pike.copy()
        self.ratio_pike, self.ratio_pike2 = get_ratio(image_pike_chessboard, 0,0)
        #self.ratio_pike, self.ratio_pike2 = get_ratio(image_pike, 0, 1)
        # right side pike
        #undistorted_r_pike_image = _undistortion('img/chessboards/*', "img/al_r_pike.jpg")
        #pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/martin/mar_r_pike.jpg")
        pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/william/william_r_pike.jpg")
        #pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/al_r_pike.jpg")

        edges_l_pike = canny_edges(im_l_pike, image_l_pike)
        predictions6, gt_anns6, image_meta6 = predictor.pil_image(pil_l_pike_im)
        data_l_pike = predictions6[0].data[:, 0:2]
        image_r_pike_chessboard = image_l_pike.copy()
        self.ratio_l_pike, self.ratio_l_pike2 = get_ratio(image_r_pike_chessboard, 0, 0)
        #self.ratio_l_pike, self.ratio_l_pike2 = get_ratio(image_l_pike, 0, 1)
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
        body_parts_pos["left_shoulder_perimeter_width"] = get_maximum_point(data[7], data[5], edges)
        body_parts_pos["right_shoulder_perimeter_width"] = get_maximum_point(data[8], data[6], edges)
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
        body_parts_pos["left_acromion"] = find_acromion_left(edges, data, 0)
        body_parts_pos["right_acromion"] = find_acromion_right(edges, data[0], data[6], 0)
        body_parts_pos["left_acromion_height"] = find_acromion_left(edges, data, 1)
        body_parts_pos["right_acromion_height"] = find_acromion_right(edges, data[0], data[6], 1)
        body_parts_pos["top_of_head"] = find_top_of_head(edges)
        body_parts_pos["left_mid_elbow_wrist"] = (data[9] + data[7]) / 2
        body_parts_pos["right_mid_elbow_wrist"] = (data[10] + data[8]) / 2
        body_parts_pos["right_maximum_forearm"] = get_maximum_point(body_parts_pos["right_mid_elbow_wrist"], data[8], edges)
        body_parts_pos["left_maximum_forearm"] = get_maximum_point(body_parts_pos["left_mid_elbow_wrist"], data[7], edges)
        body_parts_pos["right_maximum_calf"] = get_maximum_point(data[16], data[14], edges)
        body_parts_pos["left_maximum_calf"] = get_maximum_point(data[15], data[13], edges)
        body_parts_pos["right_crotch"], body_parts_pos["left_crotch"] = get_crotch_right_left(edges, data)
        body_parts_pos["right_mid_thigh"], body_parts_pos["left_mid_thigh"] = get_mid_thigh_right_left(data,
                                                                                                             body_parts_pos[
                                                                                                                 "right_crotch"],
                                                                                                             body_parts_pos[
                                                                                                                 "left_crotch"])
        body_parts_pos["left_crotch_width"] = get_maximum_start(body_parts_pos["left_crotch"], body_parts_pos["left_knee"], edges) * self.bottom_ratio
        body_parts_pos["right_crotch_width"] = get_maximum_start(body_parts_pos["right_crotch"], body_parts_pos["right_knee"], edges) * self.bottom_ratio
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
        body_parts_pos_up["left_acromion"] = find_acromion_left(edges_up, data_up, 0)
        body_parts_pos_up["right_acromion"] = find_acromion_right(edges_up, data_up[0], data_up[6], 0)
        body_parts_pos_up["top_of_head"] = find_top_of_head(edges_up)
        body_parts_pos_up["right_maximum_forearm"] = get_maximum_point(data_up[10], data_up[8], edges_up)
        body_parts_pos_up["left_maximum_forearm"] = get_maximum_point(data_up[9], data_up[7], edges_up)
        body_parts_pos_up["right_maximum_calf"] = get_maximum_point(data_up[16], data_up[14], edges_up)
        body_parts_pos_up["left_maximum_calf"] = get_maximum_point(data_up[15], data_up[13], edges_up)
        # right side
        right_lowest_front_rib_approx = np.array([data_r_side[12][0],(data_r_side[6][1] + data_r_side[12][1]) / 2])
        body_parts_pos_r["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos_r["right_nipple"] = np.array([right_lowest_front_rib_approx[0],(right_lowest_front_rib_approx[1] + data_r_side[6][1]) / 2])
        body_parts_pos_r["right_umbiculus"] = (
                                                      (right_lowest_front_rib_approx * 3) + (data_r_side[12] * 2)
                                              ) / 5
        # front pike
        point, dist = get_maximum_pit(data_pike[16], edges_pike)
        body_parts_pos_pike["right_toe_nail"] = np.array([point[0], point[1] - 2 / self.ratio_pike])
        body_parts_pos_pike["right_ball"] = get_maximum_point(data_pike[16], body_parts_pos_pike["right_toe_nail"], edges_pike)
        body_parts_pos_pike["right_arch"] = (data_pike[16] + body_parts_pos_pike["right_ball"]) / 2
        # left side pike
        body_parts_pos_l_pike["right_toe_nail"], dist = get_maximum_pit(data_l_pike[16], edges_l_pike)
        body_parts_pos_l_pike["right_heel"] = get_maximum_point(data_l_pike[16], body_parts_pos_l_pike["right_toe_nail"], edges_l_pike)
        body_parts_pos_l_pike["right_ball"] = np.array([body_parts_pos_l_pike["right_toe_nail"][0], body_parts_pos_l_pike["right_toe_nail"][1] - 3 / self.ratio_l_pike])
        body_parts_pos_l_pike["right_arch"] = (body_parts_pos_l_pike["right_heel"] + body_parts_pos_l_pike["right_ball"]) / 2

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
            "Ls1L": abs(body_parts_pos["right_umbiculus"][1] - body_parts_pos["right_hip"][1]) * self.ratio2,
            "Ls2L": abs(
                body_parts_pos["right_lowest_front_rib"][1]
                - body_parts_pos["right_hip"][1]
            ) * self.ratio2,
            "Ls3L": abs(
                body_parts_pos["right_nipple"][1] - body_parts_pos["right_hip"][1]
            ) * self.ratio2,
            "Ls4L": abs(
                body_parts_pos["right_shoulder"][1] - body_parts_pos["right_hip"][1]
            ) * self.ratio2,
            "Ls5L": abs(body_parts_pos["right_acromion_height"][1] - body_parts_pos["right_hip"][1]) * self.ratio2,
            "Ls6L": abs(body_parts_pos["left_acromion_height"][1] - body_parts_pos["nose"][1]) * self.ratio2,
            "Ls7L": np.linalg.norm(body_parts_pos["left_acromion_height"] - body_parts_pos["left_ear"]) * self.ratio2,
            "Ls8L": abs(
                body_parts_pos["left_acromion_height"][1] - body_parts_pos["top_of_head"][1]
            ) * self.ratio2,

            "Ls0p": stadium_perimeter(
                get_maximum_start(body_parts_pos["right_hip"], body_parts_pos["right_knee"], edges) * self.ratio,
                get_maximum_start(body_parts_pos_r["right_hip"], body_parts_pos_r["right_knee"],
                                        edges_r_side) * self.ratio_r_side),
            "Ls1p": stadium_perimeter(get_maximum_line(body_parts_pos["left_umbiculus"], body_parts_pos["right_umbiculus"], edges) * self.ratio,get_maximum_start(body_parts_pos_r["right_umbiculus"], body_parts_pos_r["right_knee"],edges_r_side) * self.ratio_r_side),
            "Ls2p": stadium_perimeter(max(get_maximum_line(body_parts_pos["right_lowest_front_rib"],
                                                                   body_parts_pos["left_lowest_front_rib"], edges), np.linalg.norm(body_parts_pos["right_lowest_front_rib"] -
                                                                   body_parts_pos["left_lowest_front_rib"])) * self.ratio,
                                            get_maximum_start(body_parts_pos_r["right_lowest_front_rib"],
                                                                    body_parts_pos_r["right_hip"], edges_r_side) * self.ratio_r_side),
            "Ls3p": stadium_perimeter(
                get_maximum_line(body_parts_pos["right_nipple"], body_parts_pos["left_nipple"], edges) * self.ratio,
                get_maximum_start(body_parts_pos_r["right_nipple"], body_parts_pos_r["right_hip"], edges_r_side) * self.ratio_r_side),
            "Ls5p": circle_perimeter(
                abs(body_parts_pos["left_acromion"][0] - body_parts_pos["right_acromion"][0]) * self.ratio),
            "Ls6p": circle_perimeter(
                get_maximum_start(body_parts_pos["nose_per"], [body_parts_pos["nose_per"][0],body_parts_pos["nose_per"][1] + 5], edges) * self.ratio),
            "Ls7p": circle_perimeter(np.linalg.norm(body_parts_pos["left_ear"] - body_parts_pos["right_ear"]) * self.ratio),

            "Ls0w": get_maximum_line(body_parts_pos["left_hip"], body_parts_pos["right_hip"], edges) * self.ratio,
            "Ls1w": get_maximum_line(body_parts_pos["left_umbiculus"], body_parts_pos["right_umbiculus"], edges) * self.ratio,
            "Ls2w": get_maximum_line(body_parts_pos["left_lowest_front_rib"],
                                           body_parts_pos["right_lowest_front_rib"], edges) * self.ratio,
            "Ls3w": max(get_maximum_line(body_parts_pos["left_nipple"], body_parts_pos["right_nipple"], edges) * self.ratio, np.linalg.norm(body_parts_pos["left_nipple"] - body_parts_pos["right_nipple"]) * self.ratio2),
            "Ls4w": np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["right_shoulder"]) * self.ratio,
            "Ls4d": get_maximum_start(body_parts_pos_r["right_shoulder"], body_parts_pos_r["right_elbow"], edges_r_side) * self.ratio_r_side,

            # Not needed"La1L": (np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_elbow"])) / 2,
            "La2L": abs(body_parts_pos["left_shoulder"][0] - body_parts_pos["left_elbow"][0]) * self.ratio,
            "La3L": abs(body_parts_pos["left_shoulder"][0] - body_parts_pos["left_maximum_forearm"][0]) * self.ratio,
            "La4L": abs(body_parts_pos["left_shoulder"][0] - body_parts_pos["left_wrist"][0]) * self.ratio,
            "La5L": abs(body_parts_pos["left_wrist"][0] - body_parts_pos["left_base_of_thumb"][0]) * self.ratio,
            "La6L": abs(body_parts_pos["left_wrist"][0] - body_parts_pos["left_knuckle"][0]) * self.ratio,
            "La7L": abs(body_parts_pos["left_wrist"][0] - body_parts_pos["left_nail"][0]) * self.ratio,

            "La0p": circle_perimeter(get_maximum_start(body_parts_pos["left_shoulder_perimeter_width"], body_parts_pos["left_elbow"], edges)) * self.ratio,
            "La1p": circle_perimeter(
                get_maximum_start(body_parts_pos["left_mid_arm"], body_parts_pos["left_elbow"], edges)) * self.ratio,
            "La2p": circle_perimeter(
                get_maximum_start(body_parts_pos["left_elbow"], body_parts_pos["left_mid_arm"], edges)) * self.ratio,
            "La3p": circle_perimeter(
                get_maximum_start(body_parts_pos["left_maximum_forearm"], body_parts_pos["left_elbow"], edges)) * self.ratio,
            "La4p": stadium_perimeter(get_maximum_start(body_parts_pos["left_wrist"], body_parts_pos["left_elbow"], edges) * self.ratio, get_maximum_start(body_parts_pos_up["left_wrist"], body_parts_pos_up["left_elbow"], edges_up) * self.ratio_up),
            "La5p": stadium_perimeter(get_maximum_start(body_parts_pos["left_base_of_thumb"], data[94], edges) * self.ratio, 0),#self._get_maximum_start(data_up[96], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up),
            "La6p": stadium_perimeter(get_maximum_start(body_parts_pos["left_knuckles"], body_parts_pos["left_wrist"], edges) * self.ratio, 0),#self._get_maximum_start(data_up[97], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up),
            "La7p": stadium_perimeter(get_maximum_start(body_parts_pos["left_nails"], body_parts_pos["left_wrist"], edges) * self.ratio, 0),#self._get_maximum_start(data_up[98], body_parts_pos_up["left_wrist"], edges_up) * self.ratio_up),

            "La4w": get_maximum_start(body_parts_pos["left_wrist"], body_parts_pos["left_elbow"], edges) * self.ratio,
            "La5w": get_maximum_start(body_parts_pos["left_base_of_thumb"], data[94], edges) * self.ratio,
            "La6w": get_maximum_start(body_parts_pos["left_knuckles"], body_parts_pos["left_wrist"], edges) * self.ratio,
            "La7w": get_maximum_start(body_parts_pos["left_nails"], body_parts_pos["left_wrist"], edges) * self.ratio,

            # Not needed"Lb1L": (np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_elbow"])) / 2,
            "Lb2L": abs(body_parts_pos["right_shoulder"][0] - body_parts_pos["right_elbow"][0]) * self.ratio,
            "Lb3L": abs(body_parts_pos["right_shoulder"][0] - body_parts_pos["right_maximum_forearm"][0]) * self.ratio,
            "Lb4L": abs(body_parts_pos["right_shoulder"][0] - body_parts_pos["right_wrist"][0]) * self.ratio,
            "Lb5L": abs(body_parts_pos["right_wrist"][0] - body_parts_pos["right_base_of_thumb"][0]) * self.ratio,
            "Lb6L": abs(body_parts_pos["right_wrist"][0] - body_parts_pos["right_knuckle"][0]) * self.ratio,
            "Lb7L": abs(body_parts_pos["right_wrist"][0] - body_parts_pos["right_nail"][0]) * self.ratio,

            "Lb0p": circle_perimeter(get_maximum_start(body_parts_pos["right_shoulder_perimeter_width"], body_parts_pos["right_elbow"], edges)) * self.ratio,
            "Lb1p": circle_perimeter(
                get_maximum_start(body_parts_pos["right_mid_arm"], body_parts_pos["right_elbow"], edges)) * self.ratio,
            "Lb2p": circle_perimeter(
                get_maximum_start(body_parts_pos["right_elbow"], body_parts_pos["right_mid_arm"], edges)) * self.ratio,
            "Lb3p": circle_perimeter(
                get_maximum_start(body_parts_pos["right_maximum_forearm"], body_parts_pos["right_wrist"], edges)) * self.ratio,
            "Lb4p": stadium_perimeter(get_maximum_start(body_parts_pos["right_wrist"], body_parts_pos["right_elbow"], edges) * self.ratio, get_maximum_start(body_parts_pos_up["right_wrist"], body_parts_pos_up["right_elbow"], edges_up) * self.ratio_up),
            "Lb5p": stadium_perimeter(get_maximum_start(body_parts_pos["right_base_of_thumb"], data[115], edges) * self.ratio,0),
            "Lb6p": stadium_perimeter(get_maximum_start(body_parts_pos["right_knuckles"], body_parts_pos["right_wrist"], edges) * self.ratio, 0),
            "Lb7p": stadium_perimeter(get_maximum_start(body_parts_pos["right_nails"], body_parts_pos["right_wrist"], edges) * self.ratio, 0),

            "Lb4w": get_maximum_start(body_parts_pos["right_wrist"], body_parts_pos["right_elbow"], edges) * self.ratio,
            "Lb5w": get_maximum_start(body_parts_pos["right_base_of_thumb"], data[115], edges) * self.ratio,
            "Lb6w": get_maximum_start(body_parts_pos["right_knuckles"], body_parts_pos["right_wrist"], edges) * self.ratio,
            "Lb7w": get_maximum_start(body_parts_pos["right_nails"], body_parts_pos["right_wrist"], edges) * self.ratio,

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
            "Lj1p": circle_perimeter(body_parts_pos["left_crotch_width"]),
            "Lj2p": circle_perimeter(
                get_maximum_start(body_parts_pos["left_mid_thigh"], body_parts_pos["left_knee"], edges)) * self.bottom_ratio,
            "Lj3p": circle_perimeter(
                get_maximum_start(body_parts_pos["left_knee"], body_parts_pos["left_hip"], edges)) * self.bottom_ratio,
            "Lj4p": circle_perimeter(get_maximum_start(body_parts_pos["left_maximum_calf"], body_parts_pos["left_knee"], edges)) * self.bottom_ratio,
            "Lj5p": circle_perimeter(get_maximum_start(body_parts_pos["left_ankle"], body_parts_pos["left_knee"], edges)) * self.bottom_ratio,
            "Lj6p": stadium_perimeter(get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_toe_nail"], edges_pike) * self.ratio_pike, get_maximum_start(body_parts_pos_l_pike["right_ankle"], body_parts_pos_l_pike["right_toe_nail"], edges_l_pike) * self.ratio_l_pike),
            "Lj7p": stadium_perimeter(get_maximum_start(body_parts_pos_pike["right_arch"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike, get_maximum_start(body_parts_pos_l_pike["right_arch"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lj8p": stadium_perimeter(get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike, get_maximum_start(body_parts_pos_l_pike["right_ball"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lj9p": stadium_perimeter(get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike, get_maximum_start(body_parts_pos_l_pike["right_toe_nail"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),

            "Lj8w": get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,
            "Lj9w": get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,

            "Lj6d": get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_knee"], edges_pike) * self.ratio_pike,

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
            "Lk1p": circle_perimeter(body_parts_pos["right_crotch_width"]),

            "Lk2p": circle_perimeter(
                get_maximum_start(body_parts_pos["right_mid_thigh"], body_parts_pos["right_hip"], edges)) * self.bottom_ratio,
            "Lk3p": circle_perimeter(
                get_maximum_start(body_parts_pos["right_knee"], body_parts_pos["right_hip"], edges)) * self.bottom_ratio,
            "Lk4p": circle_perimeter(get_maximum_start(body_parts_pos["right_maximum_calf"], body_parts_pos["right_knee"], edges)) * self.bottom_ratio,
            "Lk5p": circle_perimeter(get_maximum_start(body_parts_pos["right_ankle"], body_parts_pos["right_knee"], edges)) * self.bottom_ratio,
            "Lk6p": stadium_perimeter(get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_toe_nail"], edges_pike) * self.ratio_pike, get_maximum_start(body_parts_pos_l_pike["right_ankle"], body_parts_pos_l_pike["right_toe_nail"], edges_l_pike) * self.ratio_l_pike),
            "Lk7p": stadium_perimeter(
                get_maximum_start(body_parts_pos_pike["right_arch"], body_parts_pos_pike["right_ankle"],
                                        edges_pike) * self.ratio_pike,
                get_maximum_start(body_parts_pos_l_pike["right_arch"], body_parts_pos_l_pike["right_ankle"],
                                       edges_l_pike) * self.ratio_l_pike),
            "Lk8p": stadium_perimeter(get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,get_maximum_start(body_parts_pos_l_pike["right_ball"], body_parts_pos_l_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lk9p": stadium_perimeter(
                get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"],
                                        edges_pike) * self.ratio_pike,
                get_maximum_start(body_parts_pos_l_pike["right_toe_nail"], body_parts_pos_l_pike["right_ankle"],
                                       edges_l_pike) * self.ratio_l_pike),

            "Lk8w": get_maximum_start(body_parts_pos_pike["right_ball"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,
            "Lk9w": get_maximum_start(body_parts_pos_pike["right_toe_nail"], body_parts_pos_pike["right_ankle"], edges_pike) * self.ratio_pike,

            "Lk6d": get_maximum_start(body_parts_pos_pike["right_ankle"], body_parts_pos_pike["right_knee"], edges_pike) * self.ratio_pike,
        }
        self._verify_keypoints()
        self._create_txt("alexandre.txt")



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
