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
        #undistorted_image = undistortion('img/martin/chessboards/*', "img/martin/mar_front_t.jpg")
        #undistorted_image = undistortion('img/william/chessboards/*', "img/william/william_front_t.jpg")
        undistorted_image = undistortion('img/chessboards/*', "img/al_front_t.jpg")
        #undistorted_image = undistortion('img/chessboards/*', "img/kael/kael_front_t.jpg")

        pil_im, image, im = pil_resize_remove_im(undistorted_image)
        edges = thresh(im, image)
        edges_short = canny_edges(im, image)
        predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k30-wholebody")
        predictor_simple = openpifpaf.Predictor(checkpoint="shufflenetv2k30")
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
        # You can find the index here:
        # https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
        # as "predictions" is an array the index starts at 0 and not at 1 like in the github
        data = predictions[0].data[:, 0:2]

        edges_short = better_edges(edges_short, data)
        image_chessboard = image.copy()
        img = Image.fromarray(image_chessboard)
        img.save("front_t.jpg")

        self.ratio, self.ratio2 = get_ratio(image_chessboard, 1, 0)
        self.bottom_ratio, self.bottom_ratio2 = get_ratio(image_chessboard, 0, 1)
        #self.bottom_ratio, self.bottom_ratio2 = self.ratio, self.ratio2

        print("self.ratio", self.ratio)
        print("self.ratio2", self.ratio2)
        print("self.bottom_ratio", self.bottom_ratio)
        print("self.bottom_ratio2", self.bottom_ratio2)


        # right side
        #undistorted_r_image = _undistortion('img/chessboards/*', "img/al_r_side.jpg")
        #pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/martin/mar_r_side.jpg")
        #pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/william/william_r_side.jpg")
        pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/al_r_side2_undist.png")
        #pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im("img/kael/kael_r_side.jpg")



        edges_r_side = thresh(im_r_side, image_r_side)
        predictions2, gt_anns2, image_meta2 = predictor_simple.pil_image(pil_r_side_im)
        data_r_side = predictions2[0].data[:, 0:2]
        image_r_side_chessboard = image_r_side.copy()
        self.ratio_r_side, self.ratio_r_side2 = get_ratio(image_r_side_chessboard, 0,0)
        # front T pose but with the hand to the top
        #undistorted_up_image = _undistortion('img/william/chessboards/*', "img/william/wil_front_t_up.jpg")
        #pil_up_im, image_up, im_up = create_resize_remove_im("img/martin/mar_front_t_up.jpg")
        #pil_up_im, image_up, im_up = create_resize_remove_im("img/william/william_front_t_up.jpg")
        pil_up_im, image_up, im_up = create_resize_remove_im("img/al_front_t_up.jpg")
        #pil_up_im, image_up, im_up = create_resize_remove_im("img/kael/kael_front_t_up.jpg")


        edges_up = thresh(im_up, image_up)
        predictions4, gt_anns4, image_meta4 = predictor.pil_image(pil_up_im)
        data_up = predictions4[0].data[:, 0:2]
        image_up_chessboard = image_up.copy()


        self.ratio_up, self.ratio_up2 = get_ratio(image_up_chessboard, 0, 1)
        # front pike
        #undistorted_pike_image = _undistortion('img/chessboards/*', "img/al_front_pike.jpg")
        #pil_pike_im, image_pike, im_pike = create_resize_remove_im("img/martin/mar_front_pike.jpg")
        #pil_pike_im, image_pike, im_pike = create_resize_remove_im("img/william/william_front_pike.jpg")
        pil_pike_im, image_pike, im_pike = create_resize_remove_im("img/al_front_pike_undist.png")



        edges_pike = thresh(im_pike, image_pike)
        predictions5, gt_anns5, image_meta5 = predictor.pil_image(pil_pike_im)
        data_pike = predictions5[0].data[:, 0:2]
        image_pike_chessboard = image_pike.copy()
        self.ratio_pike, self.ratio_pike2 = get_ratio(image_pike_chessboard, 0,0)
        #self.ratio_pike, self.ratio_pike2 = get_ratio(image_pike, 0, 1)
        # right side pike
        #undistorted_r_pike_image = _undistortion('img/chessboards/*', "img/al_r_pike.jpg")
        #pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/martin/mar_r_pike.jpg")
        #pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/william/william_r_pike.jpg")
        pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/al_r_pike_undist.png")
        #pil_l_pike_im, image_l_pike, im_l_pike = create_resize_remove_im("img/kael/kael_r_pike.jpg")


        edges_l_pike = canny_edges(im_l_pike, image_l_pike)
        predictions6, gt_anns6, image_meta6 = predictor.pil_image(pil_l_pike_im)
        data_l_pike = predictions6[0].data[:, 0:2]
        image_r_pike_chessboard = image_l_pike.copy()
        self.ratio_l_pike, self.ratio_l_pike2 = get_ratio(image_r_pike_chessboard, 0, 1)
        #self.ratio_l_pike, self.ratio_l_pike2 = get_ratio(image_l_pike, 0, 1)
        # front
        body_parts_index = {
            "nose": 0,
            "nose_per": 80,
            "left_knuckle": 100,
            "right_knuckle": 121,
            "left_nail": 102,
            "right_nail": 123,
            #"left_ear": 39,
            #"right_ear": 23,
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
        bdy_part = {k: data[v] for k, v in body_parts_index.items()}
        # front up
        bdy_part_up = {k: data_up[v] for k, v in body_parts_index.items()}

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

        bdy_part_r_side = {k: data_r_side[v] for k, v in body_parts_index_r.items()}
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
        bdy_part_pike = {k: data_pike[v] for k, v in body_parts_index_pike.items()}
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
        bdy_part_r_pike = {k: data_l_pike[v] for k, v in body_parts_index_l_pike.items()}
        # front
        hand_part_pos = []
        for hand_position in hand_pos:
            hand_part_pos.append(data[hand_position])
        bdy_part["left_knuckles"] = np.mean(hand_part_pos[0:3], axis=0)
        bdy_part["right_knuckles"] = np.mean(hand_part_pos[4:7], axis=0)
        bdy_part["left_nails"] = np.mean(hand_part_pos[8:11], axis=0)
        bdy_part["right_nails"] = np.mean(hand_part_pos[12:15], axis=0)
        for hand_position in hand_pos:
            if data[hand_position][0] < 0:
                bdy_part["left_knuckles"] = data[100]
                bdy_part["right_knuckles"] = data[121]
                bdy_part["left_nails"] = data[102]
                bdy_part["right_nails"] = data[123]
        left_lowest_front_rib_approx = (data[5] + data[11] * 1.2) / 2.2
        bdy_part["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data[6] + data[12] * 1.2) / 2.2
        bdy_part["right_lowest_front_rib"] = right_lowest_front_rib_approx
        bdy_part["left_shoulder_perimeter_width"] = get_max_pt(data[7], data[5], edges)
        bdy_part["right_shoulder_perimeter_width"] = get_max_pt(data[8], data[6], edges)
        bdy_part["left_nipple"] = (left_lowest_front_rib_approx * 1.1 + data[5]) / 2.1
        bdy_part["right_nipple"] = (right_lowest_front_rib_approx * 1.1 + data[6]) / 2.1
        bdy_part["left_umbiculus"] = (
                                                   (left_lowest_front_rib_approx * 2.9) + (data[11] * 2.1)
                                           ) / 5
        bdy_part["right_umbiculus"] = (
                                                    (right_lowest_front_rib_approx * 2.9) + (data[12] * 2.1)
                                            ) / 5
        left_arch_approx = (data[17] + data[19]) / 2
        bdy_part["left_arch"] = left_arch_approx
        right_arch_approx = (data[20] + data[22]) / 2
        bdy_part["right_arch"] = right_arch_approx
        bdy_part["left_ball"] = (data[17] + left_arch_approx) / 2
        bdy_part["right_ball"] = (data[20] + right_arch_approx) / 2
        bdy_part["left_mid_arm"] = (data[5] + data[7]) / 2
        bdy_part["right_mid_arm"] = (data[6] + data[8]) / 2
        bdy_part["left_acromion"] = find_acromion_left(edges, data, 0)
        bdy_part["right_acromion"] = find_acromion_right(edges, data[0], data[6], 0)
        bdy_part["left_acromion_height"] = find_acromion_left(edges, data, 1)
        bdy_part["right_acromion_height"] = find_acromion_right(edges, data[0], data[6], 1)
        bdy_part["top_of_head"] = find_top_of_head(data, edges)
        bdy_part["left_mid_elbow_wrist"] = (data[9] + data[7]) / 2
        bdy_part["right_mid_elbow_wrist"] = (data[10] + data[8]) / 2
        bdy_part["right_maximum_forearm"] = get_max_pt(data[8], bdy_part["right_mid_elbow_wrist"],
                                                       edges)
        bdy_part["left_maximum_forearm"] = get_max_pt(data[7], bdy_part["left_mid_elbow_wrist"],
                                                      edges)
        bdy_part["right_maximum_calf"] = get_max_pt(data[14] + np.array([0, 5]), data[16], edges)
        bdy_part["left_maximum_calf"] = get_max_pt(data[13] + np.array([0, 5]), data[15], edges)
        bdy_part["right_crotch"], bdy_part["left_crotch"] = get_crotch_right_left(edges_short, data)
        bdy_part["right_mid_thigh"], bdy_part["left_mid_thigh"] = get_mid_thigh_right_left(data,
                                                                                           bdy_part[
                                                                                                                 "right_crotch"],
                                                                                           bdy_part[
                                                                                                                 "left_crotch"])
        bdy_part["crotch_width"] = min(max_perp(bdy_part["left_crotch"], bdy_part["left_knee"], edges_short) * self.bottom_ratio, max_perp(bdy_part["right_crotch"], bdy_part["right_knee"], edges_short) * self.bottom_ratio)
        if bdy_part["crotch_width"] > 30:
            bdy_part["crotch_width"] /= 2
        print("left_shoulder_perimeter_width", max_perp(bdy_part["left_shoulder_perimeter_width"], bdy_part["left_elbow"], edges))
        print("right_shoulder_perimeter_width", max_perp(bdy_part["right_shoulder_perimeter_width"], bdy_part["right_elbow"], edges))
        print("La0p", circle_p(max_perp(bdy_part["left_shoulder_perimeter_width"], bdy_part["left_shoulder_perimeter_width"] + np.array([1, 0]), edges)) * self.ratio)
        print("Lb0p", circle_p(max_perp(bdy_part["right_shoulder_perimeter_width"], bdy_part["right_shoulder_perimeter_width"] + np.array([-1, 0]), edges)) * self.ratio)
        hand_pos_grp = find_hand_pos_grp(data, 0) * self.ratio2
        # front up
        bdy_part_up["right_base_of_thumb"], bdy_part_up["right_knuckles"], bdy_part_up["right_nails"] = (
            np.maximum(data_up[114], data_up[93]), np.maximum(data_up[117], data_up[96]), np.maximum(data_up[119], data_up[98]))
        # right side
        right_lowest_front_rib_approx = np.array([data_r_side[12][0],(data_r_side[6][1] + data_r_side[12][1]) / 2])
        bdy_part_r_side["right_lowest_front_rib"] = right_lowest_front_rib_approx
        bdy_part_r_side["right_nipple"] = np.array([right_lowest_front_rib_approx[0],(right_lowest_front_rib_approx[1] + data_r_side[6][1]) / 2])
        bdy_part_r_side["right_umbiculus"] = (
                                                      (right_lowest_front_rib_approx * 3) + (data_r_side[12] * 2)
                                              ) / 5
        bdy_part_r_side["right_maximum_calf"] = get_max_pt(data_r_side[16], data_r_side[14], edges_r_side)
        # front pike
        point, dist = get_maximum_pit(data_pike[16], edges_pike)
        bdy_part_pike["right_toe_nail"] = np.array([point[0], point[1] - 2 / self.ratio_pike])
        bdy_part_pike["right_ball"] = get_max_pt(data_pike[16], bdy_part_pike["right_toe_nail"], edges_pike)
        bdy_part_pike["right_arch"] = (data_pike[16] + bdy_part_pike["right_ball"]) / 2
        # left side pike
        bdy_part_r_pike["right_toe_nail"], dist = get_maximum_pit(data_l_pike[16], edges_l_pike)
        bdy_part_r_pike["right_heel"] = get_max_pt(data_l_pike[16], bdy_part_r_pike["right_toe_nail"], edges_l_pike)
        bdy_part_r_pike["right_ball"] = np.array([bdy_part_r_pike["right_toe_nail"][0], bdy_part_r_pike["right_toe_nail"][1] - 3 / self.ratio_l_pike])
        bdy_part_r_pike["right_arch"] = (bdy_part_r_pike["right_heel"] + bdy_part_r_pike["right_ball"]) / 2

        # print(bdy_part_r_side)
        self.keypoints = {

            "Ls1L": abs(bdy_part["right_umbiculus"][1] - bdy_part["right_hip"][1]) * self.ratio2,
            "Ls2L": abs(bdy_part["right_lowest_front_rib"][1]- bdy_part["right_hip"][1]) * self.ratio2,
            "Ls3L": abs(bdy_part["right_nipple"][1] - bdy_part["right_hip"][1]) * self.ratio2,
            "Ls4L": abs(bdy_part["right_shoulder"][1] - bdy_part["right_hip"][1]) * self.ratio2,
            "Ls5L": abs(bdy_part["right_acromion_height"][1] - bdy_part["right_hip"][1]) * self.ratio2,
            "Ls6L": abs(bdy_part["left_acromion_height"][1] - bdy_part["nose"][1]) * self.ratio2,
            "Ls7L": np.linalg.norm(bdy_part["left_acromion_height"] - bdy_part["left_ear"]) * self.ratio2,
            "Ls8L": abs(bdy_part["left_acromion_height"][1] - bdy_part["top_of_head"][1]) * self.ratio2,

            "Ls0p": stad_p(max_line(bdy_part["right_hip"], bdy_part["left_hip"], edges_short) * self.ratio,
                max_perp(bdy_part_r_side["right_hip"], bdy_part_r_side["right_knee"], edges_r_side) * self.ratio_r_side),
            "Ls1p": stad_p(max_line(bdy_part["left_umbiculus"], bdy_part["right_umbiculus"], edges) * self.ratio,
                max_perp(bdy_part_r_side["right_umbiculus"], bdy_part_r_side["right_knee"], edges_r_side) * self.ratio_r_side),
            "Ls2p": stad_p(max(max_line(bdy_part["right_lowest_front_rib"], bdy_part["left_lowest_front_rib"], edges), np.linalg.norm(bdy_part["right_lowest_front_rib"] - bdy_part["left_lowest_front_rib"])) * self.ratio,
                max_perp(bdy_part_r_side["right_lowest_front_rib"], bdy_part_r_side["right_hip"], edges_r_side) * self.ratio_r_side),
            "Ls3p": stad_p(max_line(bdy_part["right_nipple"], bdy_part["left_nipple"], edges) * self.ratio,
                max_perp(bdy_part_r_side["right_nipple"], bdy_part_r_side["right_hip"], edges_r_side) * self.ratio_r_side),
            "Ls5p": circle_p(abs(bdy_part["left_acromion"][0] - bdy_part["right_acromion"][0]) * self.ratio),
            "Ls6p": circle_p(max_perp(bdy_part["nose_per"], [bdy_part["nose_per"][0], bdy_part["nose_per"][1] + 5], edges) * self.ratio),
            "Ls7p": circle_p(np.linalg.norm(bdy_part["left_ear"] - bdy_part["right_ear"]) * self.ratio),

            "Ls0w": max_line(bdy_part["left_hip"], bdy_part["right_hip"], edges_short) * self.ratio,
            "Ls1w": max(max_line(bdy_part["left_umbiculus"], bdy_part["right_umbiculus"], edges) * self.ratio,
                        np.linalg.norm(bdy_part["left_umbiculus"] - bdy_part["right_umbiculus"]) * self.ratio),
            "Ls2w": max(max_line(bdy_part["left_lowest_front_rib"], bdy_part["right_lowest_front_rib"], edges) * self.ratio,
                        np.linalg.norm(bdy_part["left_lowest_front_rib"] - bdy_part["right_lowest_front_rib"]) * self.ratio),
            "Ls3w": max(max_line(bdy_part["left_nipple"], bdy_part["right_nipple"], edges) * self.ratio,
                        np.linalg.norm(bdy_part["left_nipple"] - bdy_part["right_nipple"]) * self.ratio),
            "Ls4w": np.linalg.norm(bdy_part["left_shoulder"] - bdy_part["right_shoulder"]) * self.ratio,

            "Ls4d": max_perp(bdy_part_r_side["right_shoulder"], bdy_part_r_side["right_elbow"], edges_r_side) * self.ratio_r_side,

            # Not needed"La1L": (np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_elbow"])) / 2,
            "La2L": np.linalg.norm(bdy_part["left_shoulder"] - bdy_part["left_elbow"]) * self.ratio,
            "La3L": max(np.linalg.norm(bdy_part["left_shoulder"] - bdy_part["left_maximum_forearm"]) * self.ratio,
                        np.linalg.norm(bdy_part["right_shoulder"] - bdy_part["right_maximum_forearm"]) * self.ratio),
            "La4L": np.linalg.norm(bdy_part["left_shoulder"] - bdy_part["left_wrist"]) * self.ratio,
            "La5L": np.linalg.norm(bdy_part["left_wrist"] - bdy_part["left_base_of_thumb"]) * self.ratio,
            "La6L": np.linalg.norm(bdy_part["left_wrist"] - bdy_part["left_knuckle"]) * self.ratio,
            "La7L": np.linalg.norm(bdy_part["left_wrist"] - bdy_part["left_nail"]) * self.ratio,

            "La0p": (circle_p(max_perp(bdy_part["left_shoulder_perimeter_width"], bdy_part["left_shoulder_perimeter_width"] + np.array([1, 0]), edges))
                     + circle_p(max_perp(bdy_part["right_shoulder_perimeter_width"], bdy_part["right_shoulder_perimeter_width"] + np.array([1, 0]), edges))) / 2 * self.ratio2,
            "La1p": (circle_p(max_perp(bdy_part["left_mid_arm"], bdy_part["left_elbow"], edges))
                     + circle_p(max_perp(bdy_part["right_mid_arm"], bdy_part["right_elbow"], edges))) / 2 * self.ratio2,
            "La2p": (circle_p(max_perp(bdy_part["left_elbow"], bdy_part["left_mid_arm"], edges))
                     + circle_p(max_perp(bdy_part["right_elbow"], bdy_part["right_mid_arm"], edges))) / 2 * self.ratio2,
            "La3p": (circle_p(max_perp(bdy_part["left_maximum_forearm"], bdy_part["left_elbow"], edges))
                     + circle_p(max_perp(bdy_part["right_maximum_forearm"], bdy_part["right_elbow"], edges))) / 2 * self.ratio2,
            "La4p": (stad_p(
                max_perp(bdy_part["left_wrist"], bdy_part["left_elbow"], edges),
                max_perp(bdy_part_up["left_wrist"], bdy_part_up["left_elbow"], edges_up)) + stad_p(
                max_perp(bdy_part["right_wrist"], bdy_part["right_elbow"], edges),
                max_perp(bdy_part_up["right_wrist"], bdy_part_up["right_elbow"], edges_up))) / 2 * self.ratio_up2,
            "La5p": stad_p(max_perp(bdy_part["left_base_of_thumb"], data[94], edges) * self.ratio2, 0), #get_maximum_start(bdy_part_up["right_base_of_thumb"], bdy_part_up["left_wrist"], edges_up) * self.ratio_up2),
            "La6p": stad_p(max_perp(bdy_part["left_knuckles"], bdy_part["left_wrist"], edges) * self.ratio2, 0), #get_maximum_start(bdy_part_up["right_knuckles"], bdy_part_up["left_wrist"], edges_up) * self.ratio_up2),
            "La7p": stad_p(max_perp(bdy_part["left_nails"], bdy_part["left_wrist"], edges) * self.ratio2,
                max_perp(bdy_part_up["right_nails"], bdy_part_up["left_wrist"], edges_up) * self.ratio_up2),

            "La4w": max_perp(bdy_part["left_wrist"], bdy_part["left_elbow"], edges) * self.ratio2,
            "La5w": max_perp(bdy_part["left_base_of_thumb"], data[94], edges) * self.ratio2,
            "La6w": max_perp(bdy_part["left_knuckles"], bdy_part["left_wrist"], edges) * self.ratio2,
            "La7w": max_perp(bdy_part["left_nails"], bdy_part["left_wrist"], edges) * self.ratio2,

            # Not needed"Lb1L": (np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_elbow"])) / 2,
            "Lb2L": abs(bdy_part["right_shoulder"][0] - bdy_part["right_elbow"][0]) * self.ratio,
            "Lb3L": max(np.linalg.norm(bdy_part["left_shoulder"] - bdy_part["left_maximum_forearm"]) * self.ratio,
                        np.linalg.norm(bdy_part["right_shoulder"] - bdy_part["right_maximum_forearm"]) * self.ratio),
            "Lb4L": abs(bdy_part["right_shoulder"][0] - bdy_part["right_wrist"][0]) * self.ratio,
            "Lb5L": abs(bdy_part["right_wrist"][0] - bdy_part["right_base_of_thumb"][0]) * self.ratio,
            "Lb6L": abs(bdy_part["right_wrist"][0] - bdy_part["right_knuckle"][0]) * self.ratio,
            "Lb7L": abs(bdy_part["right_wrist"][0] - bdy_part["right_nail"][0]) * self.ratio,

            "Lb0p": (circle_p(max_perp(bdy_part["left_shoulder_perimeter_width"], bdy_part["left_shoulder_perimeter_width"] + np.array([1, 0]), edges))
                     + circle_p(max_perp(bdy_part["right_shoulder_perimeter_width"], bdy_part["right_shoulder_perimeter_width"] + np.array([1, 0]), edges))) / 2 * self.ratio2,
            "Lb1p": (circle_p(max_perp(bdy_part["left_mid_arm"], bdy_part["left_elbow"], edges))
                     + circle_p(max_perp(bdy_part["right_mid_arm"], bdy_part["right_elbow"], edges))) / 2 * self.ratio2,
            "Lb2p": (circle_p(max_perp(bdy_part["left_elbow"], bdy_part["left_mid_arm"], edges))
                     + circle_p(max_perp(bdy_part["right_elbow"], bdy_part["right_mid_arm"], edges))) / 2 * self.ratio2,
            "Lb3p": (circle_p(max_perp(bdy_part["left_maximum_forearm"], bdy_part["left_elbow"], edges))
                     + circle_p(max_perp(bdy_part["right_maximum_forearm"], bdy_part["right_elbow"], edges))) / 2 * self.ratio2,
            "Lb4p": (stad_p(
                max_perp(bdy_part["left_wrist"], bdy_part["left_elbow"], edges),
                max_perp(bdy_part_up["left_wrist"], bdy_part_up["left_elbow"], edges_up)) + stad_p(
                max_perp(bdy_part["right_wrist"], bdy_part["right_elbow"], edges),
                max_perp(bdy_part_up["right_wrist"], bdy_part_up["right_elbow"], edges_up))) / 2 * self.ratio_up2,
            "Lb5p": stad_p(max_perp(bdy_part["right_base_of_thumb"], data[115], edges) * self.ratio2, 0),#get_maximum_start(bdy_part_up["right_base_of_thumb"], bdy_part_up["left_wrist"], edges_up) * self.ratio_up2),
            "Lb6p": stad_p(max_perp(bdy_part["right_knuckles"], bdy_part["right_wrist"], edges) * self.ratio2, 0),#get_maximum_start(bdy_part_up["right_knuckles"], bdy_part_up["left_wrist"], edges_up) * self.ratio_up2),
            "Lb7p": stad_p(max_perp(bdy_part["right_nails"], bdy_part["right_wrist"], edges) * self.ratio2,
                           max_perp(bdy_part_up["right_nails"], bdy_part_up["left_wrist"], edges_up) * self.ratio_up2),

            "Lb4w": max_perp(bdy_part["right_wrist"], bdy_part["right_elbow"], edges) * self.ratio2,
            "Lb5w": max_perp(bdy_part["right_base_of_thumb"], data[115], edges) * self.ratio2,
            "Lb6w": max_perp(bdy_part["right_knuckles"], bdy_part["right_wrist"], edges) * self.ratio2,
            "Lb7w": max_perp(bdy_part["right_nails"], bdy_part["right_wrist"], edges) * self.ratio2,

            "Lj1L": np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_crotch"]) * self.bottom_ratio2,
            # Not needed"Lj2L": (np.linalg.norm(body_parts_pos["left_hip"] - body_par&ts_pos["left_knee"])) / 2,
            "Lj3L": np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_knee"]) * self.bottom_ratio2,
            "Lj4L": max(np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_maximum_calf"]) * self.bottom_ratio2,
                        np.linalg.norm(bdy_part["right_hip"] - bdy_part["right_maximum_calf"]) * self.bottom_ratio2),
            "Lj5L": max(np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_ankle"]) * self.bottom_ratio2,
                        np.linalg.norm(bdy_part["right_hip"] - bdy_part["right_ankle"]) * self.bottom_ratio2),
            "Lj6L": 1,
            # not measured "Lj7L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_arch"]),
            "Lj8L": min(np.linalg.norm(bdy_part["right_ankle"] - bdy_part["right_ball"]) * self.bottom_ratio2,
                        np.linalg.norm(bdy_part["left_ankle"] - bdy_part["left_ball"]) * self.bottom_ratio2),
            "Lj9L": min(np.linalg.norm(bdy_part_pike["right_ankle"] - bdy_part_pike["right_toe_nail"]) * self.ratio_pike,
                        np.linalg.norm(bdy_part_pike["left_ankle"] - bdy_part_pike["left_toe_nail"]) * self.ratio_pike),

            # not measured "Lj0p":,
            "Lj1p": circle_p(bdy_part["crotch_width"]),
            "Lj2p": (circle_p(max_perp(bdy_part["right_mid_thigh"], bdy_part["right_hip"], edges_short))
                     + circle_p(max_perp(bdy_part["left_mid_thigh"], bdy_part["left_hip"], edges_short))) / 2 * self.bottom_ratio,
            "Lj3p": (circle_p(max_perp(bdy_part["right_knee"], bdy_part["right_hip"], edges))
                     + circle_p(max_perp(bdy_part["left_knee"], bdy_part["left_hip"], edges))) / 2 * self.bottom_ratio,
            "Lj4p": (circle_p(max_perp(bdy_part["right_maximum_calf"], bdy_part["right_knee"], edges))
                     + circle_p(max_perp(bdy_part["left_maximum_calf"], bdy_part["left_knee"], edges))) / 2 * self.bottom_ratio,
            "Lj5p": (circle_p(max_perp(bdy_part_pike["right_ankle"], bdy_part_pike["right_knee"], edges_pike))
                     + circle_p(max_perp(bdy_part_pike["left_ankle"], bdy_part_pike["left_knee"], edges_pike))) / 2 * self.ratio_pike,
            "Lj6p": stad_p(max_perp(bdy_part_pike["right_ankle"], bdy_part_pike["right_toe_nail"], edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_ankle"], bdy_part_r_pike["right_toe_nail"], edges_l_pike) * self.ratio_l_pike),
            "Lj7p": stad_p(max_perp(bdy_part_pike["right_arch"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_arch"], bdy_part_r_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lj8p": stad_p(max_perp(bdy_part_pike["right_ball"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_ball"], bdy_part_r_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lj9p": stad_p(max_perp(bdy_part_pike["right_toe_nail"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_toe_nail"], bdy_part_r_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),

            "Lj8w": max_perp(bdy_part_pike["right_ball"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,
            "Lj9w": max_perp(bdy_part_pike["right_toe_nail"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,

            "Lj6d": max_perp(bdy_part_pike["right_ankle"], bdy_part_pike["right_knee"], edges_pike) * self.ratio_pike,

            "Lk1L": np.linalg.norm(bdy_part["right_hip"] - bdy_part["right_crotch"]) * self.bottom_ratio2,
            # Not needed"Lk2L": (np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_knee"]) * self.ratio) / 2,
            "Lk3L": np.linalg.norm(bdy_part["right_hip"] - bdy_part["right_knee"]) * self.bottom_ratio2,
            "Lk4L": max(np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_maximum_calf"]) * self.bottom_ratio2,
                        np.linalg.norm(bdy_part["right_hip"] - bdy_part["right_maximum_calf"]) * self.bottom_ratio2),
            "Lk5L": max(np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_ankle"]) * self.bottom_ratio2,
                        np.linalg.norm(bdy_part["right_hip"] - bdy_part["right_ankle"]) * self.bottom_ratio2),
            "Lk6L": 1,
            # not measured "Lk7L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_arch"]),
            "Lk8L": min(np.linalg.norm(bdy_part["right_ankle"] - bdy_part["right_ball"]) * self.bottom_ratio2,
                        np.linalg.norm(bdy_part["left_ankle"] - bdy_part["left_ball"]) * self.bottom_ratio2),
            "Lk9L": min(np.linalg.norm(bdy_part_pike["right_ankle"] - bdy_part_pike["right_toe_nail"]) * self.ratio_pike,
                        np.linalg.norm(bdy_part_pike["left_ankle"] - bdy_part_pike["left_toe_nail"]) * self.ratio_pike),

            # not measured "Lk0p":,
            "Lk1p": circle_p(bdy_part["crotch_width"]),
            "Lk2p": (circle_p(max_perp(bdy_part["right_mid_thigh"], bdy_part["right_hip"], edges_short))
                     + circle_p(max_perp(bdy_part["left_mid_thigh"], bdy_part["left_hip"], edges_short))) / 2 * self.bottom_ratio,
            "Lk3p": (circle_p(max_perp(bdy_part["right_knee"], bdy_part["right_hip"], edges))
                     + circle_p(max_perp(bdy_part["left_knee"], bdy_part["left_hip"], edges))) / 2 * self.bottom_ratio,
            "Lk4p": (circle_p(max_perp(bdy_part["right_maximum_calf"], bdy_part["right_knee"], edges))
                     + circle_p(max_perp(bdy_part["left_maximum_calf"], bdy_part["left_knee"], edges))) / 2 * self.bottom_ratio,
            "Lk5p": (circle_p(max_perp(bdy_part_pike["right_ankle"], bdy_part_pike["right_knee"], edges_pike))
                     + circle_p(max_perp(bdy_part_pike["left_ankle"], bdy_part_pike["left_knee"], edges_pike))) / 2 * self.ratio_pike,
            "Lk6p": stad_p(max_perp(bdy_part_pike["right_ankle"], bdy_part_pike["right_toe_nail"], edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_ankle"], bdy_part_r_pike["right_toe_nail"], edges_l_pike) * self.ratio_l_pike),
            "Lk7p": stad_p(max_perp(bdy_part_pike["right_arch"], bdy_part_pike["right_ankle"],edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_arch"], bdy_part_r_pike["right_ankle"],edges_l_pike) * self.ratio_l_pike),
            "Lk8p": stad_p(max_perp(bdy_part_pike["right_ball"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_ball"], bdy_part_r_pike["right_ankle"], edges_l_pike) * self.ratio_l_pike),
            "Lk9p": stad_p(max_perp(bdy_part_pike["right_toe_nail"], bdy_part_pike["right_ankle"],edges_pike) * self.ratio_pike,
                           max_perp(bdy_part_r_pike["right_toe_nail"], bdy_part_r_pike["right_ankle"],edges_l_pike) * self.ratio_l_pike),

            "Lk8w": max_perp(bdy_part_pike["right_ball"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,
            "Lk9w": max_perp(bdy_part_pike["right_toe_nail"], bdy_part_pike["right_ankle"], edges_pike) * self.ratio_pike,

            "Lk6d": max_perp(bdy_part_pike["right_ankle"], bdy_part_pike["right_knee"], edges_pike) * self.ratio_pike,
        }
        print(max_perp(bdy_part["right_shoulder_perimeter_width"], bdy_part["right_shoulder_perimeter_width"] + np.array([-1, 0]), edges))
        print(max_perp(bdy_part["left_shoulder_perimeter_width"], bdy_part["left_shoulder_perimeter_width"] + np.array([1, 0]), edges))
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
