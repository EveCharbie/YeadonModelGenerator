import openpifpaf
import sys

from src.utils.find_body_parts import *
from src.utils.image_config import *
from src.utils.perimeter_calculator import *
from src.utils.generate_yml import generate_yml





class YeadonModel:
    """A class used to represent a Yeadon Model.

    Attributes
    ----------
    keypoints : dict
        A dictionary containing the keypoints of the image. (Ls0, Ls1, ...)
    """

    def __init__(self, impath_front: str, impath_pike: str, impath_r_tuck: str, impath_side: str, impath_tuck: str):
        """Creates a YeadonModel object from an image path.

        Parameters
        ----------
        impath_front : str
            The path to the front image to be processed.
        impath_side : str
            The path to the side image to be processed.
        impath_tuck : str
            The path to the tuck image to be processed.
        impath_r_tuck : str
            The path to the right tuck image to be processed.
        Returns
        -------
        YeadonModel
            The YeadonModel object with the keypoints of the image.
        """
        # front
        pil_im, image, im = create_resize_remove_im(impath_front)
        edges = thresh(im, image, 2)
        # edges short was for the edges for the hip to the knee because the original detection had some difficulty to detect the black of the short
        edges_short = thresh(im, image, 2)
        predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k30-wholebody")
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
        # You can find the index here:
        # https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
        # as "predictions" is an array the index starts at 0 and not at 1 like in the github
        data = predictions[0].data[:, 0:2]
        edges = better_edges(edges, data)
        edges_short = better_edges(edges_short, data)

        self.ratio, self.ratio2 = get_ratio2(image)
        self.ratio, self.ratio2 = get_new_ratio(355.6, 355.6 - 50.8, 150, self.ratio)
        self.ratio_bottom, self.ratio_bottom2 = self.ratio, self.ratio2
        # right side
        pil_r_side_im, image_r_side, im_r_side = create_resize_remove_im(impath_side)

        edges_r_side = thresh(im_r_side, image_r_side, 1)
        predictions2, gt_anns2, image_meta2 = predictor.pil_image(pil_r_side_im)
        data_r_side = predictions2[0].data[:, 0:2]

        self.ratio_r_side, self.ratio_r_side2 = get_ratio2(image_r_side)
        self.ratio_r_side, self.ratio_r_side2 = get_new_ratio(355.6, 355.6 - 50.8, 150, self.ratio_r_side)

        # front tuck
        pil_tuck_im, image_tuck, im_tuck = create_resize_remove_im(impath_tuck)

        edges_tuck = thresh(im_tuck, image_tuck, 2)
        predictions5, gt_anns5, image_meta5 = predictor.pil_image(pil_tuck_im)
        data_tuck = predictions5[0].data[:, 0:2]
        self.ratio_tuck, self.ratio_tuck2 = get_ratio2(image_tuck)
        self.ratio_tuck, self.ratio_tuck2 = get_new_ratio(355.6, 355.6 - 50.8, 150, self.ratio_tuck)

        # right side tuck
        pil_l_tuck_im, image_l_tuck, im_l_tuck = create_resize_remove_im(impath_r_tuck)

        edges_l_tuck = thresh(im_l_tuck, image_l_tuck, 2)
        predictions6, gt_anns6, image_meta6 = predictor.pil_image(pil_l_tuck_im)
        data_l_tuck = predictions6[0].data[:, 0:2]
        self.ratio_l_tuck, self.ratio_l_tuck2 = get_ratio2(image_l_tuck)
        self.ratio_l_tuck, self.ratio_l_tuck2 = get_new_ratio(355.6, 355.6 - 50.8, 150, self.ratio_l_tuck)
        # pike
        pil_pike_im, image_pike, im_pike = create_resize_remove_im(impath_pike)

        edges_pike = thresh(im_pike, image_pike, 2)
        predictions3, gt_anns3, image_meta3 = predictor.pil_image(pil_pike_im)
        data_pike = predictions3[0].data[:, 0:2]
        self.ratio_pike, self.ratio_pike2 = get_ratio2(image_pike)
        self.ratio_pike, self.ratio_pike2 = get_new_ratio(355.6, 355.6 - 50.8, 150, self.ratio_pike)
        # front
        body_parts_index = {
            "nose": 0,
            "nose_per": 74,
            "left_knuckles": 100,
            "right_knuckles": 121,
            "left_ear": 39,
            "right_ear": 23,
            # "left_ear": 3,
            # "right_ear": 4,
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
        bdy_part = {k: data[v] for k, v in body_parts_index.items()}
        #self.ratio, self.ratio2 = get_ratio_meas_top(bdy_part["right_elbow"], bdy_part["right_wrist"])
        # right side
        body_parts_index_r = {
            "nose": 0,
            "right_ear": 23,
            "right_shoulder": 6,
            "right_elbow": 8,
            "right_wrist": 10,
            "right_hip": 12,
            "right_knee": 14,
            "right_ankle": 16,
        }

        bdy_part_r_side = {k: data_r_side[v] for k, v in body_parts_index_r.items()}
        #self.ratio_r_side, self.ratio_r_side2 = get_ratio_meas_bottom(bdy_part_r_side["right_knee"], bdy_part_r_side["right_ankle"])
        # front tuck
        body_parts_index_tuck = {
            "left_wrist": 9,
            "right_wrist": 10,
            "left_elbow": 7,
            "right_elbow": 8,
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
        bdy_part_tuck = {k: data_tuck[v] for k, v in body_parts_index_tuck.items()}
        #self.ratio_tuck, self.ratio_tuck2 = get_ratio_meas_bottom(bdy_part_tuck["right_knee"], bdy_part_tuck["right_ankle"])
        # left side tuck
        body_parts_index_r_tuck = {
            "right_ear": 4,
            "right_knuckle": 121,
            "right_shoulder": 6,
            "right_elbow": 8,
            "right_hip": 12,
            "right_knee": 14,
            "right_ankle": 16,
            "right_heel": 22,
            "right_toe_nail": 20,
        }
        bdy_part_r_tuck = {k: data_l_tuck[v] for k, v in body_parts_index_r_tuck.items()}
        #self.ratio_r_tuck, self.ratio_r_tuck2 = get_ratio_meas_bottom(bdy_part_r_tuck["right_knee"], bdy_part_r_tuck["right_ankle"])
        # pike
        body_parts_index_pike = {
            "right_knee": 14,
            "right_wrist": 10,
            "right_knuckle": 121,
            "right_shoulder": 6,
            "right_elbow": 8,

        }
        bdy_part_pike = {k: data_pike[v] for k, v in body_parts_index_pike.items()}
        bdy_part_pike["right_mid_arm"] = (data_pike[6] + data_pike[8]) / 2
        bdy_part_pike["right_hand"] = (bdy_part_pike["right_wrist"] + bdy_part_pike["right_knuckle"]) / 2
        # front
        bdy_part["left_nails"] = (data[102] + data[103]) / 2
        bdy_part["right_nails"] = (data[123] + data[124]) / 2
        left_lowest_front_rib_approx = (data[5] + data[11] * 1.1) / 2.1
        bdy_part["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data[6] + data[12] * 1.1) / 2.1
        bdy_part["right_lowest_front_rib"] = right_lowest_front_rib_approx

        bdy_part["left_nipple"] = (left_lowest_front_rib_approx + data[5] * 1.4) / 2.4
        bdy_part["right_nipple"] = (right_lowest_front_rib_approx + data[6] * 1.4) / 2.4
        bdy_part["left_umbiculus"] = ((left_lowest_front_rib_approx * 3.1) + (data[11] * 1.9)) / 5
        bdy_part["right_umbiculus"] = ((right_lowest_front_rib_approx * 3.1) + (data[12] * 1.9)) / 5
        left_arch_approx = (data[17] + data[19]) / 2
        bdy_part["left_arch"] = left_arch_approx
        right_arch_approx = (data[20] + data[22]) / 2
        bdy_part["right_arch"] = right_arch_approx
        bdy_part["left_ball"] = (data[17] + left_arch_approx) / 2
        bdy_part["right_ball"] = (data[20] + right_arch_approx) / 2
        bdy_part["left_mid_arm"] = (data[5] + data[7]) / 2
        bdy_part["right_mid_arm"] = (data[6] + data[8]) / 2
        bdy_part["left_shoulder_perimeter_width"] = (data[5] * 1.3 + bdy_part["left_mid_arm"]) / 2.3
        bdy_part["right_shoulder_perimeter_width"] = (data[6] * 1.3 + bdy_part["right_mid_arm"]) / 2.3
        bdy_part["left_acromion"] = find_acromion_left(edges, data, 0)
        bdy_part["right_acromion"] = find_acromion_right(edges, data[0], data[6], 0)
        bdy_part["left_acromion_height"] = find_acromion_left(edges, data, 1)
        bdy_part["right_acromion_height"] = find_acromion_right(edges, data[0], data[6], 1)
        bdy_part["top_of_head"] = find_top_of_head(data, edges)
        bdy_part["left_mid_elbow_wrist"] = (data[9] + data[7]) / 2
        bdy_part["right_mid_elbow_wrist"] = (data[10] + data[8]) / 2
        bdy_part["right_maximum_forearm"] = np.array(get_max_pt(data[8], bdy_part["right_mid_elbow_wrist"], edges))
        bdy_part["left_maximum_forearm"] = np.array(get_max_pt(data[7], bdy_part["left_mid_elbow_wrist"], edges))
        bdy_part["right_maximum_calf"] = np.array(get_max_pt(data[14] + np.array([0, 5]), data[16], edges))
        bdy_part["left_maximum_calf"] = np.array(get_max_pt(data[13] + np.array([0, 5]), data[15], edges))
        #if np.linalg.norm(bdy_part["right_maximum_calf"] - bdy_part["right_knee"]) < 10:
        #    bdy_part["right_maximum_calf"] = (data[14] * 3 + data[16] * 2) / 5
        #if np.linalg.norm(bdy_part["left_maximum_calf"] - bdy_part["left_knee"]) < 10:
        #    bdy_part["left_maximum_calf"] = (data[13] * 3 + data[15] * 2) / 5
        bdy_part["right_crotch"], bdy_part["left_crotch"] = get_crotch_right_left(edges_short, data)
        bdy_part["right_mid_thigh"], bdy_part["left_mid_thigh"] = get_mid_thigh_right_left(data, bdy_part["right_crotch"], bdy_part["left_crotch"])
        bdy_part["left_wrist_width"] = max_perp(bdy_part["left_wrist"], bdy_part["left_elbow"], edges, image)
        bdy_part["right_wrist_width"] = max_perp(bdy_part["right_wrist"], bdy_part["right_elbow"], edges, image)
        bdy_part["left_nails_width"] = max_perp(bdy_part["left_nails"], bdy_part["left_wrist"], edges, image)
        bdy_part["right_nails_width"] = max_perp(bdy_part["right_nails"], bdy_part["right_wrist"], edges, image)
        bdy_part["left_knuckles_width"] = max_perp(bdy_part["left_knuckles"], bdy_part["left_wrist"], edges, image)
        bdy_part["right_knuckles_width"] = max_perp(bdy_part["right_knuckles"], bdy_part["right_wrist"], edges, image)
        bdy_part["crotch_width"] = min(
            max_perp(bdy_part["left_crotch"], bdy_part["left_knee"], edges_short, image) * self.ratio_bottom,
            max_perp(bdy_part["right_crotch"], bdy_part["right_knee"], edges_short, image) * self.ratio_bottom)
        bdy_part["hip"] = (bdy_part["right_hip"] + bdy_part["left_hip"]) / 2
        bdy_part["umbiculus"] = (bdy_part["right_umbiculus"] + bdy_part["left_umbiculus"]) / 2
        bdy_part["lowest_front_rib"] = (bdy_part["right_lowest_front_rib"] + bdy_part["left_lowest_front_rib"]) / 2
        bdy_part["nipple"] = (bdy_part["right_nipple"] + bdy_part["left_nipple"]) / 2
        bdy_part["shoulder"] = (bdy_part["right_shoulder"] + bdy_part["left_shoulder"]) / 2
        bdy_part["acromion"] = (bdy_part["left_acromion_height"] + bdy_part["right_acromion_height"]) / 2
        bdy_part["ear"] = (bdy_part["right_ear"] + bdy_part["left_ear"]) / 2
        # right side
        if data[56][0] > 0 and data[74][0] > 0:
            bdy_part_r_side["nose"] = (data[56] + data[74]) / 2
        right_lowest_front_rib_approx = np.array([data_r_side[12][0], (data_r_side[6][1] + data_r_side[12][1] * 1.1) / 2.1])
        bdy_part_r_side["right_lowest_front_rib"] = right_lowest_front_rib_approx
        bdy_part_r_side["right_nipple"] = np.array([right_lowest_front_rib_approx[0], (right_lowest_front_rib_approx[1] * 1.3 + data_r_side[6][1]) / 2.3])
        bdy_part_r_side["right_umbiculus"] = (right_lowest_front_rib_approx + data_r_side[12]) / 2
        bdy_part_r_side["right_maximum_calf"] = get_max_pt(data_r_side[14] + np.array([0, 5]), data_r_side[16], edges_r_side)

        # front tuck
        point, dist = get_maximum_pit(data_tuck[16], edges_tuck)
        bdy_part_tuck["right_toe_nail"] = np.array([point[0], point[1] - 2 / self.ratio_tuck])
        bdy_part_tuck["right_ball"] = np.array(get_max_pt(data_tuck[16], bdy_part_tuck["right_toe_nail"], edges_tuck))
        bdy_part_tuck["right_arch"] = (data_tuck[16] + bdy_part_tuck["right_ball"]) / 2
        bdy_part_tuck["left_wrist_width"] = max_perp(bdy_part_tuck["left_wrist"], bdy_part_tuck["left_elbow"], edges_tuck, image_tuck)

        # left side tuck
        bdy_part_r_tuck["right_toe_nail"], dist = get_maximum_pit(data_l_tuck[16], edges_l_tuck)
        bdy_part_r_tuck["right_heel"] = get_max_pt(data_l_tuck[16], bdy_part_r_tuck["right_toe_nail"], edges_l_tuck)
        bdy_part_r_tuck["right_ball"] = np.array(
            [bdy_part_r_tuck["right_toe_nail"][0], bdy_part_r_tuck["right_toe_nail"][1] - 3 / self.ratio_l_tuck])
        bdy_part_r_tuck["right_arch"] = (bdy_part_r_tuck["right_heel"] + bdy_part_r_tuck["right_ball"]) / 2

        self.keypoints = {
            "Ls1L": get_length(bdy_part["umbiculus"], bdy_part["hip"], image) * self.ratio2,
            "Ls2L": get_length(bdy_part["lowest_front_rib"], bdy_part["hip"], image) * self.ratio2,
            "Ls3L": get_length(bdy_part["nipple"], bdy_part["hip"], image) * self.ratio2,
            "Ls4L": get_length(bdy_part["shoulder"], bdy_part["hip"], image) * self.ratio2,
            "Ls5L": get_length(bdy_part["acromion"], bdy_part["hip"], image) * self.ratio2,
            "Ls6L": get_length(bdy_part["acromion"], bdy_part["nose_per"], image) * self.ratio2,
            "Ls7L": get_length(bdy_part["acromion"], bdy_part["ear"], image) * self.ratio2,
            "Ls8L": abs(bdy_part["left_acromion_height"][1] - bdy_part["top_of_head"][1]) * self.ratio2,

            "Ls0p": stad_p(max_line(bdy_part["right_hip"], bdy_part["left_hip"], edges_short, image) * self.ratio,
                           max_perp(bdy_part_r_side["right_hip"], bdy_part_r_side["right_knee"], edges_r_side, image_r_side) * self.ratio_r_side),
            "Ls1p": stad_p(max_line(bdy_part["left_umbiculus"], bdy_part["right_umbiculus"], edges, image) * self.ratio,
                           max_perp(bdy_part_r_side["right_umbiculus"], bdy_part_r_side["right_knee"], edges_r_side, image_r_side) * self.ratio_r_side),
            "Ls2p": stad_p(
                max(max_line(bdy_part["right_lowest_front_rib"], bdy_part["left_lowest_front_rib"], edges, image), np.linalg.norm(bdy_part["right_lowest_front_rib"] - bdy_part["left_lowest_front_rib"])) * self.ratio,
                max_perp(bdy_part_r_side["right_lowest_front_rib"], bdy_part_r_side["right_hip"], edges_r_side, image_r_side) * self.ratio_r_side
            ),
            "Ls3p": stad_p(max_line(bdy_part["right_nipple"], bdy_part["left_nipple"], edges, image) * self.ratio,
                           max_perp(bdy_part_r_side["right_nipple"], bdy_part_r_side["right_hip"], edges_r_side, image_r_side) * self.ratio_r_side),
            "Ls5p": circle_p(get_length(bdy_part["left_acromion"], bdy_part["right_acromion"], image) * self.ratio),
            "Ls6p": circle_p(max_perp(bdy_part["nose_per"], bdy_part["nose"], edges, image)) * self.ratio,
            "Ls7p": circle_p(get_length(bdy_part["left_ear"], bdy_part["right_ear"], image) * self.ratio),

            "Ls0w": max_line(bdy_part["left_hip"], bdy_part["right_hip"], edges_short, image) * self.ratio,
            "Ls1w": max(max_line(bdy_part["left_umbiculus"], bdy_part["right_umbiculus"], edges, image) * self.ratio,
                        np.linalg.norm(bdy_part["left_umbiculus"] - bdy_part["right_umbiculus"]) * self.ratio),
            "Ls2w": max(max_line(bdy_part["left_lowest_front_rib"], bdy_part["right_lowest_front_rib"], edges,image) * self.ratio,
                        np.linalg.norm(bdy_part["left_lowest_front_rib"] - bdy_part["right_lowest_front_rib"]) * self.ratio),
            "Ls3w": max(max_line(bdy_part["left_nipple"], bdy_part["right_nipple"], edges, image) * self.ratio,
                        np.linalg.norm(bdy_part["left_nipple"] - bdy_part["right_nipple"]) * self.ratio),
            "Ls4w": get_length(bdy_part["left_shoulder"], bdy_part["right_shoulder"], image) * self.ratio,

            "Ls4d": max_perp(bdy_part_r_side["right_shoulder"], bdy_part_r_side["right_elbow"], edges_r_side, image_r_side) * self.ratio_r_side,

            # Not needed"La1L": (np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_elbow"])) / 2,
            "La2L": get_length(bdy_part["left_shoulder"], bdy_part["left_elbow"], image) * self.ratio,
            "La3L": get_length(bdy_part["left_shoulder"], bdy_part["left_maximum_forearm"], image) * self.ratio,
            "La4L": get_length(bdy_part["left_shoulder"], bdy_part["left_wrist"], image) * self.ratio,
            "La5L": get_length(bdy_part["left_wrist"], bdy_part["left_base_of_thumb"], image) * self.ratio,
            "La6L": get_length(bdy_part["left_wrist"], bdy_part["left_knuckles"], image) * self.ratio,
            "La7L": get_length(bdy_part["left_wrist"], bdy_part["left_nails"], image) * self.ratio,

            "La0p": circle_p(max_perp(bdy_part["left_shoulder_perimeter_width"], bdy_part["left_shoulder"], edges,
                                      image)) * self.ratio2,
            "La1p": circle_p(max_perp(bdy_part["left_mid_arm"], bdy_part["left_elbow"], edges, image)) * self.ratio2,
            "La2p": circle_p(max_perp(bdy_part["left_elbow"], bdy_part["left_mid_arm"], edges, image)) * self.ratio2,
            "La3p": circle_p(max_perp(bdy_part["left_maximum_forearm"], bdy_part["left_elbow"], edges, image)) * self.ratio2,
            "La4p": stad_p(bdy_part["left_wrist_width"], bdy_part["left_wrist_width"] / 2) * self.ratio2,
            #"La4p": stad_p(bdy_part["left_wrist_width"] * self.ratio2, bdy_part_tuck["left_wrist_width"] * self.ratio_tuck) ,
            "La5p": stad_p(max_perp(bdy_part["left_base_of_thumb"], bdy_part["left_base_of_thumb"] + np.array([1, 0]), edges,image) * self.ratio2, 0),
            "La6p": stad_p(bdy_part["left_knuckles_width"], bdy_part["left_knuckles_width"] / 3) * self.ratio2,
            "La7p": stad_p(bdy_part["left_nails_width"], bdy_part["left_nails_width"] / 3) * self.ratio2,

            "La4w": bdy_part["left_wrist_width"] * self.ratio2,
            "La5w": max_perp(bdy_part["left_base_of_thumb"], bdy_part["left_base_of_thumb"] + np.array([1, 0]), edges, image) * self.ratio2,
            "La6w": max_perp(bdy_part["left_knuckles"], bdy_part["left_wrist"], edges, image) * self.ratio2,
            "La7w": max_perp(bdy_part["left_nails"], bdy_part["left_wrist"], edges, image) * self.ratio2,

            # Not needed"Lb1L": (np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_elbow"])) / 2,
            "Lb2L": get_length(bdy_part["right_shoulder"], bdy_part["right_elbow"], image) * self.ratio,
            "Lb3L": get_length(bdy_part["right_shoulder"], bdy_part["right_maximum_forearm"], image) * self.ratio,
            "Lb4L": get_length(bdy_part["right_shoulder"], bdy_part["right_wrist"], image) * self.ratio,
            "Lb5L": get_length(bdy_part["right_wrist"], bdy_part["right_base_of_thumb"], image) * self.ratio,
            "Lb6L": get_length(bdy_part["right_wrist"], bdy_part["right_knuckles"], image) * self.ratio,
            "Lb7L": get_length(bdy_part["right_wrist"], bdy_part["right_nails"], image) * self.ratio,

            "Lb0p": circle_p(max_perp(bdy_part["right_shoulder_perimeter_width"], bdy_part["right_shoulder"], edges, image)) * self.ratio2,
            "Lb1p": circle_p(max_perp(bdy_part["right_mid_arm"], bdy_part["right_elbow"], edges, image)) * self.ratio2,
            "Lb2p": circle_p(max_perp(bdy_part["right_elbow"], bdy_part["right_mid_arm"], edges, image)) * self.ratio2,
            "Lb3p": circle_p(max_perp(bdy_part["right_maximum_forearm"], bdy_part["right_elbow"], edges, image)) * self.ratio2,
            "Lb4p": stad_p(bdy_part["right_wrist_width"], bdy_part["right_wrist_width"] / 2) * self.ratio2,
            #"Lb4p": stad_p(bdy_part["right_wrist_width"] * self.ratio2, bdy_part_tuck["left_wrist_width"] * self.ratio_tuck) ,
            "Lb5p": stad_p(max_perp(bdy_part["right_base_of_thumb"], bdy_part["right_base_of_thumb"] + np.array([1, 0]), edges, image) * self.ratio2, 0),
            "Lb6p": stad_p(bdy_part["right_knuckles_width"], bdy_part["right_knuckles_width"] / 3) * self.ratio2,
            "Lb7p": stad_p(bdy_part["left_nails_width"], bdy_part["right_nails_width"] / 4) * self.ratio2,

            "Lb4w": bdy_part["right_wrist_width"] * self.ratio2,
            "Lb5w": max_perp(bdy_part["right_base_of_thumb"], bdy_part["right_base_of_thumb"] + np.array([1, 0]), edges, image) * self.ratio2,
            "Lb6w": max_perp(bdy_part["right_knuckles"], bdy_part["right_wrist"], edges, image) * self.ratio2,
            "Lb7w": max_perp(bdy_part["right_nails"], bdy_part["right_wrist"], edges, image) * self.ratio2,

            "Lj1L": np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_crotch"]) * self.ratio_bottom,
            # Not needed"Lj2L": (np.linalg.norm(body_parts_pos["left_hip"] - body_par&ts_pos["left_knee"])) / 2,
            "Lj3L": np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_knee"]) * self.ratio_bottom,
            "Lj4L": np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_maximum_calf"]) * self.ratio_bottom,
            "Lj5L": np.linalg.norm(bdy_part["left_hip"] - bdy_part["left_ankle"]) * self.ratio_bottom,
            "Lj6L": 1,
            # Not measured "Lj7L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_arch"]),
            "Lj8L": np.linalg.norm(bdy_part_tuck["right_ankle"] - bdy_part_tuck["right_ball"]) * self.ratio_tuck2,
            "Lj9L": np.linalg.norm(bdy_part_tuck["right_ankle"] - bdy_part_tuck["right_toe_nail"]) * self.ratio_tuck2,

            # Not measured "Lj0p":,
            "Lj1p": circle_p(bdy_part["crotch_width"]),
            "Lj2p": circle_p(max_perp(bdy_part["right_mid_thigh"], bdy_part["right_hip"], edges_short, image)) * self.ratio_bottom,
            "Lj3p": circle_p(max_perp(bdy_part["right_knee"], bdy_part["right_hip"], edges, image)) * self.ratio_bottom,
            "Lj4p": circle_p(max_perp(bdy_part["right_maximum_calf"], bdy_part["right_knee"], edges, image)) * self.ratio_bottom,
            "Lj5p": circle_p(max_perp(bdy_part_tuck["right_ankle"], bdy_part_tuck["right_knee"], edges_tuck,image_tuck)) * self.ratio_tuck,
            # Inversion du depth et du width car marche mieux?
            "Lj6p": stad_p(max_perp(bdy_part_r_tuck["right_ankle"], bdy_part_r_tuck["right_toe_nail"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck,
                           max_perp(bdy_part_tuck["right_ankle"], bdy_part_tuck["right_toe_nail"], edges_tuck, image_tuck) * self.ratio_tuck),
            "Lj7p": stad_p(max_perp(bdy_part_tuck["right_arch"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
                           max_perp(bdy_part_r_tuck["right_arch"], bdy_part_r_tuck["right_ankle"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck),
            "Lj8p": stad_p(max_perp(bdy_part_tuck["right_ball"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
                           max_perp(bdy_part_r_tuck["right_ball"], bdy_part_r_tuck["right_ankle"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck),
            "Lj9p": stad_p(max_perp(bdy_part_tuck["right_toe_nail"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
                           max_perp(bdy_part_r_tuck["right_ball"], bdy_part_r_tuck["right_ankle"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck),

            "Lj8w": max_perp(bdy_part_tuck["right_ball"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
            "Lj9w": max_perp(bdy_part_tuck["right_toe_nail"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,

            "Lj6d": max_perp(bdy_part_r_tuck["right_ankle"], bdy_part_r_tuck["right_knee"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck,

            "Lk1L": get_length(bdy_part["right_hip"], bdy_part["right_crotch"], image) * self.ratio_bottom,
            # Not measured "Lk2L"
            "Lk3L": get_length(bdy_part["right_hip"], bdy_part["right_knee"], image) * self.ratio_bottom,
            "Lk4L": get_length(bdy_part["right_hip"], bdy_part["right_maximum_calf"], image) * self.ratio_bottom,
            "Lk5L": get_length(bdy_part["right_hip"], bdy_part["right_ankle"], image) * self.ratio_bottom,
            "Lk6L": 1,
            # Not measured "Lk7L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_arch"]),
            "Lk8L": get_length(bdy_part_tuck["right_ankle"], bdy_part_tuck["right_ball"], image_tuck) * self.ratio_tuck2,
            "Lk9L": get_length(bdy_part_tuck["right_ankle"], bdy_part_tuck["right_toe_nail"], image_tuck) * self.ratio_tuck2,

            # Not measured "Lk0p":,
            "Lk1p": circle_p(bdy_part["crotch_width"]),
            "Lk2p": circle_p(max_perp(bdy_part["right_mid_thigh"], bdy_part["right_hip"], edges_short, image)) * self.ratio_bottom,
            "Lk3p": circle_p(max_perp(bdy_part["right_knee"], bdy_part["right_hip"], edges, image)) * self.ratio_bottom,
            "Lk4p": circle_p(max_perp(bdy_part["right_maximum_calf"], bdy_part["right_knee"], edges, image)) * self.ratio_bottom,
            "Lk5p": circle_p(max_perp(bdy_part_tuck["right_ankle"], bdy_part_tuck["right_knee"], edges_tuck, image_tuck)) * self.ratio_tuck,
            "Lk6p": stad_p(max_perp(bdy_part_r_tuck["right_ankle"], bdy_part_r_tuck["right_toe_nail"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck,
                           max_perp(bdy_part_tuck["right_ankle"], bdy_part_tuck["right_toe_nail"], edges_tuck, image_tuck) * self.ratio_tuck),
            "Lk7p": stad_p(max_perp(bdy_part_tuck["right_arch"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
                           max_perp(bdy_part_r_tuck["right_arch"], bdy_part_r_tuck["right_ankle"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck),
            "Lk8p": stad_p(max_perp(bdy_part_tuck["right_ball"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
                           max_perp(bdy_part_r_tuck["right_ball"], bdy_part_r_tuck["right_ankle"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck),
            "Lk9p": stad_p(max_perp(bdy_part_tuck["right_toe_nail"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
                           max_perp(bdy_part_r_tuck["right_ball"], bdy_part_r_tuck["right_ankle"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck),

            "Lk8w": max_perp(bdy_part_tuck["right_ball"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,
            "Lk9w": max_perp(bdy_part_tuck["right_toe_nail"], bdy_part_tuck["right_ankle"], edges_tuck, image_tuck) * self.ratio_tuck,

            "Lk6d": max_perp(bdy_part_r_tuck["right_ankle"], bdy_part_r_tuck["right_knee"], edges_l_tuck, image_l_tuck) * self.ratio_l_tuck,
        }
        self.pelvis = abs(bdy_part["left_hip"][1] - bdy_part["top_of_head"][1]) * self.ratio / 100
        self.knuckle = self.keypoints["Lb6L"] / 100
        self.pike_hand = np.linalg.norm(bdy_part_pike["right_knee"] - bdy_part_pike["right_hand"]) * self.ratio_pike / 100
        self.tuck_hand = abs(bdy_part_r_tuck["right_knee"][1] - bdy_part_r_tuck["right_knuckle"][1]) * self.ratio_l_tuck / 100
        #generate_yml(self.pelvis, self.knuckle, self.pike_hand, self.tuck_hand)
        save_img(image, image_r_side, image_tuck, image_l_tuck, image_pike, impath_front.split('/')[-1].split('_')[0])
        self._round_keypoints()
        self._create_txt(f"{impath_front.split('/')[-1].split('_')[0]}.txt")
        self._verify_keypoints()

    def _create_txt(self, file_name: str):
        with open(file_name, "w") as file_object:
            file_object.writelines("measurementconversionfactor: .01\n")
            for key, value in self.keypoints.items():
                if key[-1].isalpha():
                    file_object.writelines("{} : {:.1f}\n".format(key, float(value)))

    def _round_keypoints(self):
        def loop(keypoint_perim, keypoint_width, keypoint_name):
            if keypoint_perim / keypoint_width <= 2:
                print(f"{keypoint_name} is too high, so it has been decreased")
                return (keypoint_perim / 2) - 0.1
            if keypoint_perim / keypoint_width >= np.pi:
                print(f"{keypoint_name} is too low, so it has been increased")
                return (keypoint_perim / np.pi) + 0.1
            return keypoint_width

        self.keypoints["Ls0w"] = loop(self.keypoints["Ls0p"], self.keypoints["Ls0w"], "Ls0w")
        self.keypoints["Ls1w"] = loop(self.keypoints["Ls1p"], self.keypoints["Ls1w"], "Ls1w")
        self.keypoints["Ls2w"] = loop(self.keypoints["Ls2p"], self.keypoints["Ls2w"], "Ls2w")
        self.keypoints["Ls3w"] = loop(self.keypoints["Ls3p"], self.keypoints["Ls3w"], "Ls3w")

        self.keypoints["La4w"] = loop(self.keypoints["La4p"], self.keypoints["La4w"], "La4w")
        self.keypoints["La5w"] = loop(self.keypoints["La5p"], self.keypoints["La5w"], "La5w")
        self.keypoints["La6w"] = loop(self.keypoints["La6p"], self.keypoints["La6w"], "La6w")
        self.keypoints["La7w"] = loop(self.keypoints["La7p"], self.keypoints["La7w"], "La7w")

        self.keypoints["Lb4w"] = loop(self.keypoints["Lb4p"], self.keypoints["Lb4w"], "Lb4w")
        self.keypoints["Lb5w"] = loop(self.keypoints["Lb5p"], self.keypoints["Lb5w"], "Lb5w")
        self.keypoints["Lb6w"] = loop(self.keypoints["Lb6p"], self.keypoints["Lb6w"], "Lb6w")
        self.keypoints["Lb7w"] = loop(self.keypoints["Lb7p"], self.keypoints["Lb7w"], "Lb7w")

        self.keypoints["Lj6d"] = loop(self.keypoints["Lj6p"], self.keypoints["Lj6d"], "Lj6d")
        self.keypoints["Lj8w"] = loop(self.keypoints["Lj8p"], self.keypoints["Lj8w"], "Lj8w")
        self.keypoints["Lj9w"] = loop(self.keypoints["Lj9p"], self.keypoints["Lj9w"], "Lj9w")

        self.keypoints["Lk6d"] = loop(self.keypoints["Lk6p"], self.keypoints["Lk6d"], "Lk6d")
        self.keypoints["Lk8w"] = loop(self.keypoints["Lk8p"], self.keypoints["Lk8w"], "Lk8w")
        self.keypoints["Lk9w"] = loop(self.keypoints["Lk9p"], self.keypoints["Lk9w"], "Lk9w")

    def _verify_keypoints(self):
        for key, value in self.keypoints.items():
            if value > 150 or value == 0:
                print(f"There is an error on this measure: {key}, this measure should not be possible: {value}")


def main():
    args = sys.argv[1:]
    if len(args) == 5:
        yeadon = YeadonModel(args[0], args[1], args[2], args[3], args[4])
    else:
        print("Wrong arguments you should have 4 args")


if __name__ == "__main__":
    main()
