import io
import numpy as np
import PIL
from PIL import Image
import requests
import torch
import openpifpaf
import matplotlib.pyplot as plt


class YeadonModel:

    def __init__(self, impath: str):
        pil_im = PIL.Image.open(impath).convert('RGB')
        im = np.asarray(pil_im)
        predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')
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
                "right_toe_nail" : 20
                }
        body_parts_pos = { k : data[v] for k,v in body_parts_index.items()}
        hand_pos = [96,100,104,108,117,121,125,129,98,102,106,110,119,123,127,131]
        hand_part_pos = []
        for hand_position in hand_pos:
            hand_part_pos.append(data[hand_position])
        body_parts_pos["left_knuckles"] = np.mean(hand_part_pos[0:3], axis = 0)
        body_parts_pos["right_knuckles"] = np.mean(hand_part_pos[4:7], axis = 0)
        body_parts_pos["left_nails"] = np.mean(hand_part_pos[8:11], axis = 0)
        body_parts_pos["right_nails"] = np.mean(hand_part_pos[12:15], axis = 0)
        left_lowest_front_rib_approx = (data[5] + data[11])/2
        body_parts_pos["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data[6] + data[12])/2
        body_parts_pos["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos["left_nipple"] = (left_lowest_front_rib_approx + data[5]) / 2
        body_parts_pos["right_nipple"] = (right_lowest_front_rib_approx + data[6]) / 2
        body_parts_pos["umbiculus"] = ((left_lowest_front_rib_approx*3) + (data[11]*2))/5
        left_arch_approx = (data[17] +data[19]) / 2
        body_parts_pos["left_arch"] = left_arch_approx
        right_arch_approx = (data[20] +data[22]) / 2
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
                #TODO"Ls5": body_parts_pos["acromium"],
                "Ls6": body_parts_pos["nose"],
                "Ls7": body_parts_pos["left_ear"],
                #TODO"Ls8": body_parts_pos["top_of_head"],
                "La0": body_parts_pos["left_shoulder"],
                "La1": body_parts_pos["left_mid_arm"],
                "La2": body_parts_pos["left_elbow"],
                #TODO"La3": body_parts_pos["left_maximum_forearm"],
                "La4": body_parts_pos["left_wrist"],
                "La5": body_parts_pos["left_base_of_thumb"],
                "La6": body_parts_pos["left_knuckles"],
                "La7": body_parts_pos["left_nails"],
                "Lb0": body_parts_pos["right_shoulder"],
                "Lb1": body_parts_pos["right_mid_arm"],
                "Lb2": body_parts_pos["right_elbow"],
                #TODO"Lb3": body_parts_pos["right_maximum_forearm"],
                "Lb4": body_parts_pos["right_wrist"],
                "Lb5": body_parts_pos["right_base_of_thumb"],
                "Lb6": body_parts_pos["right_knuckles"],
                "Lb7": body_parts_pos["right_nails"],
                "Lj0": body_parts_pos["left_hip"],
                #TODO"Lj1": body_parts_pos["left_crotch"],
                #TODO"Lj2": body_parts_pos["left_mid_thigh"],
                "Lj3": body_parts_pos["left_knee"],
                #TODO"Lj4": body_parts_pos["left_maximum_calf"],
                "Lj5": body_parts_pos["left_ankle"],
                "Lj6": body_parts_pos["left_heel"],
                "Lj7": body_parts_pos["left_arch"],
                "Lj8": body_parts_pos["left_ball"],
                "Lj9": body_parts_pos["left_toe_nail"],
                "Lk0": body_parts_pos["right_hip"],
                #TODO"Lk1": body_parts_pos["right_crotch"],
                #TODO"Lk2": body_parts_pos["right_mid_thigh"],
                "Lk3": body_parts_pos["right_knee"],
                #TODO"Lk4": body_parts_pos["right_maximum_calf"],
                "Lk5": body_parts_pos["right_ankle"],
                "Lk6": body_parts_pos["right_heel"],
                "Lk7": body_parts_pos["right_arch"],
                "Lk8": body_parts_pos["right_ball"],
                "Lk9": body_parts_pos["right_toe_nail"]
                }


if __name__ == '__main__':
    pass
