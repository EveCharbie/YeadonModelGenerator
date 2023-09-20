import cv2 as cv
import numpy as np
import openpifpaf
import PIL
from rembg import remove

RESIZE_SIZE = 600  # the maximum size of the image to be processed (in pixels)


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
        #front
        pil_im, im = self._create_resize_remove_im(impath)
        edges = self._canny_edges(im)
        predictor = openpifpaf.Predictor(checkpoint="shufflenetv2k30-wholebody")
        predictions, gt_anns, image_meta = predictor.pil_image(pil_im)
        # You can find the index here:
        # https://github.com/jin-s13/COCO-WholeBody/blob/master/imgs/Fig2_anno.png
        # as "predictions" is an array the index starts at 0 and not at 1 like in the github
        data = predictions[0].data[:,0:2]
        #right side
        pil_r_side_im, im_r_side = self._create_resize_remove_im("img/green_side.jpg")
        edges_r_side = self._canny_edges(im_r_side)
        predictions2, gt_anns2, image_meta2 = predictor.pil_image(pil_r_side_im)
        data_r_side = predictions2[0].data[:,0:2]

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
            "right_toe_nail": 20,
        }
        #front
        body_parts_pos = {k: data[v] for k, v in body_parts_index.items()}
        #right side
        body_parts_pos_r_side = {k: data_r_side[v] for k, v in body_parts_index.items()}
        hand_pos = [
            96,
            100,
            104,
            108,
            117,
            121,
            125,
            129,
            98,
            102,
            106,
            110,
            119,
            123,
            127,
            131,
        ]
        #front
        hand_part_pos = []
        for hand_position in hand_pos:
            hand_part_pos.append(data[hand_position])
        
        body_parts_pos["left_knuckles"] = np.mean(hand_part_pos[0:3], axis=0)
        body_parts_pos["right_knuckles"] = np.mean(hand_part_pos[4:7], axis=0)
        body_parts_pos["left_nails"] = np.mean(hand_part_pos[8:11], axis=0)
        body_parts_pos["right_nails"] = np.mean(hand_part_pos[12:15], axis=0)
        left_lowest_front_rib_approx = (data[5] + data[11]) / 2
        body_parts_pos["left_lowest_front_rib"] = left_lowest_front_rib_approx
        right_lowest_front_rib_approx = (data[6] + data[12]) / 2
        body_parts_pos["right_lowest_front_rib"] = right_lowest_front_rib_approx
        body_parts_pos["left_nipple"] = (left_lowest_front_rib_approx + data[5]) / 2
        body_parts_pos["right_nipple"] = (right_lowest_front_rib_approx + data[6]) / 2
        body_parts_pos["umbiculus"] = (
            (left_lowest_front_rib_approx * 3) + (data[11] * 2)
        ) / 5
        left_arch_approx = (data[17] + data[19]) / 2
        body_parts_pos["left_arch"] = left_arch_approx
        right_arch_approx = (data[20] + data[22]) / 2
        body_parts_pos["right_arch"] = right_arch_approx
        body_parts_pos["left_ball"] = (data[17] + left_arch_approx) / 2
        body_parts_pos["right_ball"] = (data[20] + right_arch_approx) / 2
        body_parts_pos["left_mid_arm"] = (data[5] + data[7]) / 2
        body_parts_pos["right_mid_arm"] = (data[6] + data[8]) / 2
        body_parts_pos["acromion"] = self._find_acromion(im, data)
        body_parts_pos["top_of_head"] = self._find_top_of_head(edges)
        body_parts_pos["right_maximum_forearm"] = self._get_maximum(data[10], data[8], edges)
        body_parts_pos["left_maximum_forearm"] = self._get_maximum(data[9], data[7], edges)
        body_parts_pos["right_maximum_calf"] = self._get_maximum(data[16], data[14], edges)
        body_parts_pos["left_maximum_calf"] = self._get_maximum(data[15], data[13], edges)
        body_parts_pos["right_crotch"],body_parts_pos["left_crotch"] = self._get_crotch_right_left(im, data)
        body_parts_pos["right_mid_thigh"],body_parts_pos["left_mid_thigh"] = self._get_mid_thigh_right_left(data, body_parts_pos["right_crotch"],body_parts_pos["right_crotch"])

        #right side
        hand_part_pos_r_side = []
        for hand_position in hand_pos:
            hand_part_pos_r_side.append(data_r_side[hand_position])
        self.keypoints = {
            "Ls0": body_parts_pos["left_hip"],
            "Ls1": body_parts_pos["umbiculus"],
            "Ls2": body_parts_pos["left_lowest_front_rib"],
            "Ls3": body_parts_pos["left_nipple"],
            "Ls4": body_parts_pos["left_shoulder"],
            "Ls5": body_parts_pos["acromion"],
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
            "Ls1L": abs(body_parts_pos["umbiculus"][1] - body_parts_pos["left_hip"][1]),
            "Ls2L": abs(
                body_parts_pos["left_lowest_front_rib"][1]
                - body_parts_pos["left_hip"][1]
            ),
            "Ls3L": abs(
                body_parts_pos["left_nipple"][1] - body_parts_pos["left_hip"][1]
            ),
            "Ls4L": abs(
                body_parts_pos["left_shoulder"][1] - body_parts_pos["left_hip"][1]
            ),
            "Ls5L": abs(body_parts_pos["acromion"][1] - body_parts_pos["left_hip"][1]),
            "Ls6L": abs(body_parts_pos["acromion"][1] - body_parts_pos["nose"][1]),
            "Ls7L": abs(body_parts_pos["acromion"][1] - body_parts_pos["left_ear"][1]),
            "Ls8L": abs(
                body_parts_pos["acromion"][1] - body_parts_pos["top_of_head"][1]
            ),

            # TODO "Ls0p":,
            # TODO "Ls1p":,
            # TODO "Ls2p":,
            # TODO "Ls3p":,
            # TODO "Ls5p":,
            # TODO "Ls6p":,
            # TODO "Ls7p":,

            "Ls0w": self._get_maximum_line(body_parts_pos["left_hip"], body_parts_pos["right_hip"],edges),
            # TODO"Ls1w": self._get_maximum_line(body_parts_pos["umbiculus"], body_parts_pos["umbiculus"],edges),
            "Ls2w": self._get_maximum_line(body_parts_pos["left_lowest_front_rib"], body_parts_pos["right_lowest_front_rib"],edges),
            "Ls3w": self._get_maximum_line(body_parts_pos["left_nipple"], body_parts_pos["right_nipple"],edges),
            "Ls4w": self._get_maximum_line(body_parts_pos["left_shoulder"], body_parts_pos["right_shoulder"],edges),

            "La1L": (np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_elbow"]))/2,
            "La2L": np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_elbow"]),
            "La3L": np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_maximum_forearm"]),
            "La4L": np.linalg.norm(body_parts_pos["left_shoulder"] - body_parts_pos["left_wrist"]),
            "La5L": np.linalg.norm(body_parts_pos["left_wrist"] - body_parts_pos["left_base_of_thumb"]),
            "La6L": np.linalg.norm(body_parts_pos["left_wrist"] - body_parts_pos["left_knuckles"]),
            "La7L": np.linalg.norm(body_parts_pos["left_wrist"] - body_parts_pos["left_nails"]),

            # TODO "La0p":,
            # TODO "La1p":,
            # TODO "La2p":,
            # TODO "La3p":,
            # TODO "La4p":,
            # TODO "La5p":,
            # TODO "La6p":,
            # TODO "La7p":,

            "La4w": self._get_maximum_start(body_parts_pos["left_wrist"], body_parts_pos["left_elbow"], edges),
            # TODO "La5w": self._get_maximum_start(body_parts_pos["left_base_of_thumb"], body_parts_pos["left_wrist"], edges),
            "La6w": self._get_maximum_start(body_parts_pos["left_knuckles"], body_parts_pos["left_wrist"], edges),
            "La7w": self._get_maximum_start(body_parts_pos["left_nails"], body_parts_pos["left_wrist"], edges),

            "Lb1L": (np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_elbow"]))/2,
            "Lb2L": np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_elbow"]),
            "Lb3L": np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_maximum_forearm"]),
            "Lb4L": np.linalg.norm(body_parts_pos["right_shoulder"] - body_parts_pos["right_wrist"]),
            "Lb5L": np.linalg.norm(body_parts_pos["right_wrist"] - body_parts_pos["right_base_of_thumb"]),
            "Lb6L": np.linalg.norm(body_parts_pos["right_wrist"] - body_parts_pos["right_knuckles"]),
            "Lb7L": np.linalg.norm(body_parts_pos["right_wrist"] - body_parts_pos["right_nails"]),

            # TODO "Lb0p":,
            # TODO "Lb1p":,
            # TODO "Lb2p":,
            # TODO "Lb3p":,
            # TODO "Lb4p":,
            # TODO "Lb5p":,
            # TODO "Lb6p":,
            # TODO "Lb7p":,

            "Lb4w": self._get_maximum_start(body_parts_pos["right_wrist"], body_parts_pos["right_elbow"], edges),
            # TODO "Lb5w": self._get_maximum_start(body_parts_pos["right_base_of_thumb"], body_parts_pos["right_wrist"], edges),
            "Lb6w": self._get_maximum_start(body_parts_pos["right_knuckles"], body_parts_pos["right_wrist"], edges),
            "Lb7w": self._get_maximum_start(body_parts_pos["right_nails"], body_parts_pos["right_wrist"], edges),

            "Lj1L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_crotch"]),
            "Lj2L": (np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_knee"]))/2,
            "Lj3L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_knee"]),
            "Lj4L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_maximum_calf"]),
            "Lj5L": np.linalg.norm(body_parts_pos["left_hip"] - body_parts_pos["left_ankle"]),
            "Lj6L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_heel"]),
            "Lj7L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_arch"]),
            "Lj8L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_ball"]),
            "Lj9L": np.linalg.norm(body_parts_pos["left_ankle"] - body_parts_pos["left_toe_nail"]),

            # TODO "Lj0p":,
            # TODO "Lj1p":,
            # TODO "Lj2p":,
            # TODO "Lj3p":,
            # TODO "Lj4p":,
            # TODO "Lj5p":,
            # TODO "Lj6p":,
            # TODO "Lj7p":,
            # TODO "Lj8p":,
            # TODO "Lj9p":,

            # TODO "Lj8w":,
            # TODO "Lj9w":,

            # TODO "Lj6d":,

            "Lk1L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_crotch"]),
            "Lk2L": (np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_knee"]))/2,
            "Lk3L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_knee"]),
            "Lk4L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_maximum_calf"]),
            "Lk5L": np.linalg.norm(body_parts_pos["right_hip"] - body_parts_pos["right_ankle"]),
            "Lk6L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_heel"]),
            "Lk7L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_arch"]),
            "Lk8L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_ball"]),
            "Lk9L": np.linalg.norm(body_parts_pos["right_ankle"] - body_parts_pos["right_toe_nail"]),

            # TODO "Lk0p":,
            # TODO "Lk1p":,
            # TODO "Lk2p":,
            # TODO "Lk3p":,
            # TODO "Lk4p":,
            # TODO "Lk5p":,
            # TODO "Lk6p":,
            # TODO "Lk7p":,
            # TODO "Lk8p":,
            # TODO "Lk9p":,

            # TODO "Lk8w":,
            # TODO "Lk9w":,

            # TODO "Lk6d":,
        }
        print(self.keypoints)
    def _get_maximum_line(self,start, end, edges):
        def pt_from(origin, angle, distance):
            """
            compute the point [x, y] that is 'distance' apart from the origin point
            perpendicular
            """
            x = origin[1] + np.sin(angle) * distance
            y = origin[0] + np.cos(angle) * distance
            return np.array([int(y),int(x)])
        def find_edge(p1,p2,angle_radians):
            distance = 0
            save = []
            while True:
                #as we want the width of the "start", we choose p1
                x,y = pt_from(p1, angle_radians, distance)
                if x < 0 or x >= edges.shape[0] or y < 0 or y >= edges.shape[1]:
                    break
                hit_zone = edges[x,y] == 255
                if np.any(hit_zone):
                    save.append((y,x))
                    break

                distance += 0.01
            return save
        def get_points(start, end):
            p1 = start[0:2]
            p1 = np.array([p1[1],p1[0]])
            p2 = end[0:2]
            p2 = np.array([p2[1],p2[0]])
            return np.array([p1,p2])
        p1,p2 = get_points(start,end)
        vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        angle_radians = np.arctan2(vector[1],vector[0])
        max1 = find_edge(p1,p2,angle_radians)
        angle_radians = -np.arctan2(vector[1],vector[0])
        max2 = find_edge(p1,p2,angle_radians)
        return np.linalg.norm(np.array(max1) - np.array(max2))
    def _get_maximum_start(self, start, end, edges):
        def pt_from(origin, angle, distance):
            """
            compute the point [x, y] that is 'distance' apart from the origin point
            perpendicular
            """
            x = origin[1] + np.sin(angle) * distance
            y = origin[0] + np.cos(angle) * distance
            return np.array([int(y),int(x)])
        def find_edge(p1,p2,angle_radians):
            distance = 0
            save = []
            while True:
                #as we want the width of the "start", we choose p1
                x,y = pt_from(p1, angle_radians, distance)
                if x < 0 or x >= edges.shape[0] or y < 0 or y >= edges.shape[1]:
                    break
                hit_zone = edges[x,y] == 255
                if np.any(hit_zone):
                    save.append((y,x))
                    break

                distance += 0.01
            return save
        def get_points(start, end):
            p1 = start[0:2]
            p1 = np.array([p1[1],p1[0]])
            p2 = end[0:2]
            p2 = np.array([p2[1],p2[0]])
            return np.array([p1,p2])
        p1,p2 = get_points(start,end)
        vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        angle_radians = np.arctan2(vector[1],vector[0]) - np.pi/2
        max1 = find_edge(p1,p2,angle_radians)
        angle_radians = np.arctan2(vector[1],vector[0]) + np.pi/2
        max2 = find_edge(p1,p2,angle_radians)
        return np.linalg.norm(np.array(max1) - np.array(max2))

    def _get_maximum(self,start, end, edges):
        def pt_from(origin, angle, distance):
            """
            compute the point [x, y] that is 'distance' apart from the origin point
            perpendicular
            """
            x = origin[1] + np.sin(angle) * distance
            y = origin[0] + np.cos(angle) * distance
            return np.array([int(y),int(x)])
        def get_max_approx(top_arr, bottom_arr):
            if len(top_arr) != len(bottom_arr):
                print("error not the same nbr of pts")
                return
            vector = np.array(top_arr) - np.array(bottom_arr)
            norms = np.linalg.norm(vector, axis=1)
            maximum = np.max(norms)
            return maximum
        def get_points(start, end):
            p1 = start[0:2]
            p1 = np.array([p1[1],p1[0]])
            p2 = end[0:2]
            p2 = np.array([p2[1],p2[0]])
            return np.array([p1,p2])

        def vector_angle_plus(p1,p2):
            vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            angle_radians = np.arctan2(vector[1],vector[0]) + np.pi/2
            return angle_radians

        def vector_angle_minus(p1,p2):
            vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            angle_radians = np.arctan2(vector[1],vector[0]) - np.pi/2
            return angle_radians

        def get_maximum_range(p1,p2, angle_radians):
            x_values = np.linspace(p1[1],p2[1], 100)
            y_values = np.linspace(p1[0],p2[0], 100)
            result = [(y, x) for x, y in zip(x_values, y_values)]

            save = []
            #for all the points between p1 and p2
            for point in result:
                distance = 0
                while True:

                    x,y = pt_from(point, angle_radians, distance)
                    if x < 0 or x >= edges.shape[0] or y < 0 or y >= edges.shape[1]:
                        break

                    # Check if we've found an edge pixel
                    #hit_zone = edges[y-1:y+2, x-1:x+2] == 255# 3 x 3
                    hit_zone = edges[x,y] == 255
                    if np.any(hit_zone):
                        save.append((y,x))
                        break

                    distance += 0.01
            return save

        #get the maximums for calf and forearm
        p1,p2 = get_points(start, end)
        #set the angle in the direction of the edges
        angle_radians = vector_angle_plus(p1,p2)
        r_side = get_maximum_range(p1,p2, angle_radians)
        #set the angle in the direction of other side
        angle_radians = vector_angle_minus(p1,p2)
        l_side = get_maximum_range(p1,p2, angle_radians)
        #find the max distance of all points
        maximum = get_max_approx(r_side, l_side)
        return maximum

    def _get_crotch_right_left(self, edges, data):
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
            return edges[y1:y2, x1:x2].copy()
        #crop the image to see the right hip to the left knee
        crotch_zone = crop(edges, data[12], data[13])
        #now the cropped image only has the crotch as an edge so we can get it like the head
        crotch_approx_crop = np.where(crotch_zone != 0)[1][0], np.where(crotch_zone != 0)[0][1]
        crotch_approx_right = np.array([data[12][0],data[12][1] +  crotch_approx_crop[1]])
        crotch_approx_left = np.array([data[11][0],data[11][1] +  crotch_approx_crop[1]])
        return crotch_approx_right, crotch_approx_left

    def _get_mid_thigh_right_left(self, data, r_crotch, l_crotch):
        mid_thigh_right = (data[14][0:2] + r_crotch)/2
        mid_thigh_left = (data[13][0:2] + l_crotch)/2
        return mid_thigh_right, mid_thigh_left

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
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        blurred = cv.medianBlur(gray, 5)
        thresh = cv.adaptiveThreshold(
            blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
        )
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
        crop_offset = [min(r_shoulder[1], r_ear[1]), min(r_shoulder[0], r_ear[0])]
        acromion_y, acromion_x = (
            np.where(shoulder2ear == 0)[0][0],
            shoulder2ear.shape[1] / 2,
        )
        acromion = np.array([acromion_y + crop_offset[0], acromion_x + crop_offset[1]])

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
        radius = depth/2
        a = width - depth
        perimeter = 2 * (np.pi * radius + a)
        return perimeter
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

    def _canny_edges(self, im):
        grayscale_image = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(grayscale_image, 10, 100)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

        # apply the dilation operation to the edges
        dilate = cv.dilate(edges, kernel, iterations=1)

        # find the contours in the dilated image
        contours, _ = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        edges = np.zeros(im.shape)
        cv.drawContours(edges, contours, -1, (0, 255, 0), 2)
        return edges

    def _create_resize_remove_im(self, impath):
        pil_im = PIL.Image.open(impath).convert("RGB")
        pil_im = self._resize(pil_im)
        im = np.asarray(pil_im)
        im = remove(im)
        return pil_im, im
if __name__ == "__main__":
    pass
