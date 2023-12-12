import numpy as np
import cv2 as cv


def pt_from(origin: np.ndarray, angle: int, distance):
    """
    Compute the point [x, y] that is 'distance' apart from the origin point perpendicular.
    """
    x = origin[1] + np.sin(angle) * distance
    y = origin[0] + np.cos(angle) * distance
    return np.array([int(y), int(x)])


def find_edge(p1: np.ndarray, angle_radians, edges: np.ndarray, save):
    """
    Find the edge pixel given a starting point, angle, and image edges.
    """
    distance = 0
    while True:
        x, y = pt_from(p1, angle_radians, distance)
        if not (0 <= x < edges.shape[0] and 0 <= y < edges.shape[1]):
            break

        hit_zone = edges[x, y] == 255
        if np.any(hit_zone):
            save.append((y, x))
            break
        distance += 0.01
    return save


def get_points(start: np.ndarray, end: np.ndarray):
    """
    Get two points and the vector between them.
    """
    p1, p2 = np.array([start[1], start[0]]), np.array([end[1], end[0]])
    return p1, p2, np.array([p2[0] - p1[0], p2[1] - p1[1]])


def get_max_approx(top_arr, bottom_arr):
    """
    Get the index of the maximum norm in two arrays.
    """
    if len(top_arr) != len(bottom_arr):
        print("Error: Not the same number of points")
        return

    vector = np.array(top_arr) - np.array(bottom_arr)
    norms = np.linalg.norm(vector, axis=1)
    max_norm = norms[0]
    save_index = 0

    for i in range(len(top_arr)):
        if norms[i] >= max_norm and norms[i] <= norms[0] * 1.5:
            max_norm = norms[i]
            save_index = i

    return save_index


def vector_angle(vector: np.ndarray, plus: int):
    """
    Calculate the angle of a vector with an optional offset.
    """
    angle = np.arctan2(vector[1], vector[0])
    return angle + np.pi / 2 if plus else angle - np.pi / 2


def get_maximum_range(angle_radians, result, edges):
    """
    Get the range of points along a given angle within image boundaries.
    """
    save = []

    for point in result:
        distance = 0
        while True:
            x, y = pt_from(point, angle_radians, distance)
            if not (0 <= x < edges.shape[0] and 0 <= y < edges.shape[1]):
                break

            hit_zone = edges[x, y] == 255
            if np.any(hit_zone):
                save.append((y, x))
                break

            distance += 0.01
    return save


def _get_maximum(start: np.ndarray, end: np.ndarray, edges: np.ndarray, img, angle, is_start: int):
    """
    Get the maximum distance between two edges along a specific angle.
    """
    p1, p2, vector = get_points(start, end)
    angle_radians = np.arctan2(vector[1], vector[0]) - angle
    max1 = find_edge(p1, angle_radians, edges, save=[])
    angle_radians = (np.arctan2(vector[1], vector[0]) + angle) * is_start
    max2 = find_edge(p1, angle_radians, edges, save=[])
    cv.line(img, max1[0], max2[0], (0, 0, 255), 1)
    return np.linalg.norm(np.array(max1) - np.array(max2))


def max_line(start: np.ndarray, end: np.ndarray, edges: np.ndarray, img: np.ndarray):
    """
    Calculate the maximum distance between two edges along a straight line.
    """
    return _get_maximum(start, end, edges, img, 0, -1)


def max_perp(start: np.ndarray, end: np.ndarray, edges: np.ndarray, img: np.ndarray):
    """
    Calculate the maximum perpendicular distance between two edges.
    """
    return _get_maximum(start, end, edges, img, np.pi / 2, 1)


def get_max_pt(start, end, edges):
    """
    Get the point where the maximum perpendicular distance occurs between two edges.
    """
    p1, p2, vector = get_points(start, end)
    # create an array with 100 points between start and end
    x_values = np.linspace(p1[1], p2[1], 100)
    y_values = np.linspace(p1[0], p2[0], 100)
    result = [(y, x) for x, y in zip(x_values, y_values)]
    # set the angle in the direction of the edges
    angle_radians = vector_angle(vector, 1)
    r_side = get_maximum_range(angle_radians, result, edges)
    # set the angle in the direction of other side
    angle_radians = vector_angle(vector, 0)
    l_side = get_maximum_range(angle_radians, result, edges)
    # get the index of the max
    index = get_max_approx(r_side, l_side)
    return result[index][::-1]


def get_maximum_pit(start: np.ndarray, edges: np.ndarray):
    """
    Get the maximum perpendicular distance along both sides of a point.
    """
    point = np.array([start[1], start[0]])
    point2 = np.array([point[0] + 5, point[1]])
    vector = np.array([point2[0] - point[0], point2[1] - point[1]])
    angle_radians = np.arctan2(vector[1], vector[0])
    max_save = find_edge(point, angle_radians, edges, save=[])
    angle_radians_left = angle_radians
    angle_radians_right = angle_radians
    max_save_right = max_save
    max_save_left = max_save

    while True:
        angle_radians_left += 0.01
        max_last_right = find_edge(point, angle_radians_left, edges, save=[])
        if np.linalg.norm(max_last_right) < np.linalg.norm(max_save_right):
            break
        max_save_right = max_last_right

    while True:
        angle_radians_right -= 0.01
        max_last_left = find_edge(point, angle_radians_right, edges, save=[])
        if np.linalg.norm(max_last_left) < np.linalg.norm(max_save_left):
            break
        max_save_left = max_last_left

    distance = np.linalg.norm(max(max_save_left, max_last_right))
    point = max(max_save_left, max_last_right)

    return point[0], int(distance)


def get_length(point1: np.ndarray, point2: np.ndarray, image: np.ndarray):
    length = np.linalg.norm(point1 - point2)
    cv.line(image, point1.astype(int), point2.astype(int), (0, 0, 255), 1)
    return length
