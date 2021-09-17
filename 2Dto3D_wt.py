#!/usr/bin/python
# -*- coding: UTF-8 -*-

# @Time    : 2020-07-16 09:01
# @Author  : WangCong
# @Email   : iwangcong@outlook.com
# https://github.com/cong/2Dto3D
# https://wangcong.net/article/UAVCoordinateSystemSolving.html


import numpy as np
import cv2

camera_parameter = {
    # R
    #"R": [[-0.91536173, 0.40180837, 0.02574754],
    #      [0.05154812, 0.18037357, -0.98224649],
    #      [-0.39931903, -0.89778361, -0.18581953]],
    "R": [[ 0.92211801,  0.30637627,  0.23628787],
            [-0.38652863,  0.75653205,  0.52749869],
            [-0.01714626, -0.57774807,  0.81603503]],
    # T
    #"T": [1841.10702775, 4955.28462345, 1563.4453959],
    "T": [-86.64825885, 73.65441705,  491.6951738],
    # f/dx, f/dy
    #"f": [1145.04940459, 1143.78109572],
    "f": [3.05703945e+03, 3.05716750e+03],
    # center point
    #"c": [512.54150496, 515.45148698]
    "c": [1.51989445e+03, 2.00678183e+03]
}


def pixel_to_world(camera_intrinsics, r, t, img_points):

    K_inv = camera_intrinsics.I
    R_inv = np.asmatrix(r).I
    R_inv_T = np.dot(R_inv, np.asmatrix(t))
    world_points = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_point in img_points:
        coords[0] = img_point[0]
        coords[1] = img_point[1]
        coords[2] = 1.0
        cam_point = np.dot(K_inv, coords)
        cam_R_inv = np.dot(R_inv, cam_point)
        scale = R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(R_inv_T)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0] = world_point[0]
        pt[1] = world_point[1]
        pt[2] = 0
        world_points.append(pt.T.tolist())

    return world_points


if __name__ == '__main__':
    f = camera_parameter["f"]
    c = camera_parameter["c"]
    camera_intrinsic = np.mat(np.zeros((3, 3), dtype=np.float64))
    camera_intrinsic[0, 0] = f[0]
    camera_intrinsic[1, 1] = f[1]
    camera_intrinsic[0, 2] = c[0]
    camera_intrinsic[1, 2] = c[1]
    camera_intrinsic[2, 2] = np.float64(1)
    r = camera_parameter["R"]
    t = np.asmatrix(camera_parameter["T"]).T
    # img_points = [[100, 200],
    #               [150, 300]]
    img_points = np.array(([976.2462, 2467.331],
                           [1119.2909, 2407.677]), dtype=np.double)
    result = pixel_to_world(camera_intrinsic, r, t, img_points)
    print('\n\n ---- pixel_to_world')
    print(result)
    
    print('\n\n ---- world_to_pixel')
    #axis = np.float32([[7700, 73407, 0], [-66029, -605036, 0]])
    axis = np.float32([[0, 0, 0], [24.5, 0, 0]])
    r2 = np.asmatrix(camera_parameter["R"])
    result2, _ = cv2.projectPoints(axis, r2, t, camera_intrinsic, 0)
    print(result2)
