import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse
import os
from shapely.geometry import Polygon
import shapely
    
def Homography(img_1, img_2):
    img1 = cv.imread(cv.samples.findFile(img_1))
    img2 = cv.imread(cv.samples.findFile(img_2))

    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # print(len(src_pts))
    # print("dst_pts: " + str(dst_pts))
    if len(src_pts) < 4:
        return np.zeros([4, 4, 4], dtype=np.uint8)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w,c = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # print("M type: " + str(type(M)))
    # print("pts: " + str(pts))
    if M is None:
        return np.zeros([4, 4, 4], dtype=np.uint8)
    dst = cv.perspectiveTransform(pts,M)
    # print("dst: " + str(dst))
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    # cv.imwrite(img_2.replace(".jpg", "_bounds.jpg"), img2)


    return M


def Looking(coords, M, frame):
    c_1 = coords[1].split(',')
    frame_conv = (((((float(len(coords)) - 1)) / (float(c_1[6]) / 1000)) / 30) * float(frame)) + 1
    line = coords[int(frame_conv)].split(',')
    if line[7] != '' and M.all() != 0:
        # print("og coords: " + line[7] + ', ' + line[8])
        np_array = np.array([[[float(line[7]), float(line[8])]], [[float(line[7]) + 1, float(line[8])]], [[float(line[7]), float(line[8]) + 1]], [[float(line[7]) + 1, float(line[8]) + 1]]])
        new_coords = cv.perspectiveTransform(np_array, M)

        ret = str(frame) + ',' + str(line[7]) + ',' + str(line[8]) + ',' + str(new_coords[0][0][0]) + ',' + str(new_coords[0][0][1])
    else:
        ret = "N/A"
    return ret



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tobii_dir",type=str,required=True)
    parser.add_argument("--video_dir_left",type=str,required=True)
    parser.add_argument("--video_dir_right",type=str,required=True)

    parser.add_argument("--tobii_coords",type=str,required=True)

    args = parser.parse_args()

    tobii_dir = args.tobii_dir
    tobii_frames = os.listdir(tobii_dir)
    tobii_frames.sort()

    video_dir_l = args.video_dir_left
    video_frames_l = os.listdir(video_dir_l)
    video_frames_l.sort()

    video_dir_r = args.video_dir_right
    video_frames_r = os.listdir(video_dir_r)
    video_frames_r.sort()

    eye_coords = args.tobii_coords
    coords = open(eye_coords, 'r')
    coords_lines = coords.readlines()

    fileL = open(os.path.join(video_dir_l, "homo_coords_left.csv"), 'w')
    fileR = open(os.path.join(video_dir_r, "homo_coords_right.csv"), 'w')

    fileL.write("Frame,Side,X Coord,Y Coord\n")
    fileR.write("Frame,Side,X Coord,Y Coord\n")

    for i in range(0, len(video_frames_l)):
        looking_transform_left = ""
        looking_transform_right = ""
        print("tf: " + str(tobii_frames[i]))
        left_M = Homography(os.path.join(tobii_dir, tobii_frames[i]), os.path.join(video_dir_l, video_frames_l[i]))
        if left_M.all() != 0:
            looking_transform_left = Looking(coords_lines, left_M, tobii_frames[i].replace(".jpg", ""))

            if looking_transform_left != "N/A":
                point = looking_transform_left.split(',')
                # print("points left: " + str(point[3]) + ', ' + str(point[4]))
                fileL.write(str(i) + ',Left,' + point[3] + ',' + point[4] + '\n')
                
                img = cv.imread(cv.samples.findFile(os.path.join(video_dir_l, video_frames_l[i])))
                image = cv.circle(img, [int(float(point[3])), int(float(point[4]))], radius=10, color=(255,0,0), thickness=5)
                # cv.imwrite(os.path.join(video_dir_l, video_frames_l[i].replace(".jpg", "_left_coord.jpg")), image)

                # cv.imshow("image left", image)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

        right_M = Homography(os.path.join(tobii_dir, tobii_frames[i]), os.path.join(video_dir_r, video_frames_r[i]))
        if right_M.all() != 0:
            looking_transform_right = Looking(coords_lines, right_M, tobii_frames[i].replace(".jpg", ""))

            if looking_transform_right != "N/A":
                point = looking_transform_right.split(',')
                # print("points right: " + str(point[3]) + ', ' + str(point[4]))
                fileR.write(str(i) + ',Right,' + point[3] + ',' + point[4] + '\n')
                
                img_right = cv.imread(cv.samples.findFile(os.path.join(video_dir_r, video_frames_r[i])))
                image_r = cv.circle(img_right, [int(float(point[3])), int(float(point[4]))], radius=10, color=(255,0,0), thickness=5)
                # cv.imwrite(os.path.join(video_dir_r, video_frames_r[i].replace(".jpg", "_right_coord.jpg")), image_r)

                # cv.imshow("image right", image_r)
                # cv.waitKey(0)
                # cv.destroyAllWindows()
        
        if looking_transform_left == "N/A" and looking_transform_right == "N/A":
            fileL.write(str(i) + ',Neither,-1,-1\n')
            fileR.write(str(i) + ',Neither,-1,-1\n')
        #     print(":(")

