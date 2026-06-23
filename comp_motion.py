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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir_left",type=str,required=True)
    parser.add_argument("--l_list",type=str,required=True)
    parser.add_argument("--r_list",type=str,required=True)
    parser.add_argument("--video_dir_right",type=str,required=True)

    args = parser.parse_args()

    l_list = args.l_list
    l = open(l_list, 'r')
    l_header = l.readline()
    l_lines = l.readlines()

    r_list = args.r_list
    r = open(r_list, 'r')
    r_header = r.readline()
    r_lines = r.readlines()

    video_dir_l = args.video_dir_left
    video_frames_l = os.listdir(video_dir_l)
    video_frames_l.sort()

    video_dir_r = args.video_dir_right
    video_frames_r = os.listdir(video_dir_r)
    video_frames_r.sort()


    file_L = open(os.path.join(video_dir_l, "compare_l.csv"), 'w')
    file_L.write("Frame,ID,Min X,Min Y,Max X,Max Y,Speed\n")

    file_R = open(os.path.join(video_dir_r, "compare_r.csv"), 'w')
    file_R.write("Frame,ID,Min X,Min Y,Max X,Max Y,Speed\n")

    designation = 0

    first_frame = video_frames_l[0]
    for i in range(0, len(video_frames_l)):
        print("l frame: " + str(video_frames_l[i]))
        # l_li = l_lines[i].split(',')
        # r_li = r_lines[i].split(',')
        # print("r li: " + str(r_li))
        M = Homography(os.path.join(video_dir_r, video_frames_r[i]), os.path.join(video_dir_l, video_frames_l[i]))
        # frame_list = [l_li for o in l_li if o[0]==str(i)]
        frame_list_r = []
        frame_list_l = []
        for x1 in r_lines:
            x = x1.split(',')
            if int(x[0])==int(video_frames_l[i].replace(".jpg","")):
                # print("x1: " + str(x1))
                frame_list_r.append(x)
        print("frame list r: " + str(frame_list_r))

        for n1 in l_lines:
            n = n1.split(',')
            if int(n[0])==int(video_frames_l[i].replace(".jpg","")):
                frame_list_l.append(n)
        # print("r lines: " + str(r_lines))
        print("frame list l: " + str(frame_list_l))

        for l_li in frame_list_l:
            # l_li = l.split(',')
            # print("l: " + str(l_li))
            if len(l_li) == 7:
        # li = line.split(",")
                src = np.float32([[float(l_li[2].replace(')','(').split('(')[1]), float(l_li[3].replace(')','(').split('(')[1])], 
                                    [float(l_li[2].replace(')','(').split('(')[1]), float(l_li[5].replace(')','(').split('(')[1])], 
                                    [float(l_li[4].replace(')','(').split('(')[1]), float(l_li[5].replace(')','(').split('(')[1])], 
                                    [float(l_li[4].replace(')','(').split('(')[1]), float(l_li[3].replace(')','(').split('(')[1])]]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(src,M)
                # print("dst: " + str(dst[0][0]))
                dst_box = shapely.box(dst[0][0][0], dst[0][0][1], dst[2][0][0], dst[2][0][1])
                # print("dst_box: " + str(dst_box))

                desig = 0
                a = 100000
                found = False
                print("here")
                for fi in frame_list_r:
                    print("r frame: " + fi[0])
                    found = False
                    # fi = f.split(',')
                    
                    f_box = shapely.box(float(fi[2].replace(')','(').split('(')[1]), float(fi[3].replace(')','(').split('(')[1]), 
                                        float(fi[4].replace(')','(').split('(')[1]), float(fi[5].replace(')','(').split('(')[1]))
                    # print("f box: " + str(f_box))
                    intersection = shapely.intersection(dst_box, f_box)
                                    
                    inter_area = intersection.area
                    # print("inter_area: " + str(inter_area))
                    if dst_box.area != 0 and inter_area != 0:
                        # print("here")
                        if(abs((inter_area/dst_box.area) - 1) < a):
                            a = abs((inter_area / dst_box.area) - 1)
                            print("a: " + str(a))
                            print(fi[1])
                            desig = int(fi[1])
                            found = True

                if not found:
                    print("no")
                    file_L.write(l_li[0] + ',' + l_li[1] + ',' + str(l_li[2].replace(')','(').split('(')[1]) + ',' + str(l_li[3].replace(')','(').split('(')[1]) + ',' + str(l_li[4].replace(')','(').split('(')[1]) + ',' + str(l_li[5].replace(')','(').split('(')[1]) + ',' + l_li[6])
                else:
                    print("yes")
                    file_R.write(l_li[0] + ',' + str(desig) + ',' + str(fi[2].replace(')','(').split('(')[1]) + ',' + str(fi[3].replace(')','(').split('(')[1]) + ',' + str(fi[4].replace(')','(').split('(')[1]) + ',' + str(fi[5].replace(')','(').split('(')[1]) + ',' + l_li[6])
                        
