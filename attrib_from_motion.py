from ultralytics.data.annotator import auto_annotate
from ultralytics import YOLO
import os
import argparse
import threading
import cv2 as cv
import numpy as np
import math
from shapely.geometry import Polygon
import shapely
# from operator import itemgetter

# final attribute variables per object
# - color contrast
# - static proportion
# - dynamic proportion
# - distance



def downsample(imagebasedir, image_name_list, lines, sample_size):
    full_list = []
    single_downsample = []
    box_prop_list = []
    boxes = []
    dist_list = []
    id_tups = []
    looking = []

    id = 0
    # f = len(image_name_list)

    designation = 0
    for img in image_name_list:
        src = cv.imread(cv.samples.findFile(os.path.join(imagebasedir, img)))

        frame_no = int(img.replace('.jpg', '')) #line up with frames
        point_tuple_r = points_tuple_raw[frame_no].split(',')
        point_tuple = (point_tuple_r[2:])

        total_diag = math.dist([src.shape[1], 0], [0, src.shape[0]])
        
        for l1 in lines:
            if len(l1[0]) != 0:
                # for l_1 in l1:
                n = 1
                l = l1.split(',')
                # print("l: " + str(l))
                if int(l[0]) == int(img.replace(".jpg","")):
                    print("img: " + str(img.replace(".jpg","")) + "; l[0]: " + str(l[0]))
                    # ids = []
                    
                    # make sure getting the right list

                    # get box
                    i = src[int(float(l[3].replace(')','(').split('(')[1])):int(float(l[5].replace(')','(').split('(')[1])), int(float(l[2].replace(')','(').split('(')[1])):int(float(l[4].replace(')','(').split('(')[1]))]
                    center_shape = shapely.box(float(l[2].replace(')','(').split('(')[1]), float(l[3].replace(')','(').split('(')[1]), float(l[4].replace(')','(').split('(')[1]), float(l[5].replace(')','(').split('(')[1]))
                    boxes.append(center_shape)
                    center_area = center_shape.area
                    center = center_shape.centroid

                    
                    # static proportion
                    box_diag = math.dist((float(l[2].replace(')','(').split('(')[1]), float(l[3].replace(')','(').split('(')[1])), (float(l[4].replace(')','(').split('(')[1]), float(l[5].replace(')','(').split('(')[1])))
                    box_proportion = box_diag / total_diag
                    box_prop_list.append([img.replace(".jpg", ""), l[1], center_shape, box_proportion])

                    # get distance
                    if point_tuple[0] != '':
                        print("point tuple: " + str(point_tuple))
                        dist = abs(math.dist((float(point_tuple[0]), float(point_tuple[1])), (center.coords[0][0], center.coords[0][1])))
                    else:
                        dist = -1

                    dist_list.append((img.replace(".jpg", ""), l[1], dist))
                    # looking at object?
                    if point_tuple[0] != '':
                        poin = shapely.Point(point_tuple)
                        print("poin: " + str(poin))
                        if(center_shape.contains(poin)):
                            looking.append([img.replace(".jpg", ""), l[1], True])
                        else:
                            looking.append([img.replace(".jpg", ""), l[1], False])
                    else:
                        looking.append([img.replace(".jpg", ""), l[1], False])

                    # gaussian pyramid
                    single_downsample.append(i)
                    rows, cols, _channels = map(int, i.shape)
                    new_src = cv.pyrDown(i, dstsize=(cols // sample_size, rows // sample_size))
                    single_downsample.append(new_src)

                    full_list.append((img.replace(".jpg", ""), l[1], single_downsample.copy()))
                    single_downsample.clear()
                    
                
        
    return full_list, id_tups, dist_list, box_prop_list, looking
    


def contrast(imagebasedir, image_name_list, lines):

    all_contrasts = []
    list, id_tup, dist_list, box_prop_list, looking = downsample(imagebasedir, image_name_list, lines, 2)

    for frame, id, samples in list:
        all_samples_R = 0
        all_samples_G = 0
        all_samples_B = 0
        vsR = []
        vsG = []
        vsB = []
        i = 0
        for sample in samples:

            # separate channels
            B,G,R=cv.split(sample)
            vR = np.var(R, dtype=np.float64)
            vsR.append(vR)
            vG = np.var(G, dtype=np.float64)
            vsG.append(vG)
            vB = np.var(B, dtype=np.float64)
            vsB.append(vB)

            
            # compute minimum and maximum in 5x5 region using erode and dilate
            kernel = np.ones((40,40),np.uint8)
            min_R = cv.erode(R,kernel,iterations = 1)
            max_R = cv.dilate(R,kernel,iterations = 1)

            min_G = cv.erode(G,kernel,iterations = 1)
            max_G = cv.dilate(G,kernel,iterations = 1)

            min_B = cv.erode(B,kernel,iterations = 1)
            max_B = cv.dilate(B,kernel,iterations = 1)

            # convert min and max to floats
            min_R = min_R.astype(np.float64) 
            max_R = max_R.astype(np.float64) 

            min_G = min_G.astype(np.float64) 
            max_G = max_G.astype(np.float64) 

            min_B = min_B.astype(np.float64) 
            max_B = max_B.astype(np.float64) 

            # compute local contrast
            contrast_R = []
            contrast_G = []
            contrast_B = []

            contrast_R.append((max_R-min_R)/(max_R+min_R))
            contrast_G.append((max_G-min_G)/(max_G+min_G))
            contrast_B.append((max_B-min_B)/(max_B+min_B))

            contrast_R = np.nan_to_num(contrast_R)
            contrast_G = np.nan_to_num(contrast_G)
            contrast_B = np.nan_to_num(contrast_B)

            # get average across whole image
            average_contrast_R = np.mean(contrast_R)
            average_contrast_G = np.mean(contrast_G)
            average_contrast_B = np.mean(contrast_B)

            all_samples_R += average_contrast_R
            all_samples_G += average_contrast_G
            all_samples_B += average_contrast_B
        
        all_samples_R = all_samples_R / 2
        all_samples_G = all_samples_G / 2
        all_samples_B = all_samples_B / 2

        overall_contrast = (all_samples_R + all_samples_G + all_samples_B) / 3.0
        # method of identifying the appropriate contrast? and make sure not overwriting
        all_contrasts.append((frame, id, overall_contrast))

    return all_contrasts, id_tup, dist_list, box_prop_list, looking




# input: image folder, image name
# output: list of objects with frame, name, and box
def label(imagebasedir, im_path):
    model = YOLO("yolo12x.pt")  # load an official model

    # Predict with the model
    results = model(os.path.join(imagebasedir,im_path))  # predict on an image

    lines = []
    full_lines = []
    # Access the results
    for result in results:
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxy_2 = xyxy.tolist()
        boxes = result.boxes
        track_ids = boxes.id
        id = track_ids #.tolist()
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box
        i = 0
        for name in names:
            line = im_path.replace(".jpg","") + "," + str(name) + "," + str(int(xyxy_2[i][0])) + "," + str(int(xyxy_2[i][1])) + "," + str(int(xyxy_2[i][2])) + "," + str(int(xyxy_2[i][3]))
            print("line: " + str(line))
            lines.append(line)
            i += 1

    return lines


# input: image folder, image list
# output: the output of the contrast function
def split(imagebasedir, image_name_list, motion):

    lines = []
    image_list = []
    while image_name_list:

        if image_name_list:
            
            frame = image_name_list.pop()
            if frame.endswith(".jpg"):  
                image_list.append(frame)
                # label_image = label(imagebasedir, frame)
                # lines.append(label_image)
        else:
            break

    all_contrasts, static_list, dist_list, box_prop_list, looking = contrast(imagebasedir, image_list, motion)
    # print("all done")
    return all_contrasts, static_list, dist_list, box_prop_list, looking



if __name__ == '__main__':

    thread_list = []
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path",type=str,required=True)
    parser.add_argument("--tobii_csv", type=str, required=True)
    parser.add_argument("--motion_data",type=str,required=True)

    args = parser.parse_args()

    imagebasedir = args.video_path
    image_name_list = os.listdir(imagebasedir)
    image_name_list.sort()

    tobiibasedir = args.tobii_csv
    center_file = open(tobiibasedir, 'r')  
    p_label = center_file.readline()           
    points_tuple_raw = center_file.readlines()

    motionfile = args.motion_data
    motion_file = open(motionfile, 'r')
    m_header = motion_file.readline()
    motion = motion_file.readlines()

    name_list = list(image_name_list)

    # attribute lists
    all_contrasts, s_list, dist_list, bp_list, looking = split(imagebasedir, name_list, motion)

    # formatting and file input
    con_file = open(os.path.join(imagebasedir, "contrast.csv"), 'w')
    con_file.write("Frame,ID,Contrast\n")
    for con in all_contrasts:
        c_float = float(con[2])
        con_file.write(str(con[0]) + "," + str(con[1]) + "," + str(c_float) + "\n")

    bp_file = open(os.path.join(imagebasedir, "static_prop.csv"), 'w')
    bp_file.write("Frame,ID,Static Proportion\n")
    for bp in bp_list:
        bp_file.write(str(bp[0]) + "," + str(bp[1]) + "," + str(bp[3]) + "\n")

    dist_file = open(os.path.join(imagebasedir, "distance.csv"), 'w')
    dist_file.write("Frame,ID,Distance\n")
    for dist in dist_list:
        dist_file.write(str(dist[0]) + "," + str(dist[1]) + "," + str(dist[2]) + "\n")

    id_file = open(os.path.join(imagebasedir, "id.csv"), 'w')
    id_file.write("Frame,ID,Min x, Min y, Max x, Max y\n")
    for s in s_list:
        id_file.write(str(s[0][0]) + "," + str(s[0][1]) + "," + str(s[0][2].bounds[0]) + "," + str(s[0][2].bounds[1]) + "," + str(s[0][2].bounds[2])  + "," + str(s[0][2].bounds[3]) + "\n")
    
    look_file = open(os.path.join(imagebasedir, "look.csv"), 'w')
    look_file.write("Frame,ID,Looking\n")
    for look in looking:
        look_file.write(str(look[0]) + "," + str(look[1]) + "," + str(look[2]) + "\n")

    print("DONE: " + str(imagebasedir))