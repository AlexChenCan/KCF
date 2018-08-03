import os
import sys
import argparse
import time
import glob

from tracker import KCFtracker

import cv2
import numpy as np
import re

def get_box_list(groundturth_path,datafromat = False):
    boxs = []
    with open(groundturth_path) as file:
        for line in file:
            box = re.split("[,\t]",line.strip())
            # box = line.strip().split(',') # x,y,w,h
            boxs.append(box)
        
        boxs = np.array(boxs)
        boxs = boxs.astype('float64')
        if datafromat:
            # change to [y,x,h,w]
            boxs[:,[0,1]] = boxs[:,[1,0]]
            boxs[:,[2,3]] = boxs[:,[3,2]]
            # change to [y_center,x_center,h,w]
            boxs[:,0] = boxs[:,0] + boxs[:,2]/2
            boxs[:,1] = boxs[:,1] + boxs[:,3]/2
        boxs = boxs.astype('int')
    return boxs

def format_outline(pos,target_size):
    result_pos = [pos[0],pos[1],target_size[0],target_size[1]]
    result_pos = np.array(result_pos)
    result_pos[0] -= result_pos[2]/2
    result_pos[1] -= result_pos[3]/2
    result_pos[[0,1,2,3]] = result_pos[[1,0,3,2]]
    result_pos.astype('int')
    out_str = ','.join(result_pos)
    out_str += '\n'
    return out_str


def main(args):
    # dataset_path = args.dataset_descriptor
    dataset_path = 'E:/ImageData/TB-50/CarScale'
    save_result = args.save_result

    imageset_path = os.path.join(dataset_path,'img')
    image_path_list = glob.glob(os.path.join(imageset_path,'*.jpg'))
    image_path_list.sort()
    target_box_list = get_box_list(os.path.join(dataset_path,'groundtruth_rect.txt'),datafromat=True)

    imageset_size = len(image_path_list)

    pos_result = []
    target_size_result = []

    for i in range(imageset_size):
        # print(" i = " + str(i))
        img = cv2.imread(image_path_list[i])
        if i == 0:
            box = target_box_list[i]
            pos = (box[0],box[1])
            target_size = (box[2],box[3])

            tracker = KCFtracker(img,pos,target_size)
        else:
            pos, target_size = tracker.dectect(img)

        pos_result.append(pos)
        target_size_result.append(target_size)
                
    if save_result:
        result_file = os.path.join(dataset_path,"result_rect.txt")
        out_file = open(result_file,'w')
        for i in range(len(pos_result)):
            out_str = format_outline(pos_result[i],target_size_result[i])
            out_file.write(out_str)



def parse_argument(argv):
    parser = argparse.ArgumentParser()
    # parser.add_argument('dataset_descriptor',type=str,help = 'The directory of video and groundturth file')
    parser.add_argument('--resize',type=int,help='Resize img or not',default=1)
    parser.add_argument('--save_result',type=bool,help="save result or not",default=False)
    return parser.parse_args(argv)


if __name__ == "__main__":

    main(parse_argument(sys.argv[1:]))
