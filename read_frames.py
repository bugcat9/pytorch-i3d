import cv2
import numpy as np
import os
import multiprocessing

video_root = 'activity_net_train_exist_name_small.txt'
root = 'origi_videos'
out_root = 'frames'
suffix = '.jpg'

def save_image(root,vid_name,num,image):
    file_name = os.path.join(root,vid_name,str(num)+suffix)
    #print(file_name)
    cv2.imwrite(file_name,image)


def process(vid_path,preffix):
    videoCapture = cv2.VideoCapture(vid_path)

    i = 0
    while True:
        success, frame = videoCapture.read()
        if success:
            i = i + 1
            save_image(out_root, preffix, i-1, frame)
            #print('save image vid name: ', file_name, '; frame num: ', i)
        else:
            break

def main(root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    #path_list = os.listdir(root)
    path_list = []
    with open(video_root,'r') as f:
        for id,line in enumerate(f):
            video_name = line.strip().split()
            path_list.append(video_name[0]+'.mp4')

    pool = multiprocessing.Pool(processes=4)
    for file_name in path_list:
        path = os.path.join(root,file_name)
        preffix = file_name.split('.')[0]
        dir_name = os.path.join(out_root,preffix)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        pool.apply_async(process,args=(path,preffix))
        #process(path,preffix)

    pool.close()
    pool.join()
        
if __name__ == '__main__':
    main(root)