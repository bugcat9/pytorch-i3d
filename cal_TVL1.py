import cv2
import os
import numpy as np
import glob
import multiprocessing

###### 使用frames帧进行 flow光流计算
video_root = 'video_list.txt'
root = 'frames'
out_root = 'flow'


def cal_for_frames(video_path):
    # print(video_path)
    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    if not os.path.exists(flow_path):
        os.mkdir(os.path.join(flow_path))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, str(i) + '_x.jpg'), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, str(i) + '_y.jpg'), flow[:, :, 1])


def process(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)


def extract_flow(root, out_root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    # dir_list = os.listdir(root)
    dir_list = []
    ### 读取txt中视频信息
    with open(video_root, 'r') as f:
        for id, line in enumerate(f):
            video_name = line.strip().split()
            preffix = video_name[0].split('.')[0]
            dir_list.append(preffix)

    pool = multiprocessing.Pool(processes=4)
    for dir_name in dir_list:
        video_path = os.path.join(root, dir_name)
        flow_path = os.path.join(out_root, dir_name)

        # flow = cal_for_frames(video_path)
        # save_flow(flow,flow_path)
        # print('save flow data: ',flow_path)
        # process(video_path,flow_path)
        pool.apply_async(process, args=(video_path, flow_path))

    pool.close()
    pool.join()


if __name__ == '__main__':
    extract_flow(root, out_root)
    print("finish!!!!!!!!!!!!!!!!!!")
