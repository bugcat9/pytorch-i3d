import os
import cv2


# 修改视频帧率为指定帧率，分辨率保持不变
def modify_video_frame_rate(videoPath, destFps):
    dir_name = os.path.dirname(videoPath)
    basename = os.path.basename(videoPath)
    video_name = basename[:basename.rfind('.')]
    # video_name = video_name + "moify_fps_rate"
    dest_dir = './videos'
    resultVideoPath = f'{dest_dir}/{video_name}.mp4'

    videoCapture = cv2.VideoCapture(videoPath)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if fps != destFps:
        frameSize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # 这里的VideoWriter_fourcc需要多测试，如果编码器不对则会提示报错，根据报错信息修改编码器即可
        videoWriter = cv2.VideoWriter(resultVideoPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), destFps, frameSize)

        i = 0;
        while True:
            success, frame = videoCapture.read()
            if success:
                i += 1
                # print('转换到第%d帧' % i)
                videoWriter.write(frame)
            else:
                print('帧率转换结束')
                break


if __name__ == '__main__':
    modify_video_frame_rate('./origi_videos/video_test_0000004.mp4', 25)
