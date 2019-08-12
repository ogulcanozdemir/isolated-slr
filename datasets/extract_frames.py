import os
import cv2 as cv


video_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\videos_640x360'
output_video_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\videos_360x360'
output_frame_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\frames_360x360'
scaled_video_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\videos_150x150'
scaled_frame_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\frames_150x150'

class_dir = sorted(os.listdir(video_path))
for cls in class_dir:
    video_dir = sorted(os.listdir(os.path.join(video_path, cls)))

    output_class_dir = os.path.join(output_video_path, cls)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    output_class_dir = os.path.join(output_frame_path, cls)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    output_class_dir = os.path.join(scaled_video_path, cls)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    output_class_dir = os.path.join(scaled_frame_path, cls)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    for vid in video_dir:
        print(cls, vid)
        input_video = os.path.join(video_path, cls, vid)
        output_video = os.path.join(output_video_path, cls, vid)
        output_frame = os.path.join(output_frame_path, cls, os.path.splitext(vid)[0])
        os.makedirs(output_frame)

        output_scaled_video = os.path.join(scaled_video_path, cls, vid)
        output_scaled_frame = os.path.join(scaled_frame_path, cls, os.path.splitext(vid)[0])
        os.makedirs(output_scaled_frame)

        os.system('ffmpeg -loglevel panic -i ' + input_video + ' -filter:v "crop=360:360:180:0" -c:a copy ' + output_video)

        vid_cap = cv.VideoCapture(output_video)
        frame_idx = 0
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if ret == False:
                break
            cv.imwrite(os.path.join(output_frame, '%06d.jpg' % frame_idx), frame)
            frame_idx += 1
        vid_cap.release()

        # os.system('ffmpeg -loglevel panic -i ' + output_video + ' -startnumber 0' + output_frame + '/%06d.jpg')
        os.system('ffmpeg -loglevel panic -i ' + output_video + ' -filter:v "scale=150:150" -c:a copy ' + output_scaled_video)

        vid_cap = cv.VideoCapture(output_scaled_video)
        frame_idx = 0
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if ret == False:
                break
            cv.imwrite(os.path.join(output_scaled_frame, '%06d.jpg' % frame_idx), frame)
            frame_idx += 1
        vid_cap.release()

        # os.system('ffmpeg -loglevel panic -i ' + output_scaled_video + ' -startnumber 0' + output_scaled_frame + '/%06d.jpg')