import os
import cv2 as cv


video_path = '/dark/Databases/BosphorusSignV2_final/videos_640x360'
output_video_path = '/dark/Databases/BosphorusSignV2_final/videos_360x360'
output_frame_path = '/dark/Databases/BosphorusSignV2_final/frames_360x360'
scaled_video_path = '/dark/Databases/BosphorusSignV2_final/videos_112x112'
scaled_frame_path = '/dark/Databases/BosphorusSignV2_final/frames_112x112'

os.environ['PATH'] += os.pathsep + '/raid/users/oozdemir/opt/local/include'
os.environ['LD_LIBRARY_PATH'] = os.pathsep + '/raid/users/oozdemir/opt/local/lib'
os.environ['PATH'] += os.pathsep + '/raid/users/oozdemir/opt/local/bin'
FFMPEG_PATH = '/raid/users/oozdemir/opt/local/bin/ffmpeg'

# FFMPEG_PATH = 'ffmpeg'
# video_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\videos_640x360'
# output_video_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\videos_360x360'
# output_frame_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\frames_360x360'
# scaled_video_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\videos_112x112'
# scaled_frame_path = 'D:\\Databases\\BosphorusSignV2\\Toydata\\frames_112x112'

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
        # os.makedirs(output_frame)

        output_scaled_video = os.path.join(scaled_video_path, cls, vid)
        output_scaled_frame = os.path.join(scaled_frame_path, cls, os.path.splitext(vid)[0])
        os.makedirs(output_scaled_frame)

        # os.system(FFMPEG_PATH + ' -loglevel panic -i ' + input_video + ' -filter:v "crop=360:360:180:0" -c:a copy ' + output_video)
        #
        # vid_cap = cv.VideoCapture(output_video)
        # frame_idx = 0
        # while vid_cap.isOpened():
        #     ret, frame = vid_cap.read()
        #     if ret == False:
        #         break
        #     cv.imwrite(os.path.join(output_frame, '%06d.jpg' % frame_idx), frame)
        #     frame_idx += 1
        # vid_cap.release()

        os.system(FFMPEG_PATH + ' -loglevel panic -i ' + output_video + ' -filter:v "scale=112:112" -c:a copy ' + output_scaled_video)

        vid_cap = cv.VideoCapture(output_scaled_video)
        frame_idx = 0
        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if ret == False:
                break
            cv.imwrite(os.path.join(output_scaled_frame, '%06d.jpg' % frame_idx), frame)
            frame_idx += 1
        vid_cap.release()