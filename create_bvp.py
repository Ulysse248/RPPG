import cv2
import numpy as np

def read_video(video_file):
    """Reads a video file, returns frames(T,H,W,3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)

def create_fake_bvp(path):
    vid = read_video(path)

    with open(path[:-4]+ '.txt', 'w') as f:
        base_value = 80
        for frame in vid:
            p = np.random.randint(-5, 5)
            f.write(str(base_value + p))
            f.write('\n')